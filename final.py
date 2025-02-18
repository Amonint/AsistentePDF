from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import tiktoken
import os
import sys
from pathlib import Path
import base64
import re
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import json
import concurrent.futures 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyA0iPQ5Y3up5RVtB7bfax8Yj_A4UwQzSZM"
PINECONE_API_KEY = "pcsk_3p1ned_FNyRBj8sYrAoNiKgzH8CnJcYHEcyRzBxj25GNuUHavhBXxjcAwDA22DBpi34Vpw"

if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
    logger.error("Missing required environment variables")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

@dataclass
class Document:
    page_content: str
    metadata: dict

class PDFProcessor:
    def __init__(self, max_workers: int = 4):
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_workers = max_workers
    
    @staticmethod
    def validate_pdf(file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception:
            return False

    def semantic_chunk_with_gemini(self, text: str, file_path: Path) -> List[Document]:
        prompt = """
        Actúa como un experto en procesamiento de documentos legales. Divide el siguiente texto en chunks semánticos siguiendo estos criterios:

        1. Mantén la coherencia semántica de cada chunk
        2. Preserva el contexto legal completo
        3. Identifica y mantén juntas las secciones relacionadas
        4. Normaliza el formato y limpia el texto
        5. Extrae metadata relevante de cada chunk

        Formato de salida requerido (JSON):
        {
            "chunks": [
                {
                    "content": "texto del chunk",
                    "metadata": {
                        "section_type": "tipo de sección",
                        "legal_references": ["referencias"],
                        "key_entities": ["entidades"],
                        "semantic_category": "categoría"
                    }
                }
            ]
        }
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt, text])
            chunks_data = json.loads(response.text)
            
            return [
                Document(
                    page_content=chunk['content'],
                    metadata={
                        **chunk['metadata'],
                        'source': str(file_path),
                        'filename': file_path.name,
                        'chunk_index': idx,
                    }
                )
                for idx, chunk in enumerate(chunks_data['chunks'])
            ]
        except Exception as e:
            logger.error(f"Error en chunking semántico: {str(e)}")
            # Fallback al chunking tradicional si falla el semántico
            return self.traditional_chunk(text, file_path)

    def traditional_chunk(self, text: str, file_path: Path) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=len,
            separators=[r"\n\n", r"\n(?=\S)", r"\.\s+", r";\s+", r",\s+", r"\s+", ""]
        )
        chunks = splitter.split_text(text)
        
        return [
            Document(
                page_content=chunk,
                metadata={
                    'source': str(file_path),
                    'filename': file_path.name,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            ) for i, chunk in enumerate(chunks)
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_text_with_gemini(self, file_data: bytes) -> Optional[str]:
        try:
            base64_pdf = base64.b64encode(file_data).decode('utf-8')
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = """
            Extrae el texto del PDF manteniendo la estructura semántica y realiza una limpieza inicial:
            1. Elimina headers y footers repetitivos
            2. Normaliza espacios y saltos de línea
            3. Preserva la estructura de secciones
            4. Mantén el formato de referencias legales
            """
            
            response = model.generate_content([prompt, {
                'mime_type': 'application/pdf',
                'data': base64_pdf
            }])
            
            if response.text:
                return self.clean_text(response.text)
            return None
        except Exception as e:
            logger.error(f"Error en extracción Gemini: {str(e)}")
            return None

    @staticmethod
    def clean_text(text: str) -> str:
        clean_patterns = [
            (r'^\s*\d+\.\s*', '', re.MULTILINE),
            (r'\n{3,}', '\n\n'),
            (r'\s{2,}', ' '),
            (r'[\x00-\x1F\x7F-\x9F]', '')
        ]
        for pattern, repl, *flags in clean_patterns:
            text = re.sub(pattern, repl, text, flags=flags[0] if flags else 0)
        return text.strip()

    def process_documents(self, pdf_dir: Path = Path("pdfs")) -> List[Document]:
        if not pdf_dir.exists():
            pdf_dir.mkdir(exist_ok=True)
            return []

        all_documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_document, pdf_path): pdf_path
                for pdf_path in pdf_dir.glob("*.pdf")
                if self.validate_pdf(pdf_path)
            }
            
            for future in concurrent.futures.as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    docs = future.result()
                    if docs:
                        all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error procesando {pdf_path}: {str(e)}")
        
        return all_documents

    def process_single_document(self, pdf_path: Path) -> List[Document]:
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            if extracted_text := self.extract_text_with_gemini(pdf_data):
                return self.semantic_chunk_with_gemini(extracted_text, pdf_path)
            return []
        except Exception as e:
            logger.error(f"Error procesando documento {pdf_path}: {str(e)}")
            return []

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = "multilingual-e5-large-v2"
        self.dimension = 1024
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert_embeddings(self, documents: List[Document], batch_size: int = 128) -> None:
        index = self.pc.Index(self.index_name)
        
        with ThreadPoolExecutor() as executor:
            # Procesar embeddings en paralelo
            future_embeddings = {
                executor.submit(self.embedding_model.encode, [doc.page_content]): doc
                for doc in documents
            }
            
            vectors = []
            for future in concurrent.futures.as_completed(future_embeddings):
                doc = future_embeddings[future]
                try:
                    embedding = future.result()[0]
                    vectors.append({
                        'id': f"doc_{doc.metadata['filename']}_{doc.metadata['chunk_index']}",
                        'values': embedding.tolist(),
                        'metadata': {**doc.metadata, 'text': doc.page_content}
                    })
                except Exception as e:
                    logger.error(f"Error generando embedding: {str(e)}")
            
            # Upsert en batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch, namespace="legal-docs")
                logger.info(f"Batch {i//batch_size + 1} subido: {len(batch)} documentos")

class QAEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.pinecone = PineconeManager()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def query(self, question: str, top_k: int = 5) -> dict:
        index = self.pinecone.pc.Index(self.pinecone.index_name)
        query_embedding = self.embedding_model.encode([question])[0].tolist()
        
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="legal-docs"
        )
    
    def generate_answer(self, question: str, context: str) -> str:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Eres un asistente legal especializado en documentos jurídicos. Responde basándote exclusivamente en el contexto proporcionado.
        
        Contexto:
        {context}
        
        Instrucciones:
        1. Respuesta precisa y técnica
        2. Formato legal formal
        3. Citar fuentes de los fragmentos
        4. Si no hay información relevante, indica 'No consta en documentos'
        5. Máximo 3 párrafos
        
        Pregunta: {question}
        """
        
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    processor = PDFProcessor()
    qa_engine = QAEngine()
    
    logger.info("Procesando documentos...")
    documents = processor.process_documents()
    
    if not documents:
        logger.warning("No se encontraron documentos válidos")
        return
    
    logger.info("Subiendo embeddings a Pinecone...")
    qa_engine.pinecone.upsert_embeddings(documents)
    
    logger.info("Sistema listo para consultas")
    while True:
        try:
            query = input("\nConsulta: ").strip()
            if query.lower() in ('exit', 'salir'):
                break
            
            results = qa_engine.query(query)
            if not results.matches:
                print("No se encontraron resultados relevantes")
                continue
            
            context = "\n".join(
                f"[Fuente {i+1}: {m.metadata['filename']}]\n{m.metadata['text']}"
                for i, m in enumerate(results.matches)
            )
            
            answer = qa_engine.generate_answer(query, context)
            print(f"\nRespuesta:\n{answer}\n\nFuentes:\n{context}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()