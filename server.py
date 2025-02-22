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
from pdfminer.high_level import extract_text as pdfminer_extract
from typing import List, Optional

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de variables de entorno
GOOGLE_API_KEY = "AIzaSyA0iPQ5Y3up5RVtB7bfax8Yj_A4UwQzSZM"
PINECONE_API_KEY = "pcsk_3p1ned_FNyRBj8sYrAoNiKgzH8CnJcYHEcyRzBxj25GNuUHavhBXxjcAwDA22DBpi34Vpw"

# Validación de claves
if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
    logger.error("Missing required environment variables")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

@dataclass
class Document:
    page_content: str
    metadata: dict

class PDFProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @staticmethod
    def validate_pdf(file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception:
            return False
    
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_text_with_gemini(self, file_data: bytes) -> Optional[str]:
        try:
            base64_pdf = base64.b64encode(file_data).decode('utf-8')
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = """
                Actúa como un experto en procesamiento de documentos legales. Realiza la extracción de texto del PDF siguiendo estrictamente estos criterios:

                1. **Preservación estructural**:
                - Mantén el orden original de todas las secciones
                - Conserva encabezados, títulos y subtítulos en su posición exacta
                - Respeta saltos de línea y párrafos originales

                2. **Formateo especial**:
                - Preserva listas numeradas/viñetas como en el original
                - Mantener tablas usando pipes (|) con alineación
                - Conservar sangrías y espaciado significativo

                3. **Integridad del contenido**:
                - NO omitas texto, incluso si parece repetido
                - Incluye notas al pie y anotaciones marginales
                - Conservar sellos, firmas digitalizadas como [FIRMA], [SELLO]

                4. **Elementos técnicos**:
                - Mantener números de página como [PÁG. 23]
                - Preservar formatos de fechas (ej: 25 de julio de 2023 → 25/07/2023)
                - Conservar referencias legales (ej: Art. 15.3 de la LECiv)

                5. **Prohibiciones estrictas**:
                - NO interpretes, parafrasees o resumas
                - NO agregues numeraciones que no existan
                - NO corrijas errores ortográficos
                - NO unir líneas rotas automáticamente

                6. **Elementos especiales**:
                - Marcar documentos adjuntos como [ANEXO: nombre_doc.pdf]
                - Conservar formatos de expedientes (ej: EXP 2023-0456)
                - Mantener referencias a artículos legales completas

                Formato de salida requerido:
                - Codificación UTF-8
                - Saltos de línea UNIX (LF)
                - Marcar texto ilegible como [ILEGIBLE]
                - Usar comillas latinas (« ») para citas

                Respuesta requerida:
                Solo el texto extraído en crudo sin comentarios adicionales.
                """
            
            response = model.generate_content([prompt, {
                'mime_type': 'application/pdf',
                'data': base64_pdf
            }])
            
            if response.text:
                cleaned_text = self.clean_text(response.text)
                self._validate_extraction(cleaned_text)
                return cleaned_text
            return None
        except Exception as e:
            logger.error(f"Error en extracción Gemini: {str(e)}")
            return None
    
    def _validate_extraction(self, text: str) -> None:
        if len(text) < 100:
            raise ValueError("Texto extraído insuficiente")
        if not re.search(r'\w{3,}', text):
            raise ValueError("Texto no contiene contenido válido")

    def split_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[r"\n\n", r"\n(?=\S)", r"\.\s+", r";\s+", r",\s+", r"\s+", ""]
        )
        return splitter.split_text(text)
    
    def split_text_with_verification(self, text, chunk_size=300, overlap=50):
        """
        Divide el texto en chunks con verificación de integridad.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[r"\n\n", r"\n(?=\S)", r"\.\s+", r";\s+", r",\s+", r"\s+", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Verificación de chunks
        total_chars = len(text)
        chars_in_chunks = sum(len(chunk) for chunk in chunks)
        overlap_chars = (len(chunks) - 1) * overlap if len(chunks) > 1 else 0
        
        print(f"\nEstadísticas de división del texto:")
        print(f"Total de caracteres en texto original: {total_chars}")
        print(f"Total de caracteres en chunks: {chars_in_chunks}")
        print(f"Número de chunks generados: {len(chunks)}")
        print(f"Tamaño promedio de chunk: {chars_in_chunks/len(chunks) if chunks else 0:.2f} caracteres")
        
        # Verificar cobertura del texto
        coverage = (chars_in_chunks - overlap_chars) / total_chars if total_chars > 0 else 0
        print(f"Cobertura del texto: {coverage*100:.2f}%")
        
        return chunks

    def process_documents(self, pdf_dir: Path = Path("pdfs")) -> List[Document]:
        processed_docs = []
        if not pdf_dir.exists():
            pdf_dir.mkdir(exist_ok=True)
            return processed_docs
        
        for pdf_path in pdf_dir.glob("*.pdf"):
            if not self.validate_pdf(pdf_path):
                logger.warning(f"Archivo inválido: {pdf_path}")
                continue
            
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                
                if extracted_text := self.extract_text_with_gemini(pdf_data):
                    chunks = self.split_text_with_verification(extracted_text)
                    processed_docs.extend([
                        Document(
                            page_content=chunk,
                            metadata={
                                'source': str(pdf_path),
                                'filename': pdf_path.name,
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            }
                        ) for i, chunk in enumerate(chunks)
                    ])
            
            except Exception as e:
                logger.error(f"Error procesando {pdf_path}: {str(e)}")
        
        return processed_docs

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
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            vectors = [{
                'id': f"doc_{doc.metadata['filename']}_{doc.metadata['chunk_index']}",
                'values': emb.tolist(),
                'metadata': {**doc.metadata, 'text': doc.page_content}
            } for doc, emb in zip(batch, embeddings)]
            
            index.upsert(vectors=vectors, namespace="legal-docs")
            logger.info(f"Batch {i//batch_size + 1} subido: {len(vectors)} documentos")

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