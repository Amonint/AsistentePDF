from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import tiktoken
import os
import sys

# Configurar Gemini API
GOOGLE_API_KEY = "AIzaSyA0iPQ5Y3up5RVtB7bfax8Yj_A4UwQzSZM"
genai.configure(api_key=GOOGLE_API_KEY)

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

def count_tokens(text, model="cl100k_base"):
    tokenizer = tiktoken.get_encoding(model)
    return len(tokenizer.encode(text))

def truncate_text(text, max_tokens=96):
    if not text:
        return ""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def initialize_pinecone():
    api_key = "pcsk_3p1ned_FNyRBj8sYrAoNiKgzH8CnJcYHEcyRzBxj25GNuUHavhBXxjcAwDA22DBpi34Vpw"
    pc = Pinecone(api_key=api_key)
    
    index_name = "multilingual-e5-large"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    return pc, pc.Index(index_name)

def process_documents():
    loader = PyPDFDirectoryLoader("pdfs")
    data = loader.load()
    
    # Usar un tamaño de chunk más pequeño para evitar exceder el límite
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Reducido para asegurar que no exceda 96 tokens
        chunk_overlap=10
    )
    docs = text_splitter.split_documents(data)
    print(f"Se han dividido {len(docs)} documentos.")
    
    # Truncar cada documento a 96 tokens
    processed_docs = []
    for doc in docs:
        truncated_content = truncate_text(doc.page_content, max_tokens=96)
        if truncated_content:  # Solo agregar si hay contenido después de truncar
            doc.page_content = truncated_content
            processed_docs.append(doc)
    
    return processed_docs

def create_embeddings_and_upload(pc, index, docs):
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        
        # Asegurar que cada texto está dentro del límite
        texts = [doc.page_content for doc in batch_docs]
        texts = [text for text in texts if text]  # Eliminar textos vacíos
        
        if not texts:
            continue
        
        try:
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)

            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                vectors.append({
                    "id": f"doc_{i+j}",
                    "values": embedding.tolist(),
                    "metadata": {'text': doc.page_content}
                })

            if vectors:
                index.upsert(vectors=vectors, namespace="example-namespace")

            print(f"Procesado lote {i//batch_size + 1} de {(len(docs) + batch_size - 1)//batch_size}")

        except Exception as e:
            print(f"Error en lote {i//batch_size + 1}: {str(e)}")
            continue

def get_gemini_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Truncar el contexto si es muy largo
        context = truncate_text(context, max_tokens=500)  # Usar un límite mayor para Gemini
        prompt = f"""
        Contexto: {context}
        Pregunta: {query}
        Por favor, responde la pregunta basándote únicamente en la información proporcionada en el contexto anterior.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al procesar la respuesta con Gemini: {str(e)}"

def main():
    print("Inicializando el sistema...")
    pc, index = initialize_pinecone()
    docs = process_documents()
    
    if not docs:
        print("No se encontraron documentos válidos para procesar.")
        return
        
    create_embeddings_and_upload(pc, index, docs)
    
    print("\n¡Bienvenido al chatbot de documentos PDF! (Escribe 'exit' para salir)")
    
    while True:
        user_input = input("\nTu pregunta: ").strip()
        
        if user_input.lower() == 'exit':
            print('¡Hasta luego!')
            sys.exit()
        
        if user_input == '':
            continue
        
        try:
            query_text = truncate_text(user_input, max_tokens=96)
            
            query_embedding = embedding_model.encode([query_text])[0].tolist()
            
            results = index.query(
                namespace="example-namespace",
                vector=query_embedding,
                top_k=3,
                include_values=False,
                include_metadata=True
            )
            
            context = " ".join([match['metadata']['text'] for match in results['matches']])
            response = get_gemini_response(user_input, context)
            
            print("\nRespuesta:", response)
            print("\nFuentes utilizadas:")
            for i, match in enumerate(results['matches'], 1):
                print(f"\nFuente {i}:")
                print(match['metadata']['text'])
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Por favor, intenta de nuevo.")

if __name__ == "__main__":
    main()
