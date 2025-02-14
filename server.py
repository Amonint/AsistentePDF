from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import tiktoken
import os

app = Flask(__name__)
CORS(app)

# Configuración de APIs
GOOGLE_API_KEY = "AIzaSyA0iPQ5Y3up5RVtB7bfax8Yj_A4UwQzSZM"
PINECONE_API_KEY = "pcsk_3p1ned_FNyRBj8sYrAoNiKgzH8CnJcYHEcyRzBxj25GNuUHavhBXxjcAwDA22DBpi34Vpw"

genai.configure(api_key=GOOGLE_API_KEY)

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
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10
    )
    docs = text_splitter.split_documents(data)
    
    processed_docs = []
    for doc in docs:
        truncated_content = truncate_text(doc.page_content, max_tokens=96)
        if truncated_content:
            doc.page_content = truncated_content
            processed_docs.append(doc)
    
    return processed_docs

def get_gemini_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = truncate_text(context, max_tokens=500)
        prompt = f"""
        Contexto: {context}
        Pregunta: {query}
        Por favor, responde la pregunta basándote únicamente en la información proporcionada en el contexto anterior.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al procesar la respuesta con Gemini: {str(e)}"

# Variables globales para mantener las instancias
pc = None
index = None
processed_docs = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global pc, index, processed_docs
    try:
        pc, index = initialize_pinecone()
        processed_docs = process_documents()
        
        if not processed_docs:
            return jsonify({'error': 'No se encontraron documentos válidos'}), 400
            
        return jsonify({'message': 'Sistema inicializado correctamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global pc, index
    
    if not pc or not index:
        return jsonify({'error': 'Sistema no inicializado'}), 400
    
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        query_text = truncate_text(user_input, max_tokens=96)
        
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query"}
        )
        
        results = index.query(
            namespace="example-namespace",
            vector=query_embedding[0]['values'],
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        
        context = " ".join([match['metadata']['text'] for match in results['matches']])
        response = get_gemini_response(user_input, context)
        
        return jsonify({
            'response': response,
            'sources': [match['metadata']['text'] for match in results['matches']]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('pdfs'):
        os.makedirs('pdfs')
    app.run(debug=True)