import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai


# Cargar variables de entorno
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuración del modelo Gemini
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

def get_gemini_model():
    """Inicializa y retorna el modelo Gemini Pro Vision para procesar PDFs"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error al inicializar el modelo Gemini: {str(e)}")
        return None
def clean_extracted_text(text):
    """Limpia y normaliza el texto extraído"""
    if not text:
        return ""
    
    # Eliminar caracteres no imprimibles
    text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar líneas vacías múltiples
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Asegurar que hay contenido sustancial
    if len(text.strip()) < 10:  # Ajusta este número según tus necesidades
        return ""
        
    return text.strip()
def process_pdf_with_gemini(pdf_file):
    """Procesa el PDF con Gemini Pro Vision usando el método de carga de archivos"""
    try:
        # Guardar temporalmente el archivo
        temp_path = f"temp_{pdf_file.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        try:
            # Cargar el archivo usando el método recomendado por Google
            uploaded_file = genai.upload_file(temp_path)
            
            # Obtener el modelo
            model = get_gemini_model()
            if not model:
                return None
            
            # Instrucciones mejoradas para el procesamiento de PDFs escaneados
            prompt = """
            Analiza este documento PDF y extrae todo su contenido textual. Es muy importante que:

            1. Si el documento es un PDF escaneado o contiene imágenes:
               - Realiza OCR detallado de TODO el texto visible
               - No omitas ningún texto, por pequeño que sea
               - Incluye números, fechas y datos específicos
               - Mantén el formato original (títulos, subtítulos, párrafos)

            2. Estructura el contenido de forma clara:
               - Respeta la jerarquía del documento
               - Preserva la separación entre secciones
               - Mantén el orden original del texto

            3. Asegúrate de incluir:
               - Texto en márgenes o notas al pie
               - Encabezados y pies de página
               - Tablas y su contenido
               - Listas y enumeraciones

            Extrae absolutamente todo el contenido textual visible en el documento.
            """
            
            # Procesar con Gemini usando el archivo cargado
            response = model.generate_content(
                contents=[prompt, uploaded_file],
                generation_config=MODEL_CONFIG
            )
            
            # Limpiar y validar el texto extraído
            extracted_text = clean_extracted_text(response.text)
            
            # Validar que se extrajo contenido útil
            if not extracted_text:
                st.warning(f"⚠️ El texto extraído de {pdf_file.name} está vacío después de la limpieza")
                return None
                
            # Mostrar información sobre el texto extraído
            st.info(f"📄 Texto extraído de {pdf_file.name}: {len(extracted_text)} caracteres")
            
            return extracted_text
            
        finally:
            # Limpiar el archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        return None


def get_pdf_text(pdf_docs):
    """Procesa múltiples PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            st.info(f"Procesando {pdf.name} con Gemini Pro Vision...")
            
            # Verificar el tamaño del archivo
            file_size = len(pdf.getvalue())
            size_mb = file_size / (1024 * 1024)
            st.info(f"Tamaño del archivo: {size_mb:.2f} MB")
            
            # Procesar PDF con Gemini
            pdf_text = process_pdf_with_gemini(pdf)
            
            if pdf_text:
                text += f"\n--- Documento: {pdf.name} ---\n{pdf_text}\n"
                st.success(f"✅ {pdf.name} procesado exitosamente")
            else:
                st.warning(f"⚠️ No se pudo extraer texto de {pdf.name}")
                
        except Exception as e:
            st.error(f"❌ Error procesando {pdf.name}: {str(e)}")
            continue
    
    if not text.strip():
        st.error("No se pudo extraer texto de ninguno de los PDFs.")
        return None
    
    return text

def get_text_chunks(text):
    """Divide el texto en fragmentos con configuración optimizada"""
    if not text:
        return None
    
    try:
        # Configuración más granular para el divisor de texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Chunks más pequeños para mejor procesamiento
            chunk_overlap=50,  # Menor superposición
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Separadores más específicos
        )
        
        chunks = text_splitter.split_text(text)
        
        # Validar los chunks
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filtrar chunks muy pequeños
        
        if not valid_chunks:
            st.error("No se pudieron generar fragmentos de texto válidos.")
            return None
        
        # Mostrar información sobre los chunks
        st.info(f"📑 Chunks generados: {len(valid_chunks)}")
        
        return valid_chunks
        
    except Exception as e:
        st.error(f"Error al dividir el texto: {str(e)}")
        return None

def get_vector_store(text_chunks):
    """Genera los embeddings y almacena en FAISS con validación mejorada"""
    if not text_chunks:
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Validar cada chunk antes de crear embeddings
        valid_chunks = []
        for i, chunk in enumerate(text_chunks):
            try:
                # Intentar crear un embedding de prueba
                _ = embeddings.embed_query(chunk)
                valid_chunks.append(chunk)
            except Exception as e:
                st.warning(f"Chunk {i} ignorado: no se pudo crear embedding")
                continue
        
        if not valid_chunks:
            st.error("No se pudieron crear embeddings válidos.")
            return None
        
        # Crear el vector store con los chunks válidos
        vector_store = FAISS.from_texts(valid_chunks, embedding=embeddings)
        
        # Crear directorio para el índice si no existe
        os.makedirs("faiss_index", exist_ok=True)
        
        # Guardar el índice y los datos
        vector_store.save_local("faiss_index")
        
        # Mostrar información sobre el vector store
        st.info(f"💾 Vector store creado con {len(valid_chunks)} chunks")
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error al crear el vector store: {str(e)}")
        return None

def get_qa_chain():
    """Configura el modelo de IA para responder preguntas"""
    prompt_template = """
    Responde la pregunta de la manera más detallada posible usando el contexto proporcionado.
    Si la respuesta no está en el contexto, di "No encontré esa información en el documento".
    No inventes información que no esté en el contexto.
    
    Contexto:
    {context}
    
    Pregunta:
    {question}
    
    Respuesta:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    return PROMPT, model

def user_input(user_question):
    """Maneja las preguntas del usuario"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Cargar el índice FAISS con la opción de deserialización segura
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Obtener el prompt y el modelo
        prompt, llm = get_qa_chain()
        
        # Crear la cadena de QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Obtener la respuesta
        response = qa_chain.invoke({"query": user_question})
        
        if response and "result" in response:
            st.write("Respuesta:", response["result"])
        else:
            st.error("No se pudo generar una respuesta.")

    except Exception as e:
        st.error(f"Error al procesar la pregunta: {str(e)}")

def main():
    """Aplicación principal"""
    st.set_page_config(page_title="Chat PDF con Gemini 1.5")
    st.header("Chat con PDFs usando Gemini 1.5 Flash 📚")
    
    # Agregar advertencia de seguridad
    if not os.path.exists("faiss_index"):
        st.warning("""
        ⚠️ Nota de seguridad: Esta aplicación utiliza almacenamiento local para los índices FAISS. 
        Solo procese documentos de fuentes confiables, ya que el sistema necesita deserializar 
        datos almacenados localmente.
        """)
    
    with st.sidebar:
        st.title("📁 Menú")
        pdf_docs = st.file_uploader(
            "Sube tus archivos PDF y haz clic en Procesar",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("🚀 Procesar PDFs"):
            if not pdf_docs:
                st.error("Por favor, sube al menos un archivo PDF.")
                return
                
            with st.spinner("Procesando documentos..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.success("¡Procesamiento completado! 🎉")
                            st.info("Ahora puedes hacer preguntas sobre el contenido de los PDFs")

    # Área de chat
    st.subheader("💭 Pregunta sobre tus documentos")
    user_question = st.text_input("Escribe tu pregunta aquí:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()