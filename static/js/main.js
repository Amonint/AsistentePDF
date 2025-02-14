document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const initButton = document.getElementById('init-button');
    const chatWindow = document.getElementById('chat-window');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const fileList = document.getElementById('file-list');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const sourcesSection = document.getElementById('sources-section');
    const sourcesList = document.getElementById('sources-list');

    let isInitialized = false;

    // Initialize system
    async function initializeSystem() {
        try {
            showLoading('Inicializando sistema...');
            const response = await fetch('/initialize', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Error al inicializar el sistema');
            }

            isInitialized = true;
            messageInput.disabled = false;
            sendButton.disabled = false;
            initButton.style.display = 'none';
            showSuccess('Sistema inicializado correctamente');
            await listPDFFiles();
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading();
        }
    }

    // Function to list PDF files from Firebase Storage
    async function listPDFFiles() {
        const { storage, ref, listAll, getDownloadURL, deleteObject } = window.firebaseStorage;
        const pdfRef = ref(storage, 'pdfs');
        
        try {
            const result = await listAll(pdfRef);
            fileList.innerHTML = '';
            
            for (const itemRef of result.items) {
                const url = await getDownloadURL(itemRef);
                const li = document.createElement('li');
                li.className = 'flex items-center justify-between p-2 hover:bg-gray-100 rounded';
                
                const fileName = itemRef.name;
                li.innerHTML = `
                    <span class="text-blue-600">${fileName}</span>
                    <div class="flex gap-2">
                        <a href="${url}" target="_blank" class="text-sm text-gray-500 hover:text-gray-700">Ver</a>
                        <button class="delete-file text-sm text-red-500 hover:text-red-700">Eliminar</button>
                    </div>
                `;
                
                // Add delete functionality
                const deleteButton = li.querySelector('.delete-file');
                deleteButton.addEventListener('click', async () => {
                    try {
                        await deleteObject(ref(storage, `pdfs/${fileName}`));
                        await listPDFFiles();
                        showSuccess('Archivo eliminado correctamente');
                    } catch (error) {
                        showError('Error al eliminar el archivo');
                    }
                });
                
                fileList.appendChild(li);
            }
        } catch (error) {
            console.error("Error listing files:", error);
            showError("Error al listar los archivos PDF");
        }
    }

    // Function to upload PDF file
    async function uploadPDF(file) {
        const { storage, ref, uploadBytes } = window.firebaseStorage;
        
        if (!file || !file.type.includes('pdf')) {
            showError("Por favor, selecciona un archivo PDF válido");
            return;
        }

        const pdfRef = ref(storage, `pdfs/${file.name}`);
        try {
            showLoading('Subiendo archivo...');
            await uploadBytes(pdfRef, file);
            await listPDFFiles();
            showSuccess("Archivo subido exitosamente");
        } catch (error) {
            console.error("Error uploading file:", error);
            showError("Error al subir el archivo");
        } finally {
            hideLoading();
        }
    }

    // Function to send message to backend
    async function sendMessage(message) {
        try {
            showLoading('Procesando mensaje...');
            appendMessage('user', message);

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }

            const data = await response.json();
            appendMessage('bot', data.response);
            updateSources(data.sources);
        } catch (error) {
            console.error("Error sending message:", error);
            showError("Error al procesar tu mensaje");
        } finally {
            hideLoading();
        }
    }

    // Function to append message to chat window
    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender === 'user' ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = text;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Function to update sources
    function updateSources(sources) {
        if (!sources || sources.length === 0) {
            sourcesSection.classList.add('hidden');
            return;
        }

        sourcesSection.classList.remove('hidden');
        sourcesList.innerHTML = '';
        sources.forEach((source, index) => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'p-2 bg-gray-50 rounded';
            sourceDiv.textContent = `Fuente ${index + 1}: ${source}`;
            sourcesList.appendChild(sourceDiv);
        });
    }

    // Function to show loading message
    function showLoading(message) {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message loading-dots';
        loadingDiv.textContent = message;
        loadingDiv.id = 'loading-message';
        chatWindow.appendChild(loadingDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Function to hide loading message
    function hideLoading() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    // Function to show error message
    function showError(message) {
        const errorDiv = document.getElementById('error-message');
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        setTimeout(() => errorDiv.classList.add('hidden'), 3000);
    }

    // Function to show success message
    function showSuccess(message) {
        const successDiv = document.getElementById('success-message');
        successDiv.textContent = message;
        successDiv.classList.remove('hidden');
        setTimeout(() => successDiv.classList.add('hidden'), 3000);
    }

    // Event Listeners
    initButton.addEventListener('click', initializeSystem);

    sendButton.addEventListener('click', () => {
        const message = messageInput.value.trim();
        if (message) {
            sendMessage(message);
            messageInput.value = '';
        }
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });

    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        files.forEach(uploadPDF);
    });
});