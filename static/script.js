// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const predictionsGrid = document.getElementById('predictions-grid');
const consensusResult = document.getElementById('consensus-result');
const previewImage = document.getElementById('preview-image');

// Camera elements
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const cameraModal = document.getElementById('camera-modal');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const captureBtn = document.getElementById('capture-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');
const closeCameraBtn = document.getElementById('close-camera');

// Camera variables
let currentStream = null;
let facingMode = 'environment'; // 'user' for front camera, 'environment' for back camera

// Event Listeners
uploadBtn.addEventListener('click', showUploadArea);
cameraBtn.addEventListener('click', openCamera);
closeCameraBtn.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);
switchCameraBtn.addEventListener('click', switchCamera);
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Show upload area
function showUploadArea() {
    uploadArea.style.display = 'block';
    document.querySelector('.upload-options').style.display = 'none';
}

// Camera functions
async function openCamera() {
    try {
        cameraModal.style.display = 'flex';
        await startCamera();
    } catch (error) {
        showError('Unable to access camera: ' + error.message);
        closeCamera();
    }
}

async function startCamera() {
    // Stop any existing stream
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    try {
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = currentStream;
    } catch (error) {
        throw new Error('Camera access denied or not available');
    }
}

function closeCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    cameraModal.style.display = 'none';
    cameraVideo.srcObject = null;
}

async function switchCamera() {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    await startCamera();
}

function capturePhoto() {
    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    context.drawImage(cameraVideo, 0, 0);

    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        closeCamera();
        handleFile(file);
    }, 'image/jpeg', 0.95);
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Validate and process file
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    // Upload and predict
    uploadAndPredict(file);
}

// Upload image and get predictions
async function uploadAndPredict(file) {
    // Show loading state
    document.querySelector('.upload-section').style.display = 'none';
    results.style.display = 'none';
    loading.style.display = 'block';

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during prediction');
        }
    } catch (error) {
        showError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    } finally {
        loading.style.display = 'none';
    }
}

// Display prediction results
function displayResults(data) {
    // Show uploaded image
    previewImage.src = data.image;

    // Clear previous results
    predictionsGrid.innerHTML = '';

    // Display predictions from each model
    const predictions = data.predictions;
    const modelNames = Object.keys(predictions);

    modelNames.forEach(modelName => {
        const pred = predictions[modelName];
        const card = createPredictionCard(modelName, pred);
        predictionsGrid.appendChild(card);
    });

    // Calculate and display consensus
    displayConsensus(predictions);

    // Show results
    results.style.display = 'block';
}

// Create prediction card for a model
function createPredictionCard(modelName, prediction) {
    const card = document.createElement('div');
    card.className = 'prediction-card';

    const modelInitial = modelName.charAt(0).toUpperCase();
    
    card.innerHTML = `
        <h4>
            <span class="model-icon">${modelInitial}</span>
            ${modelName.toUpperCase()}
        </h4>
        <div class="prediction-result">
            <div class="predicted-class">${prediction.predicted_class}</div>
            <div class="confidence">Confidence: ${prediction.confidence.toFixed(2)}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${prediction.confidence}%"></div>
            </div>
        </div>
        <div class="all-probabilities">
            <h5>All Class Probabilities:</h5>
            ${Object.entries(prediction.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([className, prob]) => `
                    <div class="probability-item">
                        <span class="probability-label">${className}</span>
                        <span class="probability-value">${prob.toFixed(2)}%</span>
                    </div>
                `).join('')}
        </div>
    `;

    return card;
}

// Calculate and display consensus prediction
function displayConsensus(predictions) {
    const modelNames = Object.keys(predictions);
    const votes = {};

    // Count votes for each class
    modelNames.forEach(modelName => {
        const predictedClass = predictions[modelName].predicted_class;
        votes[predictedClass] = (votes[predictedClass] || 0) + 1;
    });

    // Find class with most votes
    let maxVotes = 0;
    let consensusClass = '';
    Object.entries(votes).forEach(([className, count]) => {
        if (count > maxVotes) {
            maxVotes = count;
            consensusClass = className;
        }
    });

    // Calculate average confidence for consensus class
    let totalConfidence = 0;
    let count = 0;
    modelNames.forEach(modelName => {
        const pred = predictions[modelName];
        if (pred.predicted_class === consensusClass) {
            totalConfidence += pred.confidence;
            count++;
        }
    });
    const avgConfidence = totalConfidence / count;

    // Display consensus
    consensusResult.innerHTML = `
        <div class="consensus-result">${consensusClass}</div>
        <div class="consensus-confidence">
            ${maxVotes} out of ${modelNames.length} models agree (Avg. Confidence: ${avgConfidence.toFixed(2)}%)
        </div>
    `;
    
    // Trigger event for chatbot to auto-open
    const event = new CustomEvent('predictionComplete', {
        detail: {
            consensus: consensusClass,
            confidence: avgConfidence,
            votes: maxVotes,
            totalModels: modelNames.length
        }
    });
    window.dispatchEvent(event);
}

// Show error message
function showError(message) {
    loading.style.display = 'none';
    results.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';

    // Create or update error message
    let errorDiv = document.querySelector('.error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        document.querySelector('.upload-section').after(errorDiv);
    }
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';

    // Hide error after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

// Reset upload for new prediction
function resetUpload() {
    results.style.display = 'none';
    uploadArea.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    document.querySelector('.upload-options').style.display = 'grid';
    fileInput.value = '';

    // Hide any error messages
    const errorDiv = document.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// Chatbot functionality
const chatbotToggle = document.getElementById('chatbot-toggle');
const chatbotContainer = document.getElementById('chatbot-container');
const chatbotClose = document.getElementById('chatbot-close');
const chatbotMessages = document.getElementById('chatbot-messages');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotSend = document.getElementById('chatbot-send');
const chatBadge = document.getElementById('chat-badge');

let chatHistory = [];
let lastPrediction = null;

// Toggle chatbot
chatbotToggle.addEventListener('click', () => {
    chatbotContainer.style.display = chatbotContainer.style.display === 'none' ? 'flex' : 'none';
    chatBadge.style.display = 'none';
});

chatbotClose.addEventListener('click', () => {
    chatbotContainer.style.display = 'none';
});

// Send message
chatbotSend.addEventListener('click', sendChatMessage);
chatbotInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

async function sendChatMessage() {
    const message = chatbotInput.value.trim();
    if (!message) return;

    // Add user message
    addChatMessage(message, 'user');
    chatbotInput.value = '';
    chatbotSend.disabled = true;

    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot-message';
    typingDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatbotMessages.appendChild(typingDiv);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                last_prediction: lastPrediction
            })
        });

        const data = await response.json();
        
        // Remove typing indicator
        typingDiv.remove();

        if (data.success) {
            addChatMessage(data.response, 'bot');
            chatHistory.push({ role: 'user', content: message });
            chatHistory.push({ role: 'assistant', content: data.response });
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    } catch (error) {
        typingDiv.remove();
        addChatMessage('Sorry, I could not connect to the chat service. Please check your API key or try again later.', 'bot');
    } finally {
        chatbotSend.disabled = false;
    }
}

function addChatMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Render markdown for bot messages
    if (sender === 'bot') {
        contentDiv.innerHTML = renderMarkdown(content);
    } else {
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    chatbotMessages.appendChild(messageDiv);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

// Simple markdown renderer for chatbot messages
function renderMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')  // Bold **text**
        .replace(/\*(.+?)\*/g, '<em>$1</em>')              // Italic *text*
        .replace(/\n/g, '<br>')                             // Line breaks
        .replace(/`(.+?)`/g, '<code>$1</code>');            // Inline code
}

// Store last prediction for context
window.addEventListener('predictionComplete', (e) => {
    lastPrediction = e.detail;
    
    // Auto-open chatbot
    chatbotContainer.style.display = 'flex';
    chatBadge.style.display = 'none';
    
    // Clear previous chat messages
    chatbotMessages.innerHTML = '';
    chatHistory = [];
    
    // Add welcome message with prediction result
    setTimeout(() => {
        const diseaseName = e.detail.consensus || 'Unknown';
        const confidence = e.detail.confidence || 0;
        
        const welcomeMessage = `I detected **${diseaseName}** with ${confidence.toFixed(1)}% confidence. Ask me about symptoms, treatment, prevention, or any other questions about this disease!`;
        addChatMessage(welcomeMessage, 'bot');
    }, 500);
});
