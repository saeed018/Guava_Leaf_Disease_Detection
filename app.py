from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

NUM_CLASSES = 5
CLASS_NAMES = ['Canker', 'Dot', 'Healthy', 'Mummification', 'Rust']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    'resnet': {'file': 'models/model_ft_resnet.pth', 'input_size': 224},
    'alexnet': {'file': 'models/model_ft_alexnet.pt', 'input_size': 224},
    'vgg': {'file': 'models/model_ft_vgg.pt', 'input_size': 224},
    'squeezenet': {'file': 'models/model_ft_squeezenet.pt', 'input_size': 224},
    'densenet': {'file': 'models/model_ft_densenet.pt', 'input_size': 224},
    'inception': {'file': 'models/model_ft_inception.pt', 'input_size': 299},
    'efficientnet_b0': {'file': 'models/model_ft_efficientnet_b0.pth', 'input_size': 224}
}

loaded_models = {}


def initialize_model(model_name, num_classes):
    model_ft = None
    
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=False)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        
    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=False)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "efficientnet_b0":
        model_ft = models.efficientnet_b0(pretrained=False)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model_ft


def load_models():
    print("Loading models...")
    for model_name, config in MODEL_CONFIGS.items():
        try:
            model_path = config['file']
            if os.path.exists(model_path):
                print(f"Loading {model_name}...")
                model = initialize_model(model_name, NUM_CLASSES)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                loaded_models[model_name] = model
                print(f"✓ {model_name} loaded successfully")
            else:
                print(f"✗ Model file not found: {model_path}")
        except Exception as e:
            print(f"✗ Error loading {model_name}: {str(e)}")
    
    print(f"\nTotal models loaded: {len(loaded_models)}/{len(MODEL_CONFIGS)}")


load_models()


def get_transforms(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def predict_image(image, model_name):
    if model_name not in loaded_models:
        return None
    
    model = loaded_models[model_name]
    input_size = MODEL_CONFIGS[model_name]['input_size']
    transform = get_transforms(input_size)
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
    return {
        'predicted_class': CLASS_NAMES[predicted.item()],
        'confidence': float(confidence.item() * 100),
        'all_probabilities': {CLASS_NAMES[i]: float(all_probs[i] * 100) for i in range(NUM_CLASSES)}
    }


@app.route('/')
def index():
    return render_template('index.html', 
                         models=list(MODEL_CONFIGS.keys()),
                         loaded_models=list(loaded_models.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get predictions from all loaded models
        predictions = {}
        for model_name in loaded_models.keys():
            result = predict_image(image, model_name)
            if result:
                predictions[model_name] = result
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_str}',
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])
        last_prediction = data.get('last_prediction')
        
        # Build context
        context = """You are a helpful agricultural expert specializing in guava plant diseases. 
Your role is to help farmers and agricultural workers understand:
- The 5 main guava diseases: Canker, Dot, Healthy (no disease), Mummification, and Rust
- Symptoms and identification of each disease
- Treatment methods and prevention strategies
- Best agricultural practices for healthy guava cultivation

Provide practical, clear, and actionable advice. Keep responses concise but informative."""

        if last_prediction:
            context += f"\n\nThe user just received a diagnosis showing: {last_prediction}"

        # Use free API (Groq - you can also use Google Gemini, HuggingFace, etc.)
        # For this example, using a simple rule-based response
        # To use real API, uncomment the API section below and set GROQ_API_KEY environment variable
        
        response_text = get_chatbot_response(user_message, context, history)
        
        return jsonify({
            'success': True,
            'response': response_text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_chatbot_response(message, context, history):
    api_key = os.environ.get('GROQ_API_KEY', '')
    
    if api_key:
        try:
            messages = [{"role": "system", "content": context}]
            messages.extend(history[-6:])  # Last 3 exchanges
            messages.append({"role": "user", "content": message})
            
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': messages,
                    'max_tokens': 600,
                    'temperature': 0.7
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            pass  # Fall through to rule-based responses
    
    # Fallback to rule-based responses
    message_lower = message.lower()
    
    disease_info = {
        'canker': {
            'symptoms': 'Canker appears as sunken lesions on stems, branches, and fruits. The lesions are usually circular with raised margins.',
            'treatment': 'Remove and destroy infected plant parts. Apply copper-based fungicides. Ensure proper drainage and avoid waterlogging.',
            'prevention': 'Maintain plant hygiene, prune regularly, apply preventive copper sprays during rainy season.'
        },
        'dot': {
            'symptoms': 'Dot disease shows as small dark spots on leaves and fruits. Spots may have a yellow halo.',
            'treatment': 'Apply appropriate fungicides. Remove severely infected leaves. Improve air circulation around plants.',
            'prevention': 'Avoid overhead irrigation, maintain proper plant spacing, regular monitoring for early detection.'
        },
        'rust': {
            'symptoms': 'Rust appears as orange-brown powdery pustules on leaf undersides. Leaves may yellow and drop.',
            'treatment': 'Apply sulfur or copper-based fungicides. Remove infected leaves. Ensure good air circulation.',
            'prevention': 'Plant resistant varieties, avoid excess nitrogen fertilizer, maintain balanced nutrition.'
        },
        'mummification': {
            'symptoms': 'Fruits shrivel, dry up, and turn black. They remain attached to the tree in a mummified state.',
            'treatment': 'Remove and destroy all mummified fruits. Apply appropriate fungicides during flowering.',
            'prevention': 'Collect and destroy fallen fruits, prune to improve air circulation, regular sanitation practices.'
        }
    }
    
    # Check for disease-specific questions
    for disease, info in disease_info.items():
        if disease in message_lower:
            if 'symptom' in message_lower or 'identify' in message_lower or 'look' in message_lower:
                return f"**{disease.title()} Symptoms:**\n{info['symptoms']}"
            elif 'treat' in message_lower or 'cure' in message_lower or 'control' in message_lower:
                return f"**{disease.title()} Treatment:**\n{info['treatment']}"
            elif 'prevent' in message_lower or 'avoid' in message_lower:
                return f"**{disease.title()} Prevention:**\n{info['prevention']}"
            else:
                return f"**About {disease.title()}:**\n\n**Symptoms:** {info['symptoms']}\n\n**Treatment:** {info['treatment']}\n\n**Prevention:** {info['prevention']}"
    
    # General questions
    if 'all disease' in message_lower or 'types of disease' in message_lower or 'what disease' in message_lower:
        return """The main guava diseases are:
1. **Canker** - Sunken lesions on stems and fruits
2. **Dot Disease** - Small dark spots on leaves
3. **Rust** - Orange-brown pustules on leaves
4. **Mummification** - Fruits dry up and turn black

Ask me about any specific disease for detailed information!"""
    
    elif 'prevent' in message_lower or 'protection' in message_lower:
        return """**General Prevention Tips for Guava Diseases:**
1. Maintain proper plant hygiene and sanitation
2. Prune regularly to improve air circulation
3. Avoid waterlogging and ensure good drainage
4. Apply preventive fungicides during vulnerable periods
5. Remove and destroy infected plant parts immediately
6. Monitor plants regularly for early disease detection
7. Maintain balanced nutrition and avoid excess nitrogen"""
    
    elif 'organic' in message_lower or 'natural' in message_lower:
        return """**Organic Disease Management:**
1. Use neem oil or neem cake for pest and disease control
2. Apply copper-based organic fungicides
3. Use sulfur dust for rust control
4. Practice crop rotation and companion planting
5. Apply compost and organic matter to improve soil health
6. Use biological control agents when available
7. Maintain proper spacing for air circulation"""
    
    elif 'spray' in message_lower or 'fungicide' in message_lower:
        return """**Fungicide Application Tips:**
1. Apply early morning or late evening
2. Ensure complete coverage of leaves (both sides)
3. Follow recommended dosage and intervals
4. Rotate fungicides to prevent resistance
5. Common fungicides: Copper oxychloride, Mancozeb, Carbendazim
6. Always follow safety precautions
7. Observe harvest interval before fruit collection"""
    
    else:
        return """I'm here to help with guava disease information! You can ask me about:
- Specific diseases (Canker, Dot, Rust, Mummification)
- Symptoms and identification
- Treatment methods
- Prevention strategies
- Organic management
- Fungicide application

What would you like to know?"""


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(loaded_models),
        'total_models': len(MODEL_CONFIGS)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
