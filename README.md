# Guava Leaf Disease Detection

Web application for detecting guava leaf diseases using deep learning models (ResNet, AlexNet, VGG, SqueezeNet, DenseNet, Inception, EfficientNet-B0).

## Features

- Multi-model prediction with consensus voting
- Drag-and-drop image upload
- Camera capture support
- AI chatbot assistant
- Confidence scores for all predictions

## Disease Classes

1. Canker
2. Dot
3. Healthy
4. Mummification
5. Rust

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Deployment

### Docker
```bash
docker build -t guava-disease .
docker run -p 7860:7860 guava-disease
```

## Technologies

- Flask, PyTorch, torchvision
- HTML5, CSS3, JavaScript
- Transfer Learning with pre-trained CNNs
