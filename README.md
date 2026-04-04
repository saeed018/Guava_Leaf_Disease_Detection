<<<<<<< HEAD
---
title: Guava Leaf Disease Detection
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Guava Leaf Disease Detection

Web application for detecting guava leaf diseases using deep learning models (ResNet, AlexNet, VGG, SqueezeNet, DenseNet, Inception, EfficientNet-B0).

## Live Demo

🌐 [Try the live application](https://abusaeed018-guava-leaf-disease-detection.hf.space/)

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

### Hugging Face Spaces
```bash
git push
```

Configure with app.py as the main application file.

### Docker
```bash
docker build -t guava-disease .
docker run -p 7860:7860 guava-disease
```


## Technologies

- Flask, PyTorch, torchvision
- HTML5, CSS3, JavaScript
- Transfer Learning with pre-trained CNNs
=======
# Guava_Leaf_Disease_Detection
A comparative study of 7 pretrained CNN models (EfficientNet, ResNet, DenseNet, etc.) for guava leaf disease classification, including evaluation metrics and a user-friendly web interface for predictions.
>>>>>>> aa44c6e2c30feda9a8007827ae757ecc71e2eb5d
