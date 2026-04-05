# Guava Leaf Disease Detection (Deep Learning)

This project builds an automated **guava leaf disease classification** system using **transfer learning** and compares **7 CNN architectures** to select the best-performing model. A simple software interface (web app) is included so users can upload a guava leaf image and get an instant disease prediction.

## ✨ Key Features
- **5-class classification:** Healthy, Canker, Dot, Mummification, Rust  
- **Compared 7 CNN backbones:** EfficientNet, ResNet, DenseNet, AlexNet, VGG, SqueezeNet, Inception-v3  
- Evaluation with **Accuracy, Loss, Precision, Recall, F1-score, Confusion Matrix**
- **Simple web interface** for image upload and prediction

## 🧠 Models and Results (Summary)
| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| **EfficientNet** | **99.57%** | **0.0298** |
| ResNet | 98.71% | 0.0452 |
| DenseNet | 96.34% | 0.1344 |
| AlexNet | 93.76% | 0.1900 |
| SqueezeNet | 93.33% | 0.2040 |
| VGG | 90.97% | 0.2164 |
| Inception-v3 | 86.24% | 0.3599 |

**Why EfficientNet performed best:** It uses balanced scaling (depth/width/resolution) plus efficient MBConv blocks and attention (SE), which helps capture fine leaf texture patterns and generalize well on a moderate-size dataset.

## output
Predicted class: Healthy / Canker / Dot / Mummification / Rust
Confidence score
Evaluation: accuracy, loss, precision, recall, F1-score, confusion matrix
🔮 Future Work
Larger and more diverse datasets (real field conditions)
More disease types and architectures
Mobile app deployment for farmers
Real-time monitoring and alert system
🙏 Acknowledgement

Thanks to the dataset providers and the research community. Special thanks to my supervisor and committee members for guidance and support.
