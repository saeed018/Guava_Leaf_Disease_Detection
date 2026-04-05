# Guava Leaf Disease Detection (Deep Learning)

This project builds an automated **guava leaf disease classification** system using **transfer learning** and compares **7 CNN architectures** to select the best-performing model. A simple software interface (web app) is included so users can upload a guava leaf image and get an instant disease prediction.

## ✨ Key Features
- **5-class classification:** Healthy, Canker, Dot, Mummification, Rust  
- **Compared 7 CNN backbones:** EfficientNet, ResNet, DenseNet, AlexNet, VGG, SqueezeNet, Inception-v3  
- Evaluation with **Accuracy, Loss, Precision, Recall, F1-score, Confusion Matrix**
- **Simple web interface** for image upload and prediction

## 📌 Dataset
- Source: Kaggle guava leaf dataset  
- Total images: **2299**
- Class distribution (approx.):  
  - Healthy: 469  
  - Canker: 455  
  - Mummification: 430  
  - Dot: 470  
  - Rust: 475  

> Note: Please do not upload the dataset images to GitHub. Instead, download from Kaggle and place them in the `data/` folder (see structure below).

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

## 🗂️ Project Structure (Suggested)

guava-leaf-disease-detection/
│
├── data/ # (not included) dataset folder
│ ├── train/
│ ├── val/
│ └── test/
│
├── notebooks/ # optional Jupyter notebooks
├── src/ # training / evaluation scripts
│ ├── train.py
│ ├── evaluate.py
│ ├── utils.py
│ └── config.py
│
├── app/ # web app code (Streamlit/Flask/etc.)
│ └── app.py
│
├── results/ # plots, confusion matrices, metrics
├── requirements.txt
└── README.md


## ⚙️ Installation
1) Clone the repo:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Create and activate a virtual environment (recommended):
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
🚀 How to Train

Example (update command to match your script names):

python src/train.py --model efficientnet --epochs 15 --batch_size 32
📊 How to Evaluate
python src/evaluate.py --model efficientnet --weights path/to/best_model.pth
🖥️ Run the Web App

(If you used Streamlit)

streamlit run app/app.py

(If you used Flask)

python app/app.py

Then open the shown local URL and upload a guava leaf image to get the prediction.

✅ Output
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
