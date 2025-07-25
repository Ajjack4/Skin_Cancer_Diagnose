# 🧪 Skin Cancer Diagnose App

A Flask-based REST API that uses a pretrained deep learning model from Hugging Face to predict whether a given skin lesion image is **benign** or **malignant**.

Powered by 🤗 Hugging Face Transformers and PyTorch.

## 🔍 Features

- Upload a skin lesion image via POST request
- Returns:
  - Predicted label: `Benign` or `Malignant`
  - Confidence score (probability)

## 🧠 Model Info

- **Model**: [duchaba/skin_cancer_diagnose](https://huggingface.co/duchaba/skin_cancer_diagnose)
- Pretrained image classification model for skin cancer detection

## 🚀 Tech Stack

- Python
- Flask
- Hugging Face Transformers
- PyTorch
- PIL (Pillow)

## 📦 Installation

```bash
git clone https://github.com/your-username/skin-cancer-diagnose-app.git
cd skin-cancer-diagnose-app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
