# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
This project aims to build an accurate image classification system for different fish species using deep learning. It involves training convolutional neural networks (CNNs), leveraging **transfer learning**, and deploying a real-time prediction tool via **Streamlit**.

## 🧠 Skills Gained
- Python Programming
- Deep Learning with TensorFlow/Keras
- Data Preprocessing and Augmentation
- Transfer Learning
- Model Evaluation (Accuracy, Precision, Recall, F1-score)
- Confusion Matrix Analysis
- Visualization (Matplotlib, Seaborn)
- Model Saving (`.h5`, `.pkl` `.keras`)
- Streamlit Web App Development
- Deployment and User Interaction

## 🌐 Domain
**Image Classification**

## 📌 Problem Statement
Classify images of fish into multiple categories using deep learning. The project involves:
- Building CNNs from scratch and via transfer learning.
- Saving and comparing model performance.
- Deploying the best-performing model in a web app for user input predictions.

## 💼 Business Use Cases
- ✅ **Enhanced Accuracy:** Identify the most effective model for accurate fish classification.
- 🚀 **Deployment Ready:** Build a real-time web application using Streamlit.
- 📊 **Model Benchmarking:** Analyze and compare multiple models to select the best fit for deployment.

## 🧪 Approach

### 🗂 Data Preprocessing and Augmentation
- Normalize images to [0, 1] range.
- Apply rotation, zoom, flipping, and shift augmentations.
- Use `ImageDataGenerator` for memory-efficient image handling.

### 🧠 Model Training
- Train a CNN model from scratch.
- Fine-tune 5 pre-trained models:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0
- Save best-performing models (`.h5` or `.pkl` or `.keras`) based on accuracy.

### 📈 Model Evaluation
- Use metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- Plot training/validation accuracy and loss for each model.

### 💻 Deployment
- Build a **Streamlit web app** to:
  - Upload fish images.
  - Predict species category.
  - Show top-3 predictions with confidence scores.

## 📁 Dataset
- The dataset contains **images of fish**, organized in folders by species name.
- It is loaded using `ImageDataGenerator` with real-time augmentation.

### 📦 Dataset Access
- Format: ZIP archive containing class-wise folders.
- Load and extract using standard Python or Jupyter methods.

## 📄 Deliverables
- 📂 Source code (model training, evaluation, deployment).
- ✅ Best model file(s): `.h5` / `.pkl`
- 📊 Visualizations: Accuracy/loss curves, confusion matrices.
- 🌐 Streamlit app for live predictions.
- 📘 Detailed documentation in this README and comments in code.
- 🐙 GitHub repository with structured files and example predictions.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ashwini98/fish-classification
cd fish-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run model training
```bash
python Fish_classification.ipynb
```

### 4. Launch Streamlit app
```bash
streamlit Final_app.py
```
