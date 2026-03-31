# 🧠 AI-Generated Image Detection using Computer Vision

## 📌 Overview

This project focuses on detecting whether an image is **AI-generated or real** using Computer Vision techniques. With the rapid advancement of generative models, distinguishing synthetic images from real-world images has become a critical challenge.

The system uses a deep learning-based image classification model to analyze visual patterns and identify subtle artifacts present in AI-generated images.

---

## 🎯 Problem Statement

With the rise of generative AI, fake images are becoming increasingly realistic and difficult to detect. This project aims to build a system that can automatically classify images as:

- **AI-Generated**
- **Real**

Applications include:
- Deepfake detection  
- Fake news prevention  
- Content authenticity verification  

---

## 🧠 Approach

The project follows a standard Computer Vision pipeline:

Input Image → Preprocessing → CNN Model → Prediction

### Key Steps:
- Image preprocessing (resizing, normalization)
- Feature extraction using CNN
- Classification into AI-generated or real images

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV (optional)  
- Gradio (for GUI)

---

---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/Abhinavkoushik68/AIGeneratedImageDetectionModel

cd AIGeneratedImageDetectionModel


---

### 2. Install dependencies

pip install -r requirements.txt

---

## ▶️ How to Run

### ✅ Run GUI (Gradio)

python app_ui.py

- Opens a local web interface  
- Upload an image  
- Click **Submit**  
- View prediction  

---

## 🧪 Input

- Image files (`.jpg`, `.png`, `.jpeg`)
- Upload via GUI

---

## 📊 Output

- Predicted label:
  - **AI Generated**
  - **Real**
