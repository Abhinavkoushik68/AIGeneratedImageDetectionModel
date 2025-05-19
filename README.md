# AIGeneratedImageDetectionModel

# AI vs Real Image Classifier

A deep learning-based image classification model to detect whether a given image is **AI-generated** or **real**. This project leverages convolutional neural networks (CNNs) using TensorFlow and Keras to classify images with high precision, trained on a large dataset of 100,000 labeled images.

---

## 📌 Overview

### ❓ The Need for This Project

With the rapid advancement of AI image generation tools like DALL·E, MidJourney, and Stable Diffusion, distinguishing between real and synthetically generated images has become increasingly difficult for both the general public and even automated systems. This poses potential risks in media authenticity, misinformation, copyright violations, and digital forensics.

The ability to accurately **detect AI-generated images** is crucial in multiple domains:
- **Journalism** – Verifying authenticity of published visuals.
- **Cybersecurity** – Detecting fake identities or misleading visuals.
- **Social Media** – Curbing deepfakes and visual misinformation.
- **Academic & Research Integrity** – Ensuring image authenticity in submissions.

### ✅ How This Project Addresses the Problem

This project presents a practical and scalable solution by building a machine learning pipeline capable of:
- Automatically loading and preprocessing large volumes of image data.
- Training a robust CNN to distinguish between AI and real images.
- Evaluating model performance using metrics such as accuracy, precision, and recall.
- Providing prediction capabilities on unseen images.
- Saving and deploying the trained model for inference in real-world use cases.

---

## 🧠 Model Architecture

- **Input size**: (32 x 32 x 3) RGB images
- **Convolution Layers**: 3 layers with ReLU activations
- **Pooling Layers**: MaxPooling after each convolution
- **Dense Layers**: 
  - 1 Hidden layer with 32 neurons (ReLU)
  - 1 Output layer with sigmoid activation for binary classification
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Accuracy Achieved**: ~94.5% on training, ~91.6% on validation

---

## 📂 Dataset

- Total images: **100,000**
- Classes: `REAL` and `FAKE`
- Source directory structured for TensorFlow loading
- Dataset was split into:
  - 80% for training
  - 10% for validation
  - 10% for testing

---

## ⚙️ Preprocessing

- Rescaling pixel values from `[0, 255]` to `[0, 1]`
- Batch loading using `image_dataset_from_directory`
- Dataset conversion to numpy iterators for visualization
- Image reshaping and resizing using OpenCV and TensorFlow

---

## 📈 Training Performance

The model was trained over 20 epochs and logged via TensorBoard. Key highlights:

- Validation accuracy plateaued around **91.6%**
- Model showed good generalization on test data
- Evaluation metrics on test set:
  - **Precision**: 0.923
  - **Recall**: 0.913
  - **Accuracy**: 0.919

Performance visualizations:
- Loss vs Epoch
- Accuracy vs Epoch

---

## 🧪 Testing & Inference

You can test the model using new unseen images by:

1. Loading the `.keras` model from the saved path.
2. Resizing and normalizing input images.
3. Making predictions via `model.predict()`.
4. Interpreting the output probability:
   - `> 0.5`: Predicted as **REAL**
   - `<= 0.5`: Predicted as **AI**

---

## 🛠️ Installation & Usage

### 🔧 Requirements
- Python ≥ 3.8
- TensorFlow ≥ 2.8
- OpenCV
- Matplotlib
- NumPy

### 📦 Installation
```bash
pip install tensorflow opencv-python matplotlib numpy

