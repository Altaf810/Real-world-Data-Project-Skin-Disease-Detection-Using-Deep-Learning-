# Skin Disease Detection Using Deep Learning

## 📌 Project Overview
This project is a Deep Learning-based Skin Disease Detection System developed using Python and TensorFlow/Keras. The system analyzes skin lesion images and predicts different categories of skin diseases using a Convolutional Neural Network (CNN).

The project uses the HAM10000 dataset, which contains dermatoscopic images of various skin lesion types. The objective is to automate the disease detection process and assist in early medical diagnosis.

---

# 🎯 Objectives
- Detect skin diseases using image classification
- Preprocess medical images for model training
- Train a CNN model for disease prediction
- Improve healthcare assistance using AI

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| NumPy | Numerical Computation |
| Pandas | Data Analysis |
| OpenCV | Image Processing |
| Matplotlib | Visualization |
| Scikit-learn | Model Evaluation |
| Streamlit | Web Application Development |

---

# 📂 Dataset
Dataset Used: **HAM10000 (Human Against Machine with 10000 Training Images)**

The dataset contains:
- Skin lesion images
- Metadata information
- Multiple disease categories

---

# ⚙️ Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Image Resizing & Normalization
4. Label Encoding
5. CNN Model Building
6. Model Training
7. Prediction & Evaluation
8. Streamlit Website Deployment

---

# 📁 Project Structure

```plaintext
SkinDiseaseDetection/
│
├── preprocessing.py
├── predict.py
├── app.py
├── preprocessed_128.npz
├── data.pkl
├── HAM10000_metadata.csv
├── model.h5
├── README.md
└── report.pdf
