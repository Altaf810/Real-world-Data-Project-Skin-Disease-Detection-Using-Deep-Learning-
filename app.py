import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("skin_disease_model.h5")

class_labels = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

st.title("🩺 Skin Disease Detection")
st.write("Upload a skin lesion image to classify the disease.")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300) 

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)
   
    short_label = list(class_labels.keys())[pred_class].upper()
    full_label = list(class_labels.values())[pred_class]

    st.markdown(f"###  Prediction: **{full_label} ({short_label})**")
    
