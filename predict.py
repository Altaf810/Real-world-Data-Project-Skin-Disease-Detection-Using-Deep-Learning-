from tensorflow.keras.models import load_model
import numpy as np
import cv2


model = load_model("skin_disease_model.h5")


disease_classes = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis-like Lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevi (nv)",
    6: "Vascular Lesions (vasc)"
}

def predict_skin_disease(image_path):
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))   
    img = img / 255.0
    img = np.expand_dims(img, axis=0)   

   
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction: {disease_classes[class_index]} ({confidence*100:.2f}% confidence)")


predict_skin_disease(r"D:\Skin diease detection\test1_image.jpg")
