import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
import requests
import os

st.set_page_config(
    page_title="KidneyClassify",  
    page_icon="🩺", 
    layout="centered",  
    initial_sidebar_state="auto"
)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

#URL model
MODEL_URL = "https://raw.githubusercontent.com/jeremyfelix12/KidneyClassify/main/model.h5"
MODEL_PATH = "median.h5"

#download model
if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as file:
        response = requests.get(MODEL_URL)
        file.write(response.content)
#load model
model = load_model(MODEL_PATH)

class_labels = ['Batu Ginjal', 'Kista', 'Normal', 'Tumor']

# Preprocessing gambar
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = image.convert("L")  
    image_array = np.array(image)
    image_array = cv2.medianBlur(image_array, 5)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("Klasifikasi Penyakit Ginjal")

# Upload gambar
uploaded_file = st.file_uploader("Silahkan Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar di tengah
    image = Image.open(uploaded_file)
    image_resized = image.resize((224, 224)) 
    
    # CSS untuk gambar 
    st.markdown(
    f"""
    <div style="
        display: flex; 
        justify-content: center; 
        align-items: center; 
        margin-top: 20px; 
        border: 4px solid #122525; 
        border-radius: 15px; 
        padding: 8px; 
        width: fit-content; 
        margin-left: auto; 
        margin-right: auto;
    ">
        <img src="data:image/png;base64,{image_to_base64(image_resized)}" 
             alt="Uploaded Image" 
             style="max-width: 100%; 
                    height: auto; 
                    border-radius: 10px;" />
    </div>
    """,
    unsafe_allow_html=True
)

    # Preprocess 
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Prediksi
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # CSS untuk hasil prediksi dan confidence
    st.markdown(
    f"""
    <div style="display: flex; 
    justify-content: center; 
    text-align: center; 
    margin-top: 20px; 
    border: 2px solid #122525; 
    border-radius: 10px; 
    padding: 10px;">
    <h3 style="font-size: 24px; color: #FFFFFF;">Prediksi: {predicted_class}</h3>
    <h3 style="font-size: 24px; color: #FFFFFF;">Confidence: {confidence:.2f}</h3>
    </div>
    """,
    unsafe_allow_html=True
)
