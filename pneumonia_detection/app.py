import os
import requests
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Function to download a file from Google Drive
def download_from_drive(file_id, destination):
    url = f"https://drive.google.com/file/d/1YG_HzGByV8W3xKMNPPEMaJxExHwI9TfP/view?usp=drive_link"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        st.error("Failed to download the model from Google Drive. Please check the file ID and try again.")

# Path to the model
model_path = "model/xray_model.hdf5"

# Google Drive file ID (replace this with your actual file ID)
file_id = "1YG_HzGByV8W3xKMNPPEMaJxExHwI9TfP"

# Check if the model exists locally; if not, download it
if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive. Please wait...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
    download_from_drive(file_id, model_path)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

with st.spinner('Model is being loaded...'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """)

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)  # Adjust size based on model input
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)  # Display the uploaded image
    predictions = import_and_predict(image, model)  # Get predictions
    score = tf.nn.softmax(predictions[0])  # Apply softmax to predictions
    st.write(predictions)
    st.write(score)

    # Assuming you have `class_names` defined
    class_names = ['Normal', 'Pneumonia']

    st.write(
        "This image most likely belongs to **{}** with a {:.2f}% confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
