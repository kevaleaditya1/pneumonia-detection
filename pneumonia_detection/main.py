import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

@st.cache_resource
def load_model():
    model_path = "xray_model.hdf5"

    # Check if the model is already downloaded
    if not os.path.exists(model_path):
        # Google Drive file ID
        file_id = "1YG_HzGByV8W3xKMNPPEMaJxExHwI9TfP"
        url = f"https://drive.google.com/file/d/1YG_HzGByV8W3xKMNPPEMaJxExHwI9TfP/view?usp=drive_link"
        st.info("Downloading model from Google Drive. Please wait...")
        gdown.download(url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

with st.spinner('Model is being loaded..'):
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

    # Assuming you have class_names defined
    class_names = ['Normal', 'Pneumonia']
    
    st.write(
        "This image most likely belongs to **{}** with a {:.2f}% confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
