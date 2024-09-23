import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"])

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
