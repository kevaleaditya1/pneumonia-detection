# api/predict.py
import json
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model
model = tf.keras.models.load_model('model/xray_model.hdf5')

def predict(image_data):
    # Image preprocessing and model prediction logic
    image = Image.open(image_data)
    image = image.resize((224, 224))  # Resize to your model's expected input size
    image_array = np.array(image) / 255.0  # Normalize the image
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    return prediction

def handler(request):
    # Assuming image data is passed in request
    image_data = request.files.get('file')
    result = predict(image_data)
    return json.dumps({'prediction': result.tolist()})
