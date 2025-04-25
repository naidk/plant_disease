import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = load_model('crop_disease_model.h5')

# Set image size
img_size = 128

# Class labels (same order as in training)
class_labels = ['Apple___Black_rot', 'Apple___healthy', 'Corn___Cercospora_leaf_spot', ...]  # Add all your classes here

st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]

    st.success(f"🧠 Predicted Disease: **{class_label}**")