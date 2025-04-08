import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set constants
IMAGE_SIZE = (128, 128)
CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'
]

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
    return model

model = load_model()

# Streamlit app UI
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to detect plant disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocessing
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### ✅ Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
