import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title
st.title("♻️ Waste Classification using MobileNetV2")

# Load trained model
model = load_model("waste_classifier_model.h5")

# Class names (based on training categories)
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Upload image
uploaded_file = st.file_uploader("Upload an image of recyclable waste", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.markdown(f"### 🔍 Predicted Class: `{predicted_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")
