import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'cats_dogs_transfer_model.h5'
CLASSES = ['Cat', 'Dog']

# --- Page Config ---
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image to check if it's a **Cat** or a **Dog**!")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.warning("Please ensure 'cats_dogs_transfer_model.h5' is in the same directory and you have run the training script first.")
    st.stop()

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the image
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Uploaded Image', use_container_width=True)

        st.write("Classifying...")

        # Preprocess the image
        # Resize to 224x224 as required by VGG16
        img = image_pil.resize((224, 224))
        
        # Convert to array
        x = image.img_to_array(img)
        
        # Expand dims (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)
        
        # Preprocess input (VGG16 specific)
        x = preprocess_input(x)

        # Predict
        predictions = model.predict(x)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        # Display Result
        st.success(f"Prediction: **{predicted_class}**")
        st.progress(float(confidence))
        st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
