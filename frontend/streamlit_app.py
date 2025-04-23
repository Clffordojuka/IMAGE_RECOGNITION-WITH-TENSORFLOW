# frontend/streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import io

# Title
st.title("🌸 Flower Classifier (MobileNetV2)")
st.write("Upload an image of a flower and I'll predict its type using a fine-tuned MobileNetV2 model!")

# Upload image
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Flower", use_column_width=True)

    # Send to FastAPI backend
    if st.button("Predict"):
        with st.spinner("Sending image to model..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    "https://image-recognition-with-tensorflow.onrender.com/predict", 
                    files=files
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🌼 Prediction: **{result['prediction'].capitalize()}**")
                    st.write(f"Confidence: {result['confidence'] * 100:.2f}%")
                else:
                    st.error(f"❌ Server responded with status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {str(e)}")
