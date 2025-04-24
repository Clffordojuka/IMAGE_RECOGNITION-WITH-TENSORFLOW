# frontend/streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #6a1b9a;
        }
        .subtitle {
            font-size: 1.1em;
            color: #555;
        }
        .uploaded-img {
            border: 2px solid #eee;
            padding: 5px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🌸 Flower Classifier (MobileNetV2)</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of a flower and let the AI predict its type using a fine-tuned MobileNetV2 model.</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📷 Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Flower 🌺", use_column_width=True, output_format='PNG', clamp=True)
    with col2:
        st.markdown("### Ready to Predict?")
        if st.button("🔍 Predict Flower Type"):
            with st.spinner("Sending image to model... ⏳"):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        "https://image-recognition-with-tensorflow.onrender.com/predict",
                        files=files
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"🌼 Prediction: **{result['prediction'].capitalize()}**")
                        st.progress(min(result["confidence"], 1.0))  # Show confidence visually
                        st.info(f"Confidence Level: **{result['confidence'] * 100:.2f}%**")
                    else:
                        st.error(f"❌ Server Error: Status Code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"🚫 Could not connect to server.\n\n{str(e)}")
else:
    st.info("📁 Please upload a flower image to get started.")