"""Minimal Streamlit demo that calls the FastAPI /predict endpoint. Research prototype only."""

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Skin Lesion Demo", layout="centered")
st.title("Skin lesion classifier (research prototype)")
st.warning("Not a medical device. Do not use for diagnosis or any clinical decision.")

upload = st.file_uploader("Upload a dermoscopy image", type=["jpg", "jpeg", "png"])
if upload:
    st.image(upload, caption="Input", use_container_width=True)
    if st.button("Predict"):
        try:
            resp = requests.post(
                f"{API_URL}/predict", files={"file": (upload.name, upload.getvalue())}, timeout=60
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
        else:
            body = resp.json()
            st.subheader(f"Top class: {body['top_class']} ({body['confidence']:.1%})")
            st.bar_chart(body["probabilities"])
            overlay = Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
            st.caption(body["disclaimer"])
