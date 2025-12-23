import os

import requests
import streamlit as st

"""
Streamlit front-end for the suicide risk classification model.

This app sends the input text to the FastAPI backend (`inference.py`)
running at `http://localhost:8000/predict` by default and displays
the predicted label and confidence.

Before running this app, make sure to start the FastAPI server:
    uvicorn inference:app --host 0.0.0.0 --port 8000 --reload
"""

# URL of the FastAPI prediction endpoint
API_URL = os.getenv("SUICIDE_API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Suicide Risk Classifier", layout="centered")

st.title("Suicide Risk Text Classifier")
st.write(
    "Enter a text in Russian or English. The model will predict whether the text "
    "contains suicide-related content."
)

user_input = st.text_area("Input text", height=200, placeholder="Type or paste text here...")

if st.button("Run prediction"):
    if not user_input.strip():
        st.warning("Please enter some text before running the prediction.")
    else:
        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=60,
                )
                if response.status_code != 200:
                    st.error(f"Request failed with status code {response.status_code}: {response.text}")
                else:
                    data = response.json()
                    label = data.get("label")
                    confidence = data.get("confidence")
                    device = data.get("device", "unknown")

                    st.subheader("Prediction")
                    st.write(f"**Predicted label:** {label}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                    st.write(f"**Device:** {device}")

                    with st.expander("Raw response"):
                        st.json(data)
            except Exception as e:
                st.error(f"Error while calling the API: {e}")


