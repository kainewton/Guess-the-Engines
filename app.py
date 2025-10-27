import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from keras.preprocessing import image

FILE_ID = "1LG2f1wB2tS08-Jn564WAcLvDV8mvABze"
MODEL_URL = f"https://drive.google.com/drive/folders={FILE_ID}"
MODEL_PATH = "piston_classifier.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model from Google Drive.")
            st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Guess the Piston", layout="centered")

st.title("Piston Image Classifier")
st.write("Upload your piston image and let the computer guess if it's perfect or defected")

uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image uploaded", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Perfect" if prediction[0][0] < 0.5 else "Defected"

    st.subheader(f"Prediction result: **{result}**")

    if result == "Perfect":
        st.success("The piston is perfect.")
    else:
        st.warning("The piston is defected")

st.markdown("---")
st.caption("belajar machine learning | @khaisaint")