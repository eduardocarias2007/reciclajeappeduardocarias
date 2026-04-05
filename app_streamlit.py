import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array

st.set_page_config(page_title="Detector de enfermedades de rosa", page_icon="🌹")

MODEL_DIR = Path("modelo_rosa_mobilenet")
MODEL_PATH = MODEL_DIR / "rose_disease_mobilenet.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def prepare_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title("🌹 Reconocimiento de enfermedades en hojas de rosa")
st.write("Sube una imagen y el modelo estimará la clase más probable.")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.warning("No encuentro el modelo guardado. Primero ejecuta el notebook y genera la carpeta modelo_rosa_mobilenet.")
    st.stop()

model, class_names = load_model_and_classes()

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    x = prepare_image(image)
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))

    st.subheader("Resultado")
    st.write(f"**Clase predicha:** {class_names[top_idx]}")
    st.write(f"**Confianza:** {preds[top_idx]*100:.2f}%")

    st.subheader("Probabilidades por clase")
    for label, score in sorted(zip(class_names, preds), key=lambda z: z[1], reverse=True):
        st.write(f"- {label}: {score*100:.2f}%")
