import numpy as np
import tensorflow as tf
from PIL import Image
# Manipulación de imagenes 
# Convertir RGB
# Redimensionar

from .config import IM_SIZE, THRESHOLD, MODEL_PATH
# Importamos las constantes desde el archivo config.py

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
# Cargar desde el disco el modelo ya entrenado
# Organizado

def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IM_SIZE, IM_SIZE))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,224,224,3)
    return arr


def parasite_or_not(prob: float) -> str:
    return "P" if prob < THRESHOLD else "U"
    # Probabilidad numerica
    # P o U

def predict_pil(pil_img: Image.Image):
    model = load_model()
    x = preprocess_pil_image(pil_img)
    prob = float(model.predict(x, verbose=0)[0][0])
    pred = parasite_or_not(prob)
    return pred, prob
# Cargar el modelo
# Prepocesar la imagen
# Ejecución de la predicción
# Convertir la probabilidad en una clase