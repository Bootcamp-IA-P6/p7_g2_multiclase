"""
Lógica de inferencia para el modelo de detección de incendios.

Pipeline:
    Imagen → EfficientNetB0 (features) → VotingClassifier (predicción)
"""

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────
MODEL_PATH  = Path("models/02_model.pkl")
IMG_SIZE    = (224, 224)

# Mapeo de índices a nombres de clase
# Orden del notebook original: ['Smoke', 'fire', 'non fire']
CLASS_NAMES = {
    0: "smoke",
    1: "fire",
    2: "non fire"
}

# Configuración visual para la app
CLASS_CONFIG = {
    "fire": {
        "icon":    "🔥",
        "label":   "FUEGO DETECTADO",
        "color":   "red",
        "message": "Activar protocolo de emergencia inmediatamente",
        "level":   "danger"
    },
    "smoke": {
        "icon":    "💨",
        "label":   "HUMO DETECTADO",
        "color":   "orange",
        "message": "Posible foco incipiente — verificar zona urgente",
        "level":   "warning"
    },
    "non fire": {
        "icon":    "🌲",
        "label":   "SIN INCIDENCIA",
        "color":   "green",
        "message": "No se detectan señales de incendio",
        "level":   "success"
    }
}


def load_feature_extractor():
    """Carga EfficientNetB0 sin cabeza clasificadora para extraer features."""
    print("⏳ Cargando EfficientNetB0...")
    extractor = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    print("✅ EfficientNetB0 cargado")
    return extractor


def load_classifier():
    """Carga el VotingClassifier entrenado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ No se encuentra el modelo en {MODEL_PATH}\n"
            f"   Asegúrate de tener models/02_model.pkl"
        )
    classifier = joblib.load(MODEL_PATH)
    print(f"✅ Clasificador cargado: {type(classifier).__name__}")
    return classifier


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocesa una imagen PIL para EfficientNetB0.
    Mismo preprocesamiento que en el notebook de entrenamiento.
    """
    img_rgb    = img.convert("RGB")
    img_resized = img_rgb.resize(IMG_SIZE)
    img_array  = np.array(img_resized, dtype=np.float32)
    img_array  = np.expand_dims(img_array, axis=0)
    img_array  = preprocess_input(img_array)
    return img_array


def predict(
    img:       Image.Image,
    extractor: EfficientNetB0,
    classifier
) -> dict:
    """
    Realiza una predicción completa sobre una imagen.

    Parámetros:
        img:        imagen PIL subida por el usuario
        extractor:  EfficientNetB0 para extraer features
        classifier: VotingClassifier entrenado

    Retorna:
        dict con clase predicha, confianza y probabilidades
    """
    # 1. Preprocesar imagen
    img_array = preprocess_image(img)

    # 2. Extraer features con EfficientNetB0
    features = extractor.predict(img_array, verbose=0)  # shape: (1, 1280)

    # 3. Predecir con el ensemble
    probs     = classifier.predict_proba(features)[0]   # shape: (3,)
    pred_idx  = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # 4. Nivel de confianza en lenguaje natural
    if confidence >= 0.85:
        conf_label = "Alta confianza"
    elif confidence >= 0.65:
        conf_label = "Confianza media"
    else:
        conf_label = "Baja confianza — revisar manualmente"

    return {
        "class":       pred_class,
        "confidence":  confidence,
        "conf_label":  conf_label,
        "probs": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(probs))
        },
        "config": CLASS_CONFIG[pred_class]
    }


if __name__ == "__main__":
    # Test rápido
    extractor  = load_feature_extractor()
    classifier = load_classifier()

    # Crear imagen de prueba
    test_img = Image.new("RGB", (224, 224), color=(255, 100, 0))
    result   = predict(test_img, extractor, classifier)

    print(f"\n✅ Test predictor.py:")
    print(f"   Clase:      {result['class']}")
    print(f"   Confianza:  {result['confidence']*100:.1f}%")
    print(f"   Probs:      {result['probs']}")