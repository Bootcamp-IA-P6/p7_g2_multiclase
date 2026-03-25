"""
Fixtures compartidos entre todos los tests.
Se ejecutan una sola vez y se reutilizan.
"""

import pytest
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock


# ── Rutas del proyecto ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "02_model.pkl"
EXCLUDED_PATH = PROJECT_ROOT / "data" / "nonfire_excluded.json"
ENV_PATH     = PROJECT_ROOT / ".env"


@pytest.fixture(scope="session")
def model():
    """Carga el modelo pkl una sola vez para toda la sesión de tests."""
    assert MODEL_PATH.exists(), (
        f"❌ Modelo no encontrado: {MODEL_PATH}\n"
        f"   Asegúrate de tener models/02_model.pkl"
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def feature_extractor():
    """
    Carga EfficientNetB0 una sola vez.
    Usa scope='session' para no recargar en cada test.
    """
    from tensorflow.keras.applications import EfficientNetB0
    return EfficientNetB0(
        weights=None,
        include_top=False,
        pooling="avg"
    )


@pytest.fixture
def sample_image_rgb():
    """Imagen RGB de prueba 224x224."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


@pytest.fixture
def sample_image_fire():
    """Imagen simulando fuego (tonos rojos/naranjas)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = 220   # Canal R alto
    arr[:, :, 1] = 80    # Canal G medio
    arr[:, :, 2] = 20    # Canal B bajo
    return Image.fromarray(arr)


@pytest.fixture
def sample_features():
    """Vector de features simulado (1280 dims como EfficientNetB0)."""
    np.random.seed(42)
    return np.random.rand(1, 1280).astype(np.float32)