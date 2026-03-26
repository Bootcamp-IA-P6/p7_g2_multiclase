"""
Tests de funcionamiento del pipeline de inferencia.
Verifican que el modelo recibe el input correcto
y devuelve el output esperado.
"""

import pytest
import numpy as np
from PIL import Image
from sklearn.ensemble import VotingClassifier


class TestModelLoading:
    """Verifica que el modelo se carga correctamente."""

    def test_model_is_voting_classifier(self, model):
        """El modelo debe ser un VotingClassifier."""
        assert isinstance(model, VotingClassifier), \
            f"❌ Se esperaba VotingClassifier, se obtuvo {type(model)}"

    def test_model_has_three_classes(self, model):
        """El modelo debe tener exactamente 3 clases."""
        assert len(model.classes_) == 3, \
            f"❌ Se esperaban 3 clases, se obtuvieron {len(model.classes_)}"

    def test_model_has_estimators(self, model):
        """El modelo debe tener los 3 estimadores base."""
        estimator_names = [name for name, _ in model.estimators]
        assert "rf"  in estimator_names, "❌ Falta RandomForest (rf)"
        assert "xgb" in estimator_names, "❌ Falta XGBoost (xgb)"
        assert "svm" in estimator_names, "❌ Falta SVM (svm)"

    def test_model_is_fitted(self, model):
        """El modelo debe estar entrenado."""
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(model)
        except Exception:
            pytest.fail("❌ El modelo no está entrenado (not fitted)")


class TestFeatureExtractor:
    """Verifica que EfficientNetB0 extrae features correctamente."""

    def test_extractor_output_shape(self, feature_extractor, sample_image_rgb):
        """EfficientNetB0 debe devolver un vector de 1280 features."""
        import numpy as np
        from tensorflow.keras.applications.efficientnet import preprocess_input

        arr = np.array(sample_image_rgb.resize((224, 224)),
                    dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        features = feature_extractor.predict(arr, verbose=0)
        assert features.shape == (1, 1280), \
            f"❌ Shape incorrecto: {features.shape}, esperado (1, 1280)"

    def test_extractor_output_dtype(self, feature_extractor, sample_image_rgb):
        """Las features deben ser float32."""
        import numpy as np
        from tensorflow.keras.applications.efficientnet import preprocess_input

        arr = np.array(sample_image_rgb.resize((224, 224)),
                    dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        features = feature_extractor.predict(arr, verbose=0)
        assert features.dtype in [np.float32, np.float64], \
            f"❌ dtype incorrecto: {features.dtype}"


class TestInferencePipeline:
    """Verifica el pipeline completo de inferencia."""

    def test_predict_returns_probabilities(self, model, sample_features):
        """El modelo debe devolver probabilidades para 3 clases."""
        probs = model.predict_proba(sample_features)
        assert probs.shape == (1, 3), \
            f"❌ Shape incorrecto: {probs.shape}, esperado (1, 3)"

    def test_probabilities_sum_to_one(self, model, sample_features):
        """Las probabilidades deben sumar 1."""
        probs = model.predict_proba(sample_features)[0]
        assert abs(probs.sum() - 1.0) < 1e-6, \
            f"❌ Las probabilidades no suman 1: {probs.sum()}"

    def test_probabilities_between_zero_and_one(self, model, sample_features):
        """Todas las probabilidades deben estar entre 0 y 1."""
        probs = model.predict_proba(sample_features)[0]
        assert all(0 <= p <= 1 for p in probs), \
            f"❌ Probabilidades fuera de rango [0,1]: {probs}"

    def test_predict_returns_valid_class(self, model, sample_features):
        """La predicción debe ser una de las 3 clases válidas."""
        pred = model.predict(sample_features)[0]
        assert pred in [0, 1, 2], \
            f"❌ Clase predicha inválida: {pred}"

    def test_image_preprocessing(self, sample_image_rgb):
        """El preprocesamiento de imagen debe producir el shape correcto."""
        import numpy as np
        from tensorflow.keras.applications.efficientnet import preprocess_input

        img_resized = sample_image_rgb.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        assert arr.shape == (1, 224, 224, 3), \
            f"❌ Shape incorrecto tras preprocesamiento: {arr.shape}"

    def test_rgba_image_converts_to_rgb(self):
        """Imágenes RGBA deben convertirse a RGB sin error."""
        rgba_img = Image.new("RGBA", (224, 224), color=(255, 0, 0, 128))
        rgb_img  = rgba_img.convert("RGB")
        assert rgb_img.mode == "RGB", \
            f"❌ La conversión RGBA→RGB falló: modo={rgb_img.mode}"

    def test_different_image_sizes_work(self, model, feature_extractor):
        """Imágenes de distintos tamaños deben funcionar tras resize."""
        import numpy as np
        from tensorflow.keras.applications.efficientnet import preprocess_input

        for size in [(100, 100), (640, 480), (1920, 1080)]:
            img  = Image.new("RGB", size, color=(100, 100, 100))
            img  = img.resize((224, 224))
            arr  = np.array(img, dtype=np.float32)
            arr  = np.expand_dims(arr, axis=0)
            arr  = preprocess_input(arr)
            feat = feature_extractor.predict(arr, verbose=0)
            pred = model.predict(feat)
            assert pred[0] in [0, 1, 2], \
                f"❌ Fallo con imagen de tamaño {size}"


class TestPredictorModule:
    """Verifica las funciones del módulo app/predictor.py."""

    def test_predictor_imports(self):
        """El módulo predictor debe importarse sin errores."""
        try:
            from app.predictor import (
                CLASS_NAMES, CLASS_CONFIG,
                preprocess_image, predict
            )
        except ImportError as e:
            pytest.fail(f"❌ Error importando predictor: {e}")

    def test_class_names_mapping(self):
        """El mapeo de clases debe tener 3 entradas."""
        from app.predictor import CLASS_NAMES
        assert len(CLASS_NAMES) == 3, \
            f"❌ CLASS_NAMES debe tener 3 entradas: {CLASS_NAMES}"
        assert 0 in CLASS_NAMES, "❌ Falta clase 0 en CLASS_NAMES"
        assert 1 in CLASS_NAMES, "❌ Falta clase 1 en CLASS_NAMES"
        assert 2 in CLASS_NAMES, "❌ Falta clase 2 en CLASS_NAMES"

    def test_class_config_has_all_classes(self):
        """CLASS_CONFIG debe tener configuración para fire, smoke y non fire."""
        from app.predictor import CLASS_CONFIG
        for cls in ["fire", "smoke", "non fire"]:
            assert cls in CLASS_CONFIG, \
                f"❌ Falta '{cls}' en CLASS_CONFIG"

    def test_class_config_has_required_keys(self):
        """Cada clase en CLASS_CONFIG debe tener las claves necesarias."""
        from app.predictor import CLASS_CONFIG
        required = ["icon", "label", "color", "message", "level"]
        for cls, cfg in CLASS_CONFIG.items():
            for key in required:
                assert key in cfg, \
                    f"❌ Falta '{key}' en CLASS_CONFIG['{cls}']"

    def test_preprocess_image_output_shape(self, sample_image_rgb):
        """preprocess_image debe devolver shape (1, 224, 224, 3)."""
        from app.predictor import preprocess_image
        arr = preprocess_image(sample_image_rgb)
        assert arr.shape == (1, 224, 224, 3), \
            f"❌ Shape incorrecto: {arr.shape}"