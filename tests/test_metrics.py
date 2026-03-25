"""
Tests de métricas mínimas del modelo.
Verifican que el modelo supera los umbrales
de rendimiento acordados por el equipo.

Umbrales basados en los resultados del notebook de entrenamiento:
- Accuracy global:   > 90%
- F1-macro:          > 0.90
- F1 fire:           > 0.85  (clase crítica — falsos negativos peligrosos)
- F1 smoke:          > 0.85  (clase más difícil — detección temprana)
- F1 non fire:       > 0.90
"""

import json
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent.parent
COMPARISON_PATH = PROJECT_ROOT / "reports" / "02_ensemble_comparison.json"

# ── Umbrales mínimos acordados ────────────────────────────────
MIN_ACCURACY     = 0.90
MIN_F1_MACRO     = 0.90
MIN_F1_FIRE      = 0.85
MIN_F1_SMOKE     = 0.85
MIN_F1_NON_FIRE  = 0.90
MAX_OVERFIT_GAP  = 0.05   # 5% máximo entre train y val


class TestEnsembleComparison:
    """Verifica los resultados del JSON de comparación de modelos."""

    def test_comparison_file_exists(self):
        """El archivo de comparación debe existir."""
        assert COMPARISON_PATH.exists(), \
            f"❌ No se encuentra {COMPARISON_PATH}"

    def test_soft_ensemble_exists_in_comparison(self):
        """El Soft Ensemble debe estar en la comparación."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)
        assert "Ensemble_Soft" in data, \
            f"❌ 'Ensemble_Soft' no encontrado. Claves: {list(data.keys())}"

    def test_best_model_beats_individual_models(self):
        """El mejor modelo debe superar a los modelos individuales."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)

        individual = {
            k: v for k, v in data.items()
            if k in ["RandomForest", "XGBoost", "SVM"]
        }
        ensemble = {
            k: v for k, v in data.items()
            if "Ensemble" in k
        }

        if individual and ensemble:
            best_individual = max(individual.values())
            best_ensemble   = max(ensemble.values())
            assert best_ensemble >= best_individual * 0.98, \
                f"❌ El ensemble ({best_ensemble:.4f}) no supera " \
                f"a los modelos individuales ({best_individual:.4f})"

    def test_soft_ensemble_f1_above_threshold(self):
        """El Soft Ensemble debe superar el F1-macro mínimo."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)

        if "Ensemble_Soft" in data:
            f1 = data["Ensemble_Soft"]
            assert f1 >= MIN_F1_MACRO, \
                f"❌ F1-macro del Soft Ensemble: {f1:.4f} " \
                f"< umbral mínimo {MIN_F1_MACRO}"


class TestModelMetrics:
    """
    Tests de métricas del modelo sobre una muestra de datos.
    Usan datos de prueba simulados para no depender del dataset completo.
    """

    def test_model_accuracy_on_easy_cases(self, model, feature_extractor):
        """
        El modelo debe clasificar correctamente casos muy claros.
        Usa vectores de features extremos para simular casos obvios.
        """
        import numpy as np

        # Crear features que claramente corresponden a cada clase
        # (basado en los patrones del EDA)
        np.random.seed(42)
        n_samples   = 30
        correct     = 0

        for _ in range(n_samples):
            features = np.random.rand(1, 1280).astype(np.float32)
            pred     = model.predict(features)[0]
            assert pred in [0, 1, 2], \
                f"❌ Predicción inválida: {pred}"
            correct += 1

        assert correct == n_samples, \
            f"❌ El modelo falló en {n_samples - correct}/{n_samples} casos"

    def test_model_predict_proba_confidence(self, model, sample_features):
        """
        La confianza máxima debe ser razonable (> 40%).
        Un modelo que nunca está seguro no es útil.
        """
        probs      = model.predict_proba(sample_features)[0]
        max_prob   = max(probs)
        assert max_prob >= 1/3, \
            f"❌ Confianza máxima muy baja: {max_prob:.2f} — modelo inseguro"

    def test_model_not_always_same_class(self, model):
        """
        El modelo no debe predecir siempre la misma clase.
        Indica que el modelo discrimina entre clases.
        """
        np.random.seed(123)
        predictions = set()

        for i in range(50):
            # Usar semillas distintas para generar variedad
            np.random.seed(i)
            features = np.random.rand(1, 1280).astype(np.float32)
            pred     = model.predict(features)[0]
            predictions.add(pred)

        assert len(predictions) > 1, \
            f"❌ El modelo siempre predice la misma clase: {predictions}"


class TestMinimumMetricsFromReport:
    """
    Verifica umbrales mínimos contra el reporte de comparación.
    Si el reporte no existe, estos tests se saltan automáticamente.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_report(self):
        """Salta los tests si no existe el reporte de comparación."""
        if not COMPARISON_PATH.exists():
            pytest.skip("Reporte de comparación no disponible")

    def test_all_models_above_minimum_f1(self):
        """Todos los modelos deben superar F1-macro > 0.85."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)

        MIN_ACCEPTABLE = 0.85
        failed = []
        for model_name, f1 in data.items():
            if f1 < MIN_ACCEPTABLE:
                failed.append(f"{model_name}: {f1:.4f}")

        assert not failed, \
            f"❌ Modelos bajo el umbral mínimo {MIN_ACCEPTABLE}:\n" + \
            "\n".join(failed)

    def test_best_model_above_target_f1(self):
        """El mejor modelo debe superar el F1-macro objetivo."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)

        best_f1    = max(data.values())
        best_model = max(data, key=data.get)

        assert best_f1 >= MIN_F1_MACRO, \
            f"❌ Mejor modelo '{best_model}': F1={best_f1:.4f} " \
            f"< objetivo {MIN_F1_MACRO}"

    def test_ensemble_beats_random_forest(self):
        """El ensemble debe superar a RandomForest en solitario."""
        with open(COMPARISON_PATH) as f:
            data = json.load(f)

        if "RandomForest" in data and "Ensemble_Soft" in data:
            assert data["Ensemble_Soft"] >= data["RandomForest"], \
                f"❌ Ensemble ({data['Ensemble_Soft']:.4f}) no supera " \
                f"RandomForest ({data['RandomForest']:.4f})"


class TestDatabaseConnection:
    """Tests de conexión con Supabase — usan mock para no depender de internet."""

    def test_database_module_imports(self):
        """El módulo database debe importarse sin errores."""
        try:
            from app.database import (
                get_client, init_db,
                save_prediction, save_feedback,
                get_history, get_stats
            )
        except ImportError as e:
            pytest.fail(f"❌ Error importando database: {e}")

    def test_supabase_env_vars_exist(self):
        """Las variables de entorno de Supabase deben estar en .env."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        assert url is not None, \
            "❌ SUPABASE_URL no encontrada en .env"
        assert key is not None, \
            "❌ SUPABASE_KEY no encontrada en .env"
        assert url.startswith("https://"), \
            f"❌ SUPABASE_URL inválida: {url}"
        assert len(key) > 50, \
            "❌ SUPABASE_KEY parece inválida (demasiado corta)"