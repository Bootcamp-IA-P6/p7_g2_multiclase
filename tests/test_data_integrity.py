"""
Tests de integridad del dataset y archivos de configuración.
Validan que los datos están correctamente preparados
antes de entrenar o servir el modelo.
"""

import json
import os
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
EXCLUDED_PATH = PROJECT_ROOT / "data" / "nonfire_excluded.json"
MODEL_PATH    = PROJECT_ROOT / "models" / "02_model.pkl"
ENV_EXAMPLE   = PROJECT_ROOT / ".env.example"


class TestProjectStructure:
    """Verifica que la estructura de carpetas del proyecto es correcta."""

    def test_models_directory_exists(self):
        """La carpeta models/ debe existir."""
        assert (PROJECT_ROOT / "models").exists(), \
            "❌ No existe la carpeta models/"

    def test_model_file_exists(self):
        """El archivo del modelo debe existir."""
        assert MODEL_PATH.exists(), \
            f"❌ No se encuentra {MODEL_PATH}"

    def test_model_file_not_empty(self):
        """El modelo no debe ser un archivo vacío."""
        size = MODEL_PATH.stat().st_size
        assert size > 1000, \
            f"❌ El modelo parece estar vacío: {size} bytes"

    def test_app_directory_exists(self):
        """La carpeta app/ debe existir."""
        assert (PROJECT_ROOT / "app").exists(), \
            "❌ No existe la carpeta app/"

    def test_app_main_exists(self):
        """El archivo principal de la app debe existir."""
        app_file = (PROJECT_ROOT / "app" / "app.py").exists() or \
                (PROJECT_ROOT / "app" / "main.py").exists()
        assert app_file, \
            "❌ No se encuentra app/app.py ni app/main.py"

    def test_src_directory_exists(self):
        """La carpeta src/ debe existir."""
        assert (PROJECT_ROOT / "src").exists(), \
            "❌ No existe la carpeta src/"

    def test_data_loader_exists(self):
        """El data loader debe existir."""
        assert (PROJECT_ROOT / "src" / "data_loader.py").exists(), \
            "❌ No se encuentra src/data_loader.py"

    def test_env_example_exists(self):
        """El archivo .env.example debe existir en el repo."""
        assert ENV_EXAMPLE.exists(), \
            "❌ No se encuentra .env.example — añádelo al repo"

    def test_env_example_has_required_keys(self):
        """El .env.example debe tener las variables necesarias."""
        content = ENV_EXAMPLE.read_text()
        required_keys = ["KAGGLE_USERNAME", "KAGGLE_KEY",
                        "SUPABASE_URL", "SUPABASE_KEY"]
        for key in required_keys:
            assert key in content, \
                f"❌ Falta {key} en .env.example"


class TestExcludedImages:
    """Verifica el archivo de imágenes excluidas del EDA."""

    def test_excluded_file_exists(self):
        """El archivo de exclusiones debe existir."""
        assert EXCLUDED_PATH.exists(), \
            f"❌ No se encuentra {EXCLUDED_PATH}"

    def test_excluded_file_valid_json(self):
        """El archivo de exclusiones debe ser JSON válido."""
        try:
            with open(EXCLUDED_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"❌ JSON inválido en {EXCLUDED_PATH}: {e}")

    def test_excluded_file_has_required_keys(self):
        """El JSON de exclusiones debe tener las claves esperadas."""
        with open(EXCLUDED_PATH) as f:
            data = json.load(f)
        assert "filenames" in data, \
            "❌ Falta la clave 'filenames' en nonfire_excluded.json"
        assert "total" in data, \
            "❌ Falta la clave 'total' en nonfire_excluded.json"

    def test_excluded_filenames_is_list(self):
        """La lista de exclusiones debe ser una lista."""
        with open(EXCLUDED_PATH) as f:
            data = json.load(f)
        assert isinstance(data["filenames"], list), \
            "❌ 'filenames' debe ser una lista"

    def test_excluded_total_matches_list(self):
        """El total declarado debe coincidir con la longitud de la lista."""
        with open(EXCLUDED_PATH) as f:
            data = json.load(f)
        assert data["total"] == len(data["filenames"]), \
            f"❌ 'total' ({data['total']}) no coincide con " \
            f"len(filenames) ({len(data['filenames'])})"

    def test_excluded_filenames_are_strings(self):
        """Todos los nombres de archivo deben ser strings."""
        with open(EXCLUDED_PATH) as f:
            data = json.load(f)
        for name in data["filenames"]:
            assert isinstance(name, str), \
                f"❌ Nombre de archivo no es string: {name}"


class TestReportsDirectory:
    """Verifica que los reportes del EDA existen."""

    def test_reports_directory_exists(self):
        """La carpeta reports/ debe existir."""
        assert (PROJECT_ROOT / "reports").exists(), \
            "❌ No existe la carpeta reports/"

    def test_figures_directory_exists(self):
        """La carpeta reports/figures/ debe existir."""
        assert (PROJECT_ROOT / "reports" / "figures").exists(), \
            "❌ No existe la carpeta reports/figures/"

    def test_ensemble_comparison_exists(self):
        """El JSON de comparación de modelos debe existir."""
        json_path = PROJECT_ROOT / "reports" / "02_ensemble_comparison.json"
        assert json_path.exists(), \
            "❌ No se encuentra reports/02_ensemble_comparison.json"

    def test_ensemble_comparison_valid(self):
        """El JSON de comparación debe ser válido y tener modelos."""
        json_path = PROJECT_ROOT / "reports" / "02_ensemble_comparison.json"
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) > 0, \
            "❌ El JSON de comparación está vacío"
        assert all(isinstance(v, float) for v in data.values()), \
            "❌ Los valores del JSON deben ser floats (F1 scores)"
