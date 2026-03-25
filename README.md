# 🔥 Forest Fire Detector

Sistema de detección temprana de incendios forestales mediante análisis de imágenes con deep learning. Clasifica imágenes en tres categorías: **fire**, **smoke** y **non fire**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase)

---

## Cómo funciona

El sistema combina una CNN preentrenada con un ensemble de clasificadores clásicos:

```
Imagen → EfficientNetB0 (features 1280-d) → VotingClassifier (RF + XGBoost + SVM) → {fire, smoke, non fire}
```

1. **EfficientNetB0** (ImageNet, pesos congelados, sin cabeza) extrae un vector de 1280 características por imagen.
2. Un **ensemble** de Random Forest, XGBoost y SVM entrenado sobre esos vectores produce las probabilidades finales.
3. **Streamlit** expone la interfaz web, **Supabase** persiste las predicciones y el feedback de operadores.

---

## Resultados del modelo

Validación cruzada estratificada 3-fold (StratifiedKFold, K=3):

| Pliegue | Accuracy | F1-Macro |
|---------|----------|----------|
| Fold 1  | 0.9833   | 0.9833   |
| Fold 2  | 0.9839   | 0.9839   |
| Fold 3  | 0.9826   | 0.9825   |
| **Media** | — | **0.9832** |
| Std     | —        | 0.0007   |

✅ Modelo estable: STD = 0.0007 < 0.03

---

## Estructura del proyecto

```
p7_g2_multiclase/
├── app/
│   ├── main.py               # App Streamlit (4 páginas)
│   ├── predictor.py          # Pipeline de inferencia
│   ├── database.py           # CRUD Supabase
│   └── utils.py
├── src/
│   ├── data_loader.py        # Dataset PyTorch + DataLoaders
│   ├── model_factory.py      # Construcción de modelos
│   ├── train.py              # Loop de entrenamiento
│   └── evaluate.py           # Métricas de evaluación
├── notebooks/
│   ├── 02_EDA.ipynb          # Análisis exploratorio
│   ├── 03_EDA_Joaquin.ipynb
│   ├── 02a_modeling.ipynb    # Entrenamiento del ensemble
│   └── 02b_cross_validation.ipynb
├── models/
│   └── 02_model.pkl          # VotingClassifier serializado
├── data/
│   ├── processed/            # Features .npy extraídas
│   └── nonfire_excluded.json # Imágenes filtradas en EDA
├── reports/
│   └── 02_cv_results.json
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml            # Dependencias UV
└── .env                      # Credenciales Supabase (no committear)
```

---

## Puesta en marcha

### Opción 1 — Docker (recomendado)

**Prerrequisitos:** Docker Desktop instalado y en ejecución.

```bash
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-P6/p7_g2_multiclase.git
cd p7_g2_multiclase

# 2. Crear el fichero de credenciales
cp .env.example .env
# Editar .env con tus credenciales de Supabase

# 3. Construir y levantar
docker compose up --build
```

La app estará disponible en **http://localhost:8501**

### Opción 2 — Local con UV

```bash
# Instalar UV
pip install uv

# Instalar dependencias
uv sync

# Levantar la app
uv run streamlit run app/main.py
```

---

## Variables de entorno

Crea un fichero `.env` en la raíz del proyecto:

```env
SUPABASE_URL=https://<proyecto>.supabase.co
SUPABASE_KEY=<anon-key>
```

---

## Tablas de base de datos (Supabase)

Antes de arrancar, crea las siguientes tablas en tu proyecto de Supabase:

```sql
-- Predicciones
create table predictions (
  id          serial primary key,
  timestamp   timestamptz default now(),
  filename    text,
  prediction  text,
  confidence  float,
  prob_fire   float,
  prob_smoke  float,
  prob_nonfire float,
  feedback    text
);

-- Feedback de operadores
create table feedback (
  id             serial primary key,
  prediction_id  int references predictions(id),
  timestamp      timestamptz default now(),
  feedback_type  text,  -- correct | incorrect | unsure
  true_label     text,
  comment        text
);
```

---

## Interfaz web

| Página | Descripción |
|--------|-------------|
| 🔍 Analizar imagen | Sube una imagen y obtén la predicción con nivel de alerta y probabilidades |
| 📋 Historial | Últimas 50 predicciones con métricas rápidas |
| 💬 Feedback | Valida o corrige predicciones pasadas |
| 📊 Dashboard | Distribución de clases, tasa de acierto, info del modelo |

---

## Entrenamiento

Para reentrenar el modelo:

1. Abre `notebooks/02a_modeling.ipynb`
2. Ejecuta todas las celdas — descarga el dataset vía kagglehub, extrae features con EfficientNetB0 y entrena el ensemble
3. El modelo se guarda en `models/02_model.pkl`
4. Valida la estabilidad con `notebooks/02b_cross_validation.ipynb`

**Credenciales de Kaggle necesarias** para la descarga del dataset. Configura `~/.kaggle/kaggle.json` o las variables de entorno `KAGGLE_USERNAME` / `KAGGLE_KEY`.

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.11 |
| Gestor de paquetes | UV |
| CNN feature extractor | EfficientNetB0 (TensorFlow 2.21) |
| DataLoaders | PyTorch >= 2.10 + torchvision |
| Clasificador | scikit-learn VotingClassifier / StackingClassifier |
| Boosting | XGBoost >= 3.2 |
| Interfaz web | Streamlit >= 1.55 |
| Base de datos | Supabase (PostgreSQL) |
| Contenedor | Docker + Docker Compose |
| Tests | pytest >= 9.0 |
| Monitorización | Evidently >= 0.4.36 |
| HPO | Optuna >= 4.8 |

---

## Tests

```bash
uv run pytest
```

---

## Clases y niveles de alerta

| Clase | Icono | Alerta | Acción recomendada |
|-------|-------|--------|-------------------|
| `fire` | 🔥 | 🔴 PELIGRO | Activar protocolo de emergencia inmediatamente |
| `smoke` | 💨 | 🟠 AVISO | Posible foco incipiente — verificar zona urgente |
| `non fire` | 🌲 | 🟢 OK | No se detectan señales de incendio |

---

## Licencia

Proyecto académico — p7_g2_multiclase.
