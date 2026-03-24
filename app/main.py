# app/app.py
"""
Aplicación Streamlit para detección de incendios forestales.
3 páginas: Analizar imagen | Historial | Dashboard
"""

import streamlit as st
from PIL import Image
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.predictor import (
    load_feature_extractor,
    load_classifier,
    predict,
    CLASS_CONFIG
)
from app.database import (
    init_db,
    save_prediction,
    save_feedback,
    get_history,
    get_stats
)

# ── Configuración de la página ────────────────────────────────
st.set_page_config(
    page_title="Forest Fire Detector",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inicialización ────────────────────────────────────────────
init_db()

@st.cache_resource
def load_models():
    """Carga los modelos una sola vez y los cachea."""
    extractor  = load_feature_extractor()
    classifier = load_classifier()
    return extractor, classifier

extractor, classifier = load_models()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/emoji/96/fire.png",
    width=80
)
st.sidebar.title("Forest Fire Detector")
st.sidebar.caption("Sistema de detección temprana de incendios forestales")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navegación",
    ["🔍 Analizar imagen",
     "📋 Historial",
     "📊 Dashboard"]
)

st.sidebar.divider()
st.sidebar.caption("Modelo: EfficientNetB0 + Soft Voting Ensemble")
st.sidebar.caption("Clases: fire · smoke · non fire")

# ══════════════════════════════════════════════════════════════
# PÁGINA 1 — Analizar imagen
# ══════════════════════════════════════════════════════════════
if page == "🔍 Analizar imagen":
    st.title("🔥 Detección de Incendios Forestales")
    st.caption(
        "Sube una imagen de cámara de vigilancia para analizar "
        "si contiene fuego, humo o ninguna incidencia."
    )
    st.divider()

    uploaded = st.file_uploader(
        "Arrastra una imagen o haz clic para seleccionar",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG"
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")

        # Realizar predicción
        with st.spinner("Analizando imagen..."):
            result = predict(img, extractor, classifier)

        # Guardar en base de datos
        pred_id = save_prediction(
            filename   = uploaded.name,
            prediction = result["class"],
            confidence = result["confidence"],
            probs      = result["probs"]
        )

        # ── Layout ────────────────────────────────────────────
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.image(
                img,
                caption=f"📁 {uploaded.name}",
                use_container_width=True
            )

        with col_result:
            cfg = result["config"]

            # Resultado principal
            if cfg["level"] == "danger":
                st.error(f"{cfg['icon']} {cfg['label']}")
            elif cfg["level"] == "warning":
                st.warning(f"{cfg['icon']} {cfg['label']}")
            else:
                st.success(f"{cfg['icon']} {cfg['label']}")

            st.markdown(f"**{result['conf_label']}** — {result['confidence']*100:.1f}%")
            st.info(f"💡 {cfg['message']}")

            # Barra de progreso de confianza
            st.progress(result["confidence"])

            # Gráfico de probabilidades
            st.markdown("#### Probabilidad por clase")
            probs_display = {
                "🔥 Fuego":      result["probs"]["fire"],
                "💨 Humo":       result["probs"]["smoke"],
                "🌲 Sin incid.": result["probs"]["non fire"]
            }
            st.bar_chart(probs_display)

            # ── Feedback ──────────────────────────────────────
            st.divider()
            st.markdown("**¿Es correcta esta predicción?**")

            fb_col1, fb_col2, fb_col3 = st.columns(3)

            if fb_col1.button("✅ Correcta", key="correct"):
                save_feedback(pred_id, "correct")
                st.success("¡Gracias por tu feedback!")

            if fb_col2.button("❌ Incorrecta", key="incorrect"):
                st.session_state[f"show_fix_{pred_id}"] = True

            if fb_col3.button("🤔 No seguro", key="unsure"):
                save_feedback(pred_id, "unsure")
                st.info("Registrado para revisión")

            # Selector clase correcta
            if st.session_state.get(f"show_fix_{pred_id}"):
                true_label = st.selectbox(
                    "¿Cuál es la clase correcta?",
                    ["fire", "smoke", "non fire"],
                    key=f"fix_{pred_id}"
                )
                if st.button("Confirmar", key=f"confirm_{pred_id}"):
                    save_feedback(pred_id, f"incorrect:{true_label}")
                    st.success(f"Registrado — etiquetada como '{true_label}'")
                    st.session_state[f"show_fix_{pred_id}"] = False

# ══════════════════════════════════════════════════════════════
# PÁGINA 2 — Historial
# ══════════════════════════════════════════════════════════════
elif page == "📋 Historial":
    st.title("📋 Historial de predicciones")
    st.caption("Últimas 50 imágenes analizadas por el sistema.")
    st.divider()

    rows = get_history(limit=50)

    if not rows:
        st.info("⏳ Aún no hay predicciones. Ve a 'Analizar imagen' para empezar.")
    else:
        # Métricas rápidas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",         len(rows))
        m2.metric("🔥 Fuego",      sum(1 for r in rows if r["prediction"] == "fire"))
        m3.metric("💨 Humo",       sum(1 for r in rows if r["prediction"] == "smoke"))
        m4.metric("🌲 Sin incid.", sum(1 for r in rows if r["prediction"] == "non fire"))

        st.divider()

        # Tabla de historial
        df = pd.DataFrame(rows)
        df["timestamp"]  = pd.to_datetime(df["timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"

        # Iconos por clase
        icons = {"fire": "🔥", "smoke": "💨", "non fire": "🌲"}
        df["prediction"] = df["prediction"].map(
            lambda x: f"{icons.get(x, '')} {x}"
        )

        st.dataframe(
            df[["timestamp", "filename", "prediction",
                "confidence", "feedback"]].rename(columns={
                "timestamp":  "Fecha",
                "filename":   "Archivo",
                "prediction": "Predicción",
                "confidence": "Confianza",
                "feedback":   "Feedback"
            }),
            use_container_width=True,
            hide_index=True
        )

# ══════════════════════════════════════════════════════════════
# PÁGINA 3 — Dashboard
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Dashboard del sistema")
    st.caption("Métricas de uso y rendimiento del modelo en producción.")
    st.divider()

    stats = get_stats()

    if stats["total"] == 0:
        st.info("⏳ Sin datos aún. Analiza algunas imágenes primero.")
    else:
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total predicciones", stats["total"])
        col2.metric("🔥 Fuego",           stats["fire"])
        col3.metric("💨 Humo",            stats["smoke"])
        col4.metric("🌲 Sin incidencia",  stats["non_fire"])

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Distribución de predicciones")
            dist_data = {
                "🔥 Fuego":      stats["fire"],
                "💨 Humo":       stats["smoke"],
                "🌲 Sin incid.": stats["non_fire"]
            }
            st.bar_chart(dist_data)

        with col_right:
            st.markdown("#### Sistema de feedback")
            st.metric(
                "Predicciones con feedback",
                stats["with_feedback"],
                f"{stats['with_feedback']}/{stats['total']}"
            )
            if stats["with_feedback"] > 0:
                st.metric(
                    "Predicciones correctas",
                    f"{stats['accuracy_feedback']}%",
                    f"{stats['correct']} confirmadas"
                )
                st.progress(stats["accuracy_feedback"] / 100)

        st.divider()
        st.markdown("#### Información del modelo")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.info("**Extractor**\nEfficientNetB0\nImageNet weights")
        col_m2.info("**Clasificador**\nSoft Voting Ensemble\nRF + XGBoost + SVM")
        col_m3.info("**Input**\n224×224 px\nRGB normalizado")