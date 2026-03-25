# app/database.py
"""
Gestión de base de datos con Supabase (PostgreSQL en la nube)
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ── Conexión ──────────────────────────────────────────────────
def get_client() -> Client:
    """Crea y devuelve el cliente de Supabase."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "❌ Faltan credenciales de Supabase en el .env\n"
            "   Añade SUPABASE_URL y SUPABASE_KEY"
        )
    return create_client(url, key)


# ── Operaciones ───────────────────────────────────────────────
def init_db():
    """Verifica la conexión con Supabase."""
    try:
        client = get_client()
        client.table("predictions").select("id").limit(1).execute()
        print("✅ Conexión con Supabase establecida")
    except Exception as e:
        print(f"❌ Error conectando con Supabase: {e}")
        raise


def save_prediction(
    filename:   str,
    prediction: str,
    confidence: float,
    probs:      dict
) -> int:
    """
    Guarda una predicción en Supabase.
    Devuelve el ID de la predicción guardada.
    """
    client = get_client()
    data   = {
        "timestamp":   datetime.now().isoformat(),
        "filename":    filename,
        "prediction":  prediction,
        "confidence":  confidence,
        "prob_fire":   probs.get("fire",     0.0),
        "prob_smoke":  probs.get("smoke",    0.0),
        "prob_nonfire": probs.get("non fire", 0.0)
    }
    response = client.table("predictions").insert(data).execute()
    return response.data[0]["id"]


def save_feedback(prediction_id: int, feedback: str):
    """Guarda el feedback del operador."""
    client = get_client()
    client.table("predictions").update(
        {"feedback": feedback}
    ).eq("id", prediction_id).execute()


def get_history(limit: int = 50) -> list:
    """Devuelve las últimas predicciones."""
    client   = get_client()
    response = client.table("predictions") \
                    .select("*") \
                    .order("timestamp", desc=True) \
                    .limit(limit) \
                    .execute()
    return response.data


def get_stats() -> dict:
    """Devuelve estadísticas para el dashboard."""
    client = get_client()

    # Total predicciones
    all_preds = client.table("predictions") \
                    .select("prediction, feedback") \
                    .execute().data

    total     = len(all_preds)
    fire      = sum(1 for r in all_preds if r["prediction"] == "fire")
    smoke     = sum(1 for r in all_preds if r["prediction"] == "smoke")
    non_fire  = sum(1 for r in all_preds if r["prediction"] == "non fire")

    with_feedback = sum(1 for r in all_preds if r["feedback"] is not None)
    correct       = sum(1 for r in all_preds if r["feedback"] == "correct")

    return {
        "total":          total,
        "fire":           fire,
        "smoke":          smoke,
        "non_fire":       non_fire,
        "with_feedback":  with_feedback,
        "correct":        correct,
        "accuracy_feedback": round(correct / with_feedback * 100, 1)
                            if with_feedback > 0 else 0
    }
    
def save_feedback(
    prediction_id: int,
    feedback_type: str,
    true_label:    str  = None,
    comment:       str  = None
):

    client = get_client()
    data   = {
        "prediction_id": prediction_id,
        "timestamp":     datetime.now().isoformat(),
        "feedback_type": feedback_type,
        "true_label":    true_label,
        "comment":       comment
    }
    client.table("feedback").insert(data).execute()
    print(f"✅ Feedback guardado para predicción {prediction_id}")


def get_feedback_history(limit: int = 50) -> list:
    """Devuelve el historial de feedback con info de la predicción."""
    client   = get_client()
    response = client.table("feedback") \
                    .select("*, predictions(filename, prediction, confidence)") \
                    .order("timestamp", desc=True) \
                    .limit(limit) \
                    .execute()
    return response.data


def get_feedback_stats() -> dict:
    """Estadísticas del feedback para el dashboard."""
    client = get_client()
    all_fb = client.table("feedback") \
                .select("feedback_type, true_label") \
                .execute().data

    total     = len(all_fb)
    correct   = sum(1 for r in all_fb if r["feedback_type"] == "correct")
    incorrect = sum(1 for r in all_fb if r["feedback_type"] == "incorrect")
    unsure    = sum(1 for r in all_fb if r["feedback_type"] == "unsure")

    return {
        "total":     total,
        "correct":   correct,
        "incorrect": incorrect,
        "unsure":    unsure,
        "accuracy":  round(correct / total * 100, 1) if total > 0 else 0
    }