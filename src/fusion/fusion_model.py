"""Experimental Logistic Regression fusion inference."""

from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fusion_model.pkl"


def predict_fusion(face_score, speech_score, movement_score):
    scores = np.array(
        [[face_score, speech_score, movement_score]],
        dtype=float,
    )

    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Fusion inputs must be finite scores between 0 and 1.")

    if not MODEL_PATH.exists():
        return {
            "status": "model_not_trained",
            "score": None,
            "method": "logistic_regression",
            "note": "Paired labeled multimodal training data required.",
        }

    model = joblib.load(MODEL_PATH)

    probability = float(model.predict_proba(scores)[0, 1])

    return {
        "status": "success",
        "score": probability,
        "method": "logistic_regression",
        "note": (
            "Experimental model output. Not a clinically validated "
            "stroke probability or medical diagnosis."
        ),
    }
