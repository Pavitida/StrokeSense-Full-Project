"""Collect paired multimodal scores for fusion research."""

import csv
import math
from pathlib import Path

COLUMNS = [
    "participant_id",
    "face_score",
    "speech_score",
    "movement_score",
    "label",
]

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "fusion" / "paired_fusion_data.csv"
)


def save_fusion_sample(
    participant_id,
    face_score,
    speech_score,
    movement_score,
    label=None,
    *,
    consent=False,
):
    if not consent:
        raise ValueError("Research consent is required.")

    if not isinstance(participant_id, str) or not participant_id.strip():
        raise ValueError("A pseudonymous participant ID is required.")

    scores = [face_score, speech_score, movement_score]

    if not all(
        isinstance(s, (int, float))
        and math.isfinite(s)
        and 0 <= s <= 1
        for s in scores
    ):
        raise ValueError("All scores must be finite and between 0 and 1.")

    if label is not None and label not in (0, 1):
        raise ValueError("Label must be 0, 1, or None.")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        with DATA_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if any(r["participant_id"] == participant_id for r in rows):
            raise ValueError("Participant ID already exists.")

    is_new = not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0

    with DATA_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()

        writer.writerow({
            "participant_id": participant_id,
            "face_score": face_score,
            "speech_score": speech_score,
            "movement_score": movement_score,
            "label": "" if label is None else label,
        })

    return {"status": "saved", "participant_id": participant_id}
