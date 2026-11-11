from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path("review_results")
OUT = ROOT / "movement_speech_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

def evaluate(name, y, pred, prob, labels):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    prob = np.asarray(prob, dtype=float)

    assert len(y) == len(pred) == len(prob)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(pred)).issubset({0, 1})
    assert np.isfinite(prob).all()
    assert ((prob >= 0) & (prob <= 1)).all()

    metrics = {
        "module": name,
        "samples": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y, prob)
            if len(np.unique(y)) == 2 else np.nan
        )
    }

    print("\n==========", name.upper(), "==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        cm, display_labels=labels
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{name} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"{name} - Test ROC Curve"
        )
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT / f"{name}_roc.png", dpi=300)
        plt.close()

    return metrics

def check_overlap(df, group, name):
    print(f"\n{name} split distribution:")
    print(df.groupby("split").agg(
        rows=(group, "size"),
        groups=(group, "nunique")
    ))
    splits = list(df["split"].dropna().unique())
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            aa = set(df.loc[df["split"] == a, group])
            bb = set(df.loc[df["split"] == b, group])
            overlap = aa & bb
            print(f"{a} vs {b}: {len(overlap)} overlapping groups")
            if overlap:
                print("  Example:", list(overlap)[:10])

movement = pd.read_csv(
    "outputs/movement/movement_test_predictions.csv"
)
movement_split = pd.read_csv(
    "outputs/movement/movement_split.csv"
)
speech = pd.read_csv(
    "outputs/speech/speech_final_test_predictions.csv"
)
speech_split = pd.read_csv(
    "outputs/speech/torgo_split.csv"
)

# Movement person ID from original filename.
movement_split["person_id"] = (
    pd.to_numeric(movement_split["id_part1"], errors="raise")
)
check_overlap(movement_split, "person_id", "Movement")
check_overlap(speech_split, "speaker_id", "Speech")

assert movement["split"].eq("test").all()
assert speech["speech_status"].isin(
    ["healthy", "dysarthria"]
).all(), "Unexpected speech labels; inspect before evaluation."

# Verify each test audio belongs to the designated test split.
test_audio = speech_split.loc[
    speech_split["split"] == "test", "audio_path"
]
if test_audio.duplicated().any():
    print("WARNING: Duplicate audio_path in speech test split")
if not speech["audio_path"].isin(set(test_audio)).all():
    raise RuntimeError(
        "Some speech predictions are not in the test split."
    )

results = []
results.append(evaluate(
    "movement",
    movement["true_target"],
    movement["predicted_target"],
    movement["probability_incomplete"],
    ["Complete", "Incomplete"]
))

results.append(evaluate(
    "speech",
    speech["speech_status"].eq("dysarthria").astype(int),
    speech["prediction"],
    speech["dysarthria_probability"],
    ["Healthy", "Dysarthria"]
))

pd.DataFrame(results).to_csv(
    OUT / "test_metrics.csv", index=False
)
print("\nSaved results to:", OUT)
