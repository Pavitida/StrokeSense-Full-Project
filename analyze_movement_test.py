from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

OUT = Path("review_results/movement/deep_analysis")
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(
    "outputs/movement/movement_test_predictions.csv"
)
split = pd.read_csv(
    "outputs/movement/movement_split.csv"
)

pred["correct"] = (
    pred["true_target"] == pred["predicted_target"]
)

print("\n========== 1. OVERALL TEST ==========")
print("Evaluated:", len(pred))
print("Correct:", pred["correct"].sum())
print("Incorrect:", (~pred["correct"]).sum())

cm = confusion_matrix(
    pred["true_target"],
    pred["predicted_target"],
    labels=[0, 1]
)
print("\nConfusion Matrix:")
print(cm)
print("Rows: Actual Complete, Incomplete")
print("Cols: Predicted Complete, Incomplete")

print("\n========== 2. BY EXERCISE ==========")

rows = []
for name, group in pred.groupby("exercise"):
    y = group["true_target"]
    p = group["predicted_target"]
    score = group["probability_incomplete"]

    result = {
        "exercise": name,
        "samples": len(group),
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(
            y, p, zero_division=0
        ),
        "recall": recall_score(
            y, p, zero_division=0
        ),
        "f1": f1_score(
            y, p, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y, score)
            if y.nunique() == 2 else np.nan
        ),
        "mean_pose_detection_rate":
            group["pose_detection_rate"].mean()
    }
    rows.append(result)

by_exercise = pd.DataFrame(rows)
print(by_exercise.to_string(index=False))
by_exercise.to_csv(
    OUT / "by_exercise.csv", index=False
)

print("\n========== 3. ERROR TYPES ==========")

pred["error_type"] = np.select(
    [
        (pred.true_target == 0) &
        (pred.predicted_target == 1),
        (pred.true_target == 1) &
        (pred.predicted_target == 0)
    ],
    [
        "False Positive",
        "False Negative"
    ],
    default="Correct"
)

print(pred["error_type"].value_counts())
print("\nBy exercise:")
print(pd.crosstab(
    pred["exercise"], pred["error_type"]
))

print("\n========== 4. POSE DETECTION ==========")

pred["pose_quality"] = pd.cut(
    pred["pose_detection_rate"],
    bins=[-0.001, 0.5, 0.7, 0.9, 1.001],
    labels=["<50%", "50-70%", "70-90%", ">90%"]
)

print(pd.crosstab(
    pred["pose_quality"],
    pred["correct"],
    margins=True
))

print("\n========== 5. BY PERSON ==========")

by_person = (
    pred.groupby("person_id")
    .agg(
        samples=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_pose_detection_rate=(
            "pose_detection_rate", "mean"
        )
    )
    .reset_index()
)

print(by_person.to_string(index=False))
by_person.to_csv(
    OUT / "by_person.csv", index=False
)

print("\n========== 6. MISSING TEST VIDEOS ==========")

expected = split[
    split["split"].astype(str).str.lower() == "test"
].copy()

print("Expected test videos:", len(expected))
print("Evaluated test videos:", len(pred))
print("Difference:", len(expected) - len(pred))

print("\nExpected by exercise:")
print(expected["exercise"].value_counts())

print("\nEvaluated by exercise:")
print(pred["exercise"].value_counts())

print("\nExpected by label:")
print(expected["label"].value_counts())

print("\nEvaluated by label:")
print(pred["label"].value_counts())

failed_path = Path(
    "outputs/movement/movement_failed_videos.txt"
)
if failed_path.exists():
    print("\nFailed video log:")
    print(failed_path.read_text()[:5000])

pred.to_csv(
    OUT / "test_with_error_types.csv", index=False
)

print("\nSaved analysis to:", OUT)
