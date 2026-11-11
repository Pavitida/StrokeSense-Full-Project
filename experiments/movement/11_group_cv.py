from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

df = pd.read_csv(
    "outputs/movement/movement_pose_features.csv"
)

df = df[
    df["split"].isin(["train", "validation"])
].copy()

df = df[
    df["pose_detection_rate"] >= 0.5
].copy()

df["target"] = df["label"].map({
    "Complete": 0,
    "Incomplete": 1
})

exclude = {
    "person_id", "target", "video_fps",
    "video_total_frames", "processed_frames",
    "detected_frames", "pose_detection_rate"
}

features = [
    c for c in df.select_dtypes(
        include=[np.number]
    ).columns if c not in exclude
]

X = df[features]
y = df["target"]
groups = df["person_id"]

n_groups = groups.nunique()
n_splits = min(5, n_groups)

if n_splits < 2:
    raise ValueError("Not enough people for GroupKFold")

cv = GroupKFold(n_splits=n_splits)

results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    prob = model.predict_proba(
        X.iloc[val_idx]
    )[:, 1]

    pred = (prob >= 0.5).astype(int)
    truth = y.iloc[val_idx]

    auc = (
        roc_auc_score(truth, prob)
        if truth.nunique() == 2
        else np.nan
    )

    results.append({
        "fold": fold,
        "people": groups.iloc[val_idx].nunique(),
        "samples": len(val_idx),
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(
            truth, pred, zero_division=0
        ),
        "recall": recall_score(
            truth, pred, zero_division=0
        ),
        "f1": f1_score(
            truth, pred, zero_division=0
        ),
        "roc_auc": auc
    })

result = pd.DataFrame(results)

print("\n=== GROUP CROSS-VALIDATION ===")
print(result.to_string(index=False))

print("\n=== MEAN METRICS ===")
print(
    result[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean().to_string()
)

out = Path(
    "review_results/movement/group_cv.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)

print("\nSaved:", out)
