from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

df = pd.read_csv(
    "outputs/movement/movement_pose_features.csv"
)

df["target"] = df["label"].map({
    "Complete": 0,
    "Incomplete": 1
})

threshold = 0.5

train = df[
    (df["split"] == "train") &
    (df["pose_detection_rate"] >= threshold)
].copy()

val = df[
    (df["split"] == "validation") &
    (df["pose_detection_rate"] >= threshold)
].copy()

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

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

selector = make_model()
selector.fit(train[features], train["target"])

coef = np.abs(
    selector.named_steps["classifier"].coef_[0]
)

ranked = [
    features[i] for i in np.argsort(coef)[::-1]
]

results = []

for n in [10, 20, 30, len(features)]:
    selected = ranked[:n]

    model = make_model()
    model.fit(train[selected], train["target"])

    prob = model.predict_proba(
        val[selected]
    )[:, 1]

    pred = (prob >= 0.5).astype(int)
    y = val["target"]

    results.append({
        "feature_count": n,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, prob)
    })

result = pd.DataFrame(results)

print("\n=== FEATURE SELECTION VALIDATION ===")
print(result.to_string(index=False))

out = Path(
    "review_results/movement/feature_selection.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)

print("\nSaved:", out)
