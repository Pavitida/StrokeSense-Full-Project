from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

df = pd.read_csv(
    "outputs/movement/movement_pose_features.csv"
)

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
    ).columns
    if c not in exclude
]

results = []

for threshold in [0.3, 0.4, 0.5]:

    train = df[
        (df["split"] == "train") &
        (df["pose_detection_rate"] >= threshold)
    ].copy()

    val = df[
        (df["split"] == "validation") &
        (df["pose_detection_rate"] >= threshold)
    ].copy()

    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]

    for kernel in ["linear", "rbf"]:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=kernel,
                C=1.0,
                class_weight="balanced",
                probability=True,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        results.append({
            "model": f"SVM_{kernel}",
            "pose_threshold": threshold,
            "train_samples": len(train),
            "validation_samples": len(val),
            "accuracy": accuracy_score(y_val, pred),
            "precision": precision_score(
                y_val, pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, pred, zero_division=0
            ),
            "f1": f1_score(
                y_val, pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_val, prob)
        })

result = pd.DataFrame(results)

out = Path(
    "review_results/movement/svm_comparison.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)

print("\n=== SVM VALIDATION RESULTS ===")
print(result.to_string(index=False))

print("\n=== BEST BY F1 ===")
print(
    result.sort_values("f1", ascending=False)
    .to_string(index=False)
)

print("\nSaved:", out)
