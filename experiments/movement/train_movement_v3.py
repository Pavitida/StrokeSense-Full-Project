from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from experiments.movement.hand_feature_extractor_v2 import extract_hand_features

ROOT = Path("models/movement_v3")
ROOT.mkdir(parents=True, exist_ok=True)

POSE_PATH = Path("outputs/movement/movement_pose_features_all_491.csv")
HAND_PATH = Path("review_results/movement/hand_features_100_dev.csv")

pose = pd.read_csv(POSE_PATH)
hand_old = pd.read_csv(HAND_PATH)

META = {
    "video_path", "person_id", "exercise", "label", "split",
    "video_fps", "video_total_frames", "processed_frames",
    "detected_frames", "pose_detection_rate"
}

pose_features = [
    c for c in pose.columns
    if c not in META and pd.api.types.is_numeric_dtype(pose[c])
]

hand_features = [
    "tip_distance_start_end",
    "finger_extension_start_end",
    "finger_spread_start_end"
]

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

def validate_and_train(df, features, exercise_id, feature_type):
    if df[features].isna().all(axis=0).any():
        raise ValueError("One or more feature columns contain only NaN")

    X = df[features].copy()
    y = df["label"].astype(str).to_numpy()
    groups = df["person_id"].astype(str).to_numpy()

    predictions = np.empty(len(df), dtype=object)

    logo = LeaveOneGroupOut()

    for train_idx, val_idx in logo.split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions[val_idx] = model.predict(X.iloc[val_idx])

    labels = ["Complete", "Incomplete"]

    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=labels, zero_division=0
    )

    print(f"\n=== EXERCISE {exercise_id} | {feature_type} ===")
    print("Samples:", len(df))
    print("People:", len(np.unique(groups)))
    print("Features:", len(features))
    print("LOGO Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(float(np.mean(precision)), 4))
    print("Macro Recall:", round(float(np.mean(recall)), 4))
    print("Macro F1:", round(float(np.mean(f1)), 4))
    print("Confusion Matrix [Complete, Incomplete]:")
    print(confusion_matrix(y, predictions, labels=labels))

    model = make_model()
    model.fit(X, y)

    output = ROOT / f"exercise_{exercise_id}_{feature_type.lower()}_rf.pkl"

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    with output.open("wb") as f:
        pickle.dump({
            "model": model,
            "features": features,
            "exercise": exercise_id,
            "feature_type": feature_type,
            "labels": labels
        }, f)

    result = df[["video_path", "person_id", "label"]].copy()
    result["prediction"] = predictions
    result["correct"] = result["label"] == result["prediction"]

    result.to_csv(
        ROOT / f"exercise_{exercise_id}_logo_predictions.csv",
        index=False
    )

    return {
        "exercise": exercise_id,
        "feature_type": feature_type,
        "samples": len(df),
        "people": len(np.unique(groups)),
        "features": len(features),
        "logo_accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1))
    }

print("=== MOVEMENT V3 TRAINING ===")
print("Pose features:", len(pose_features))

if len(pose_features) != 55:
    raise ValueError("Expected exactly 55 Pose features")

# Verify outputs do not already exist before starting.
for exercise_id, kind in [(1, "pose"), (2, "pose"), (3, "pose"), (4, "hand")]:
    target = ROOT / f"exercise_{exercise_id}_{kind}_rf.pkl"
    if target.exists():
        raise FileExistsError(f"Existing model will not be overwritten: {target}")

# Re-extract Hand features using the new consistent procedure.
hand_rows = []

print("\n=== EXTRACTING HAND FEATURES ===")

for i, (_, row) in enumerate(hand_old.iterrows(), 1):
    features = extract_hand_features(row["video_path"])

    hand_rows.append({
        "video_path": row["video_path"],
        "person_id": row["person_id"],
        "exercise": row["exercise"],
        "label": row["label"],
        "split": row["split"],
        **features
    })

    if i % 10 == 0:
        print(f"Hand extraction: {i}/100", flush=True)

hand_new = pd.DataFrame(hand_rows)

if len(hand_new) != 100:
    raise ValueError("Expected 100 development Hand videos")

hand_new.to_csv(
    ROOT / "hand_features_100_dev_v3.csv",
    index=False
)

print("Hand extraction complete:", len(hand_new))

# Train all four exercise-specific models.
results = []

for exercise_id in [1, 2, 3]:
    exercise_df = pose[
        pose["exercise"].astype(str).str.startswith(f"{exercise_id}_")
        & (pose["split"] != "test")
    ].copy()

    results.append(
        validate_and_train(
            exercise_df,
            pose_features,
            exercise_id,
            "Pose"
        )
    )

results.append(
    validate_and_train(
        hand_new,
        hand_features,
        4,
        "Hand"
    )
)

summary = pd.DataFrame(results)
summary.to_csv(ROOT / "validation_summary.csv", index=False)

manifest = {
    "version": "movement_v3",
    "algorithm": "RandomForest",
    "validation": "LeaveOneGroupOut",
    "group": "person_id",
    "development_samples": int(summary["samples"].sum()),
    "reserved_test_samples": 80,
    "hand_extraction": "experiments/movement/hand_feature_extractor_v2.py",
    "status": "trained_not_tested"
}

(ROOT / "training_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("\n=== FINAL V3 VALIDATION SUMMARY ===")
print(summary.to_string(index=False))
print("\nALL FOUR MODELS TRAINED")
print("TEST SET NOT USED")
