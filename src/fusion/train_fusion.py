"""Train experimental Logistic Regression fusion on paired data."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ["face_score", "speech_score", "movement_score"]


def train(csv_path, output_path):
    df = pd.read_csv(csv_path)

    required = ["participant_id", *FEATURES, "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    groups = df["participant_id"]

    if not np.isfinite(X).all() or ((X < 0) | (X > 1)).any():
        raise ValueError("Scores must be between 0 and 1.")

    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Labels must contain both classes: 0 and 1.")

    if df.groupby("participant_id")["label"].nunique().max() > 1:
        raise ValueError("Conflicting labels for a participant.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    if len(np.unique(y[train_idx])) != 2:
        raise ValueError("Training split must contain both classes.")

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    model.fit(X[train_idx], y[train_idx])

    predictions = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx])[:, 1]

    print(classification_report(
        y[test_idx], predictions, labels=[0, 1], zero_division=0
    ))

    if len(np.unique(y[test_idx])) == 2:
        print("ROC-AUC:", roc_auc_score(y[test_idx], probabilities))
    else:
        print("ROC-AUC: unavailable (test split has one class)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print("Model saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output", default="models/fusion_model.pkl"
    )
    args = parser.parse_args()
    train(args.data, args.output)
