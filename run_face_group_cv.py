from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(".")
DATA = ROOT / "review_results/face_landmark/palsynet_split.csv"
SCRIPT = ROOT / "experiments/face/11_train_palsynet_model.py"
OUT = ROOT / "review_results/face_landmark/group_cv"
OUT.mkdir(parents=True, exist_ok=True)

# Read original feature definition without executing training code.
tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
features = None

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "FEATURES":
                features = ast.literal_eval(node.value)

if not features:
    raise RuntimeError("Cannot find a literal FEATURES list in training script.")

df = pd.read_csv(DATA)

required = features + ["person_id", "label", "frame_number"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()
df["person_id"] = df["person_id"].astype(str)
df["target"] = df["label"].map({
    "unaffected": 0,
    "affected": 1
})

if df["target"].isna().any():
    raise ValueError("Unknown class labels.")

# Ensure each person has exactly one class label.
person_labels = df.groupby("person_id")["target"].nunique()
if (person_labels > 1).any():
    raise ValueError("Some persons have multiple class labels.")

df["target"] = df["target"].astype(int)

print("Frames:", len(df))
print("People:", df["person_id"].nunique())
print("Features:", len(features))

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = []
fold_results = []

X = df[features].to_numpy(dtype=float)
y = df["target"].to_numpy()
groups = df["person_id"].to_numpy()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    assert set(train.person_id).isdisjoint(set(test.person_id))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X[train_idx], y[train_idx])

    test["affected_probability"] = (
        model.predict_proba(X[test_idx])[:, 1]
    )

    people = (
        test.groupby(["person_id", "label", "target"], as_index=False)
        .agg(
            affected_probability=("affected_probability", "mean"),
            frames=("frame_number", "count")
        )
    )

    people["prediction"] = (
        people["affected_probability"] >= 0.5
    ).astype(int)
    people["fold"] = fold

    yt = people["target"].to_numpy()
    yp = people["prediction"].to_numpy()
    ys = people["affected_probability"].to_numpy()

    auc = (
        roc_auc_score(yt, ys)
        if len(np.unique(yt)) == 2
        else np.nan
    )

    metrics = {
        "fold": fold,
        "train_people": train.person_id.nunique(),
        "test_people": len(people),
        "test_frames": len(test),
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "roc_auc": auc
    }

    fold_results.append(metrics)
    predictions.append(people)

    print(f"\nFold {fold}: {metrics}")

results = pd.DataFrame(fold_results)
oof = pd.concat(predictions, ignore_index=True)

assert len(oof) == df.person_id.nunique()
assert oof.person_id.nunique() == len(oof)

results.to_csv(OUT / "fold_metrics.csv", index=False)
oof.to_csv(OUT / "person_oof_predictions.csv", index=False)

yt = oof["target"].to_numpy()
yp = oof["prediction"].to_numpy()
ys = oof["affected_probability"].to_numpy()

cm = confusion_matrix(yt, yp, labels=[0, 1])
auc = roc_auc_score(yt, ys)
fpr, tpr, _ = roc_curve(yt, ys)

print("\n" + "=" * 60)
print("OUT-OF-FOLD PERSON-LEVEL RESULTS")
print("=" * 60)
print("People:", len(oof))
print("Accuracy:", accuracy_score(yt, yp))
print("Precision:", precision_score(yt, yp, zero_division=0))
print("Recall:", recall_score(yt, yp, zero_division=0))
print("F1:", f1_score(yt, yp, zero_division=0))
print("ROC-AUC:", auc)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2.5,
         label=f"Group CV (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray",
         label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("StrokeSense Face Landmark - Group CV ROC")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "group_cv_roc.png", dpi=300)
plt.close()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Unaffected", "Affected"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("StrokeSense Face Landmark - Group CV")
plt.tight_layout()
plt.savefig(OUT / "group_cv_confusion_matrix.png", dpi=300)
plt.close()

print("\nSaved results to:", OUT)
