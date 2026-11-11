from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

ROOT = Path(".")
DATA = ROOT / "outputs/face"
META = ROOT / "review_results/face"
OUT = META / "final_test"
OUT.mkdir(parents=True, exist_ok=True)

model = joblib.load(ROOT / "models/face_hog_svm_baseline.pkl")
classes = list(model.classes_)

X_val = np.load(DATA / "X_val.npy")
X_test = np.load(DATA / "X_test.npy")
val = pd.read_csv(META / "val_metadata.csv")
test = pd.read_csv(META / "test_metadata.csv")
saved = pd.read_csv(META / "validation_predictions.csv")

assert len(X_val) == len(val) == len(saved)
assert len(X_test) == len(test)
assert X_val.shape[1] == X_test.shape[1] == model.n_features_in_

# Confirm validation metadata ordering against saved predictions.
if not (
    val["file_path"].astype(str).to_numpy()
    == saved["file_path"].astype(str).to_numpy()
).all():
    raise RuntimeError(
        "Validation metadata order differs from saved predictions."
    )

# Confirm loaded model reproduces original validation predictions.
val_pred = model.predict(X_val)
agreement = np.mean(
    val_pred == saved["predicted_severity"].to_numpy()
)
val_acc = accuracy_score(val["severity"], val_pred)

print("Validation prediction agreement:", agreement)
print("Recomputed validation accuracy:", val_acc)

if agreement < 1.0 or abs(val_acc - 0.913829) > 0.00001:
    raise RuntimeError(
        "Validation reproduction failed. Stop before final test."
    )

print("\nValidation reproduction PASSED.")
print("IMPORTANT: Test feature row ordering must still match metadata.")
print("Check original feature/split scripts if not previously verified.")

answer = input(
    "Have you verified X_test row order matches test_metadata.csv? "
    "Type YES to continue: "
)
if answer != "YES":
    raise SystemExit("Stopped before final test.")

y_true = test["severity"].to_numpy()
y_pred = model.predict(X_test)
scores = model.decision_function(X_test)

print("\n========== YFP FINAL TEST ==========")
print("Images:", len(y_true))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro Precision:", precision_score(
    y_true, y_pred, average="macro", zero_division=0
))
print("Macro Recall:", recall_score(
    y_true, y_pred, average="macro", zero_division=0
))
print("Macro F1:", f1_score(
    y_true, y_pred, average="macro", zero_division=0
))
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred, labels=classes, zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=classes)
print("\nConfusion Matrix:\n", cm)

predictions = test.copy()
predictions["predicted_severity"] = y_pred
predictions["correct"] = y_true == y_pred
predictions.to_csv(
    OUT / "test_predictions.csv", index=False
)

metrics = pd.DataFrame([{
    "samples": len(y_true),
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_precision": precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "macro_recall": recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
    "macro_f1": f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
}])
metrics.to_csv(OUT / "test_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("YFP Final Test - Confusion Matrix")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "test_confusion_matrix.png", dpi=300)
plt.close()

# One-vs-Rest ROC from LinearSVC decision scores.
y_binary = label_binarize(y_true, classes=classes)

fig, ax = plt.subplots(figsize=(8, 7))
aucs = []

for i, label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_binary[:, i], scores[:, i])
    class_auc = auc(fpr, tpr)
    aucs.append(class_auc)
    ax.plot(fpr, tpr, label=f"{label} (AUC={class_auc:.3f})")

macro_auc = float(np.mean(aucs))
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"YFP Final Test - One-vs-Rest ROC (Macro AUC={macro_auc:.3f})"
)
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "test_roc.png", dpi=300)
plt.close()

print("\nMacro OvR ROC-AUC:", macro_auc)
print("\nSaved to:", OUT)
