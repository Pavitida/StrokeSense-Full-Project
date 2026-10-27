# Fusion Logistic Regression — Synthetic Proof of Concept

## Dataset
- 200 synthetic samples
- Labels generated using an artificial mean-score threshold
- No patient data or clinical stroke labels

## Training
- Algorithm: Logistic Regression
- Group-aware train/test split
- Test samples: 40

## Test results
- Accuracy: 0.95
- Macro F1-score: 0.95
- ROC-AUC: 1.00

## Inference examples
| Face | Speech | Movement | Class-1 probability |
|---|---|---|---|
| 0.20 | 0.30 | 0.20 | 0.0590 |
| 0.50 | 0.50 | 0.50 | 0.4807 |
| 0.80 | 0.70 | 0.90 | 0.9503 |

## Model
models/fusion_demo_model.pkl

## Important limitations
This experiment only verifies that the training and
inference pipeline functions correctly.

The synthetic labels are not stroke diagnoses.
The reported metrics do not measure clinical performance.

The production fusion model remains untrained.
Paired multimodal data with reliable labels is required
before meaningful stroke-screening evaluation.
