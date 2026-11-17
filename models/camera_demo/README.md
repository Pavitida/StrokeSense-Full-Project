# Frozen camera demo models

These are copies of our selected trained artifacts; the older repository models
are unchanged. SHA-256 digests are recorded in
`experiments/fusion/fusion_v2_sources.json` and checked during inference.

| File | Model | Features | Training artifact |
| --- | --- | --- | --- |
| face_LinearSVM.pkl | Linear SVM | 29 landmarks | palsynet_landmark_gpu/20261112T190038952644Z/LinearSVM.pkl |
| speech_LogisticRegression.pkl | Logistic Regression | 26 MFCC statistics | combined_gpu/speech/20261116T122048959397Z/combined_LogisticRegression.pkl |
| movement_LogisticRegression.pkl | Logistic Regression | 45 pose statistics | combined_gpu/movement/20261116T135802988738Z/combined_LogisticRegression.pkl |
| fusion_synthetic_demo.pkl | Logistic Regression, synthetic only | 3 scores | own_fusion_v2_gpu/synthetic_demo/20261117T082020058948Z/fusion_lr.pkl |

Training artifact paths are relative to the original workstation's `outputs/`.
Model bytes have not been changed or retrained for publication. The pickles include
training normalization parameters and feature order, and require the matching
cuML/CUDA runtime documented in the camera demo README.

Face labels are unaffected/affected facial asymmetry. Speech labels are
healthy/dysarthria; combined training uses TORGO and EasyCall. Movement labels
are complete/correct versus incomplete/incorrect exercise performance, combining
original data with IntelliRehabDS. These are different prediction targets and
must not be reinterpreted as confirmed Stroke/Non-stroke labels.

Fusion was trained only on generated features and artificial labels (30 training,
10 validation synthetic participants). Its 60% synthetic validation accuracy is
a software smoke result and has no clinical meaning. The web UI explicitly labels
it as a demonstration, not a probability of stroke.

No datasets, participant records or camera recordings are included in this bundle.
Dataset access and reuse remain subject to their respective providers' terms;
publication of these artifacts does not relicense underlying datasets.
