# StrokeSense Fusion Backend

## Overview

The backend connects three experimental modules:

- Face: facial abnormality (`affected`)
- Speech: dysarthria (`dysarthria`)
- Movement: rehabilitation movement incompleteness (`Incomplete`)

These scores are not validated stroke probabilities.

## Web integration

Import:

    from src.fusion.run_pipeline import run_pipeline

Call:

    result = run_pipeline(image_path, audio_path, video_path)

The function returns a dictionary containing:

- `status`
- `face`
- `speech`
- `movement`
- `fusion`

If a modality fails, the result includes `failed_modality`
and `reason`.

If the fusion model has not been trained, the overall status
is `incomplete` and no fusion score is returned.

## Training

Required CSV columns:

    participant_id,face_score,speech_score,movement_score,label

Each row must contain scores from the same participant,
with a reliable binary outcome label.

Train using:

    python3 -m src.fusion.train_fusion \
      --data data/fusion/paired_fusion_data.csv \
      --output models/fusion_model.pkl

Do not combine unrelated datasets or invent labels.

## Testing

    python3 -m unittest discover \
      -s tests -p 'test_fusion*.py' -v

The current tests use mocked modality outputs.
They do not validate real image, audio, or video inference.

## Limitations

- Fusion training requires paired, labeled multimodal data.
- The movement model predicts rehabilitation task
  completeness, not stroke-related arm weakness.
- The movement model was saved with a different
  scikit-learn version from the currently used environment.
- No clinical validation has been performed.
- This is a research prototype, not a diagnostic tool.
