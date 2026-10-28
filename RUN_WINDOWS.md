# Running on Windows

Python 3.11 and dependencies are installed in `.venv`.
The local Python runtime is in `.python`; keep this folder because `.venv` uses it.

## Start the app

Double-click `run-app.cmd`, or run this in the VS Code PowerShell terminal:

```powershell
cd C:\StrokeSense-Project
.\.venv\Scripts\python.exe scripts/serve.py
```

Open http://localhost:8501. Stop with Ctrl+C in the terminal.
VS Code: Terminal > Run Task > Run StrokeSense.
No environment activation is required for these commands.

## Model inference from the terminal

```powershell
.\.venv\Scripts\python.exe -m src.face.face_inference --image "path\to\face.jpg"
.\.venv\Scripts\python.exe -m src.speech.speech_inference --audio "path\to\speech.wav"
.\.venv\Scripts\python.exe -m src.movement.movement_inference --video "path\to\movement.mp4"
```

Replace the paths with your own files. The repository does not include sample inputs.

## Current limitations

The three modality models and MediaPipe assets are included. `models/fusion_model.pkl` is missing.
The combined pipeline returns `incomplete` until a trained fusion model is provided.
The current UI displays individual scores only when the full pipeline succeeds; use the commands above to inspect individual model results.
Fusion training requires paired, labeled data from the same participants. No training data has been fabricated or training performed.

## Dependencies and verification

`requirements-windows.txt` preserves the original model-library versions and uses Streamlit 1.66.1.
The original Streamlit 1.45.1 requires pandas <3, conflicting with the project's pandas 3.0.6.
`requirements-windows-lock.txt` records all installed package versions.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_fusion*.py" -v
```

The fusion tests use mocks and do not establish real-input model accuracy.
The full `test_pipeline.py` suite needs input fixtures and the missing fusion model.
