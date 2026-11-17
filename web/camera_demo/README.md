# Camera demo with local GPU inference

Run from the repository root with Python 3.11 on Linux/WSL:

```bash
python3 scripts/serve_camera_demo.py --port 8767
```

Or on Windows (workers use the Ubuntu-22.04 WSL distribution):

```powershell
python scripts/serve_camera_demo.py --port 8767
```

Open http://localhost:8767 in Chrome/Edge. Allow camera/microphone, capture the
face, record speech, then record movement. Each recording is analyzed automatically;
results appear on the page. Once all three succeed, the synthetic Fusion demo runs.
Downloads are optional. This is a local Python service, not a GitHub Pages site.

## GPU environments

The repository root must contain two Linux Python 3.11 environments:

- `.venv-train-wsl`: numpy 2.4.6, pandas 3.1.1, librosa 0.11.0,
  soundfile 0.14.0, mediapipe 1.2.2, opencv-contrib-python 5.0.0.93,
  imageio-ffmpeg 0.6.0. Used for feature extraction and WebM decoding.
- `.venv-face-gpu`: cuml-cu12 26.10.0, cupy-cuda12x 14.2.0,
  numpy 2.4.6, scikit-learn 1.9.1. Used to load the frozen CUDA pickles.

These are the versions on the tested workstation, not a claim of compatibility
with other versions. Provision compatible NVIDIA CUDA 12 drivers, WSL GPU/WSLg
and Linux graphics libraries before creating the environments. Existing CPU/macOS
requirements for the older app do not install this GPU runtime. The environments
are not committed. Do not copy Windows venvs into WSL or reuse pickles with an
unverified cuML version. Only load the trusted artifacts included in this repository.

To check the bundled models without recording or training:

```bash
.venv-face-gpu/bin/python scripts/check_camera_models.py
.venv-train-wsl/bin/python experiments/combined/test_combined_features.py
```

## Model behavior and storage

Registry: `experiments/fusion/fusion_v2_sources.json`.
Artifacts and provenance: [models/camera_demo](../../models/camera_demo/README.md).
The older application and its existing models are separate from this demo.

Workers load lazily and run serially to avoid GPU conflicts. Processing can take
tens of seconds per modality. Errors and pending states appear on the page.
Camera/microphone open only after a button click and browser permission.
Media is sent to the local loopback server, not an external service. Inputs,
features, results and worker logs persist in `outputs/camera_demo_inference/`;
this directory is ignored by Git. The reference label remains unknown/null.

Face reports an SVM margin, not a calibrated probability. Single-image inference
is not equivalent to the multi-frame person-level benchmark. Speech was trained
on TORGO (English) and EasyCall (Italian), not validated for Thai. Movement scores
apply to the training exercise tasks, not arbitrary gestures. The Fusion model
uses artificial labels and synthetic examples: its score is NOT stroke risk.
No training, diagnosis or automatic generation of clinical labels occurs here.
