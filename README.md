# StrokeSense

Local research prototype for facial asymmetry, speech characteristics and rehabilitation movement completeness. **Not a medical diagnosis or a validated stroke probability.** Sudden facial weakness, arm weakness or speech changes require urgent care; in Thailand call **1669 without waiting for AI**. [Department of Disease Control](https://www.ddc.moph.go.th/odpc4/news.php?deptcode=odpc4&news=61128).

## macOS installation

See [RUN_MACOS.md](RUN_MACOS.md) for an Apple Silicon (M-series, native arm64) quick start, frontend-only setup in VS Code, connecting an existing backend, or running the full app with Python 3.11 and CPU inference. macOS installation and hardware capture remain unverified.

## Ubuntu / WSL installation

Python 3.11 is required. Clone the repository and run from its root:

```bash
sudo apt-get update
sudo apt-get install -y libgles2 libgl1 libglib2.0-0
python3.11 -m venv .venv-ubuntu
.venv-ubuntu/bin/python -m pip install -r requirements-ubuntu.txt
bash run-ubuntu.sh
```

For the prepared workstation the project is `C:\StrokeSense-Project`, exposed in Ubuntu as `/mnt/c/StrokeSense-Project`:

```bash
cd /mnt/c/StrokeSense-Project
bash run-ubuntu.sh
```

Open **http://localhost:8501**. The launcher starts a private API on `127.0.0.1:8502` and Streamlit on `127.0.0.1:8501`. It creates a random API token in child-process environments; do not publish it. Stop both services with **Ctrl+C**. If a port is in use, stop the previous server first. Do not run `streamlit run app.py` alone: the guided UI needs the authenticated API.

`run-ubuntu.sh` selects NVIDIA MediaPipe GPU via WSLg/Mesa. It does not convert scikit-learn or librosa to CUDA. For a CPU comparison:

```bash
STROKESENSE_DEVICE=cpu bash run-ubuntu.sh
```

Windows (optional): create `.venv` with Python 3.11, install `requirements-windows.txt`, run `run-app.cmd`. Windows hardware capture and inference have not been revalidated for this release.

## User flow

Home → instructions and processing consent → Face → Speech → Movement → Result. Select an upload or the browser camera/microphone, preview it, then choose **ใช้ข้อมูลนี้และไปต่อ**. Navigating without that button does not select the new recording. Result supports one, two or all three inputs; missing/failed modules show `incomplete`. Retake and cancel are available. The interface uses responsive cards and a collapsible sidebar.

- Face: JPEG/PNG ≤5 MB; each dimension 128–4096 pixels. Front-facing photo, natural smile.
- Speech: WAV ≤10 MB; 1–30 seconds, 1–2 channels, 8–96 kHz. Native recording is 16 kHz. The displayed Thai phrase is a recording prompt; Thai clinical accuracy and spoken-word correctness are not established.
- Movement: MP4/MOV/WebM ≤20 MB; 1–30 seconds, ≤1080p and ≤60 FPS. Keep both arms in view. The arm-raise demo is not equivalent to the rehabilitation tasks used for training.

Allow browser permissions only when you want to capture. If denied, use site settings to restore permissions or upload files instead. Camera/microphone access needs localhost or HTTPS. In-app browsers may not support capture. Chrome/Edge are suggested for local use; **Safari and real mobile hardware remain unverified**.

## Results and model limitations

The display separates the three modules and shows `success`, `incomplete` or `model_not_trained`. A score is a model class score, not accuracy and not a stroke percentage. Low scores do not establish health. Fusion is shown only when all modules succeed and its model is present. `models/fusion_model.pkl` is not included: no substitute fusion scores or fabricated labels are produced.

Saved-data accuracy reports are available from the evaluation scripts; the Data results panel has been removed from the web UI. Current sample checks are small and not clinical validation. Face split membership is unverified. Movement CPU/GPU outputs differ materially and accuracy on the selected samples was poor. **The model remains unsuitable for health decisions; this release improves workflow, not validated medical accuracy.**

## API

The API requires `Authorization: Bearer <STROKESENSE_API_TOKEN>` and multipart field `consent=true`.

| Method/path | Files | Result |
|---|---|---|
| GET `/health` | none | service readiness |
| POST `/api/analyze` | any of `image`, `audio`, `video` | calls `run_pipeline(image_path, audio_path, video_path)` |
| POST `/api/analyze/face` | `image` | partial pipeline result |
| POST `/api/analyze/speech` | `audio` | partial pipeline result |
| POST `/api/analyze/movement` | `video` | partial pipeline result |

JSON preserves per-module results even when another module fails. Status codes: 401 unauthorized, 400 missing consent/malformed request, 409 processing busy, 413 too large, 422 invalid media. No automatic retries. An in-process lock permits one inference request at a time; run **one API worker**. Upload length is bounded before multipart parsing, filenames are replaced, and actual decoders validate content. No upload executable is run and model pickles are never accepted from users.

## Privacy and retention

See [PRIVACY.md](PRIVACY.md). Research collection is disabled and separate from processing consent. Do not add real participant media, consent forms, identifiers, API tokens or output reports to Git. `/Data`, `/data`, `/outputs`, environments and secrets are ignored.

## Tests

```bash
.venv-ubuntu/bin/python -m pip install -r requirements-dev.txt
.venv-ubuntu/bin/python -m unittest discover -s tests -p 'test_*.py' -v
node tests/test_video_recorder.mjs
```

Tests cover consent gates, navigation/reset, partial results, missing fusion, input decoding/limits, auth, duplicate submission and cleanup. `test_api` includes real models on generated negative inputs. Other tests deliberately use mocks to induce errors; they do not establish real-input accuracy. Legacy full-pipeline tests skip when their private fixtures/fusion model are absent.

With local datasets present:

```bash
PYTHONPATH=. .venv-ubuntu/bin/python -u tests/run_data_samples.py
.venv-ubuntu/bin/python -m src.accuracy_summary
```

The sample runner selects 16 examples, not the entire test set. Reports are in `outputs/data_sample_test`. Do not interpret its metrics as clinical performance.

## Team deployment

This release intentionally binds to loopback. Before network use, provision a real domain/TLS certificate, authenticated access and an approved data policy. Keep the API private behind the frontend. A reverse proxy must support WebSockets, body limits and appropriate timeouts. Never expose `8502`, disable CORS/XSRF protection, or publish the bearer token to the browser. No public deployment has been performed. Use [DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) for completed and pending verification.
