# Run with Ubuntu 22.04 (WSL)

Project location: C:\StrokeSense-Project on Windows, /mnt/c/StrokeSense-Project on Ubuntu.
Linux Python 3.11 is installed in .python-ubuntu and its environment is .venv-ubuntu.
Keep both folders. Windows .venv cannot be used on Linux.

Run in the Ubuntu terminal:

```bash
cd /mnt/c/StrokeSense-Project
bash run-ubuntu.sh
```

Open http://localhost:8501. Press Ctrl+C to stop a foreground server.
In VS Code choose Terminal > Run Task > Run StrokeSense (Ubuntu).

Run a single model:

```bash
.venv-ubuntu/bin/python -m src.face.face_inference --image /path/to/face.jpg
.venv-ubuntu/bin/python -m src.speech.speech_inference --audio /path/to/speech.wav
.venv-ubuntu/bin/python -m src.movement.movement_inference --video /path/to/movement.mp4
```

The combined score requires models/fusion_model.pkl, which is not provided.
Real-input testing requires your own image, audio and video.
The UI supports individual modality results even when the fusion model is missing.

The background server started during setup writes logs to outputs/streamlit-ubuntu.log and its PID to outputs/streamlit-ubuntu.pid.
Before starting another instance, check http://localhost:8501.

System dependency installed in Ubuntu: libgles2 (required by MediaPipe).


## GPU

`bash run-ubuntu.sh` selects the MediaPipe GPU delegate and the NVIDIA adapter.
Verified renderer: D3D12 (NVIDIA GeForce RTX 5070 Ti), via WSLg/Mesa OpenGL ES.
Face and pose landmark extraction use this delegate. Some MediaPipe operations,
scikit-learn classification and librosa audio processing still run on CPU.
This is not a CUDA conversion of the models. GPU errors are reported, not silently retried on CPU.

For individual model commands, first run:

```bash
export STROKESENSE_DEVICE=gpu
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

To explicitly run the app on CPU: `STROKESENSE_DEVICE=cpu bash run-ubuntu.sh`.
GPU smoke testing used blank images/video to check execution, not model accuracy or speed.

## Camera and microphone inputs

Open http://localhost:8501 in a browser and choose the camera/microphone option
in each section. Grant camera or microphone permission when prompted.
Photo capture uses Streamlit camera_input; speech recording uses audio_input at 16 kHz.
Video uses the browser MediaRecorder (WebM/MP4), without audio, up to 30 seconds / 20 MB.
The camera is released after recording stops or the recorder is unmounted.
Use Chrome or Edge if the in-app browser does not allow camera/microphone access.
Files are processed locally by the Ubuntu backend. Results stay visible until cleared or the session ends.
You may analyze just one input; three successful scores are required for fusion.

Validation: three Python capture/UI tests and mock-media JavaScript lifecycle tests passed.
Live camera and microphone hardware recording still needs user verification.
API reference: https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component

## Accuracy display

Accuracy is available in terminal reports; the web Data panel has been removed.
Live camera inference scores are separate from accuracy. The sample runner prints accuracy at completion.
To summarize saved results without rerunning inference:

```bash
cd /mnt/c/StrokeSense-Project
.venv-ubuntu/bin/python -m src.accuracy_summary
```

Reports: outputs/data_sample_test/ACCURACY.md and accuracy_summary.json.
Speech/movement figures are selected test-subset accuracy; face split is unverified.
