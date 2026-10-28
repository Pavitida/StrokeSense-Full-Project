"""Select the MediaPipe inference device explicitly via the environment."""
import os
import mediapipe as mp


def mediapipe_delegate():
    device = os.environ.get("STROKESENSE_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "gpu"}:
        raise ValueError("STROKESENSE_DEVICE must be cpu or gpu")
    return (mp.tasks.BaseOptions.Delegate.GPU if device == "gpu"
            else mp.tasks.BaseOptions.Delegate.CPU)
