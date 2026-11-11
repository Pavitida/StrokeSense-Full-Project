from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

MODEL_PATH = Path("models/mediapipe/hand_landmarker.task")

def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    v1 = a - b
    v2 = c - b

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator < 1e-8:
        return np.nan

    cosine = np.clip(
        np.dot(v1, v2) / denominator,
        -1.0, 1.0
    )

    return np.degrees(np.arccos(cosine))


def extract_hand_features(video_path):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = Options(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=1
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 9:
        cap.release()
        raise ValueError("Video has fewer than 9 frames")

    positions = np.linspace(
        0.1, 0.9, 9
    )

    values = []

    with HandLandmarker.create_from_options(options) as detector:
        for pos in positions:
            frame_id = int((total - 1) * pos)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(image)

            if not result.hand_landmarks:
                continue

            points = np.array([
                [p.x, p.y, p.z]
                for p in result.hand_landmarks[0]
            ])

            palm = np.linalg.norm(
                points[9] - points[0]
            )

            if palm < 1e-8:
                continue

            tips = [4, 8, 12, 16, 20]

            tip_distance = np.mean([
                np.linalg.norm(points[t] - points[0])
                / palm
                for t in tips
            ])

            joints = [
                (2, 3, 4),
                (5, 6, 7),
                (6, 7, 8),
                (9, 10, 11),
                (10, 11, 12),
                (13, 14, 15),
                (14, 15, 16),
                (17, 18, 19),
                (18, 19, 20)
            ]

            finger_extension = np.nanmean([
                angle(points[a], points[b], points[c])
                for a, b, c in joints
            ])

            finger_spread = np.mean([
                np.linalg.norm(
                    points[tips[j]] - points[tips[j + 1]]
                ) / palm
                for j in range(4)
            ])

            values.append([
                tip_distance,
                finger_extension,
                finger_spread
            ])

    cap.release()

    if len(values) < 2:
        raise ValueError(
            f"Insufficient hand detections: {video_path}"
        )

    arr = np.array(values)

    return {
        "tip_distance_start_end":
            float(arr[-1, 0] - arr[0, 0]),
        "finger_extension_start_end":
            float(arr[-1, 1] - arr[0, 1]),
        "finger_spread_start_end":
            float(arr[-1, 2] - arr[0, 2])
    }
