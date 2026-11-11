from pathlib import Path
from tempfile import TemporaryDirectory
from src.fusion.streamlit_backend import analyze_stroke_signs

MAX_SIZE = 100 * 1024 * 1024
EXTENSIONS = {"face": {".jpg", ".jpeg", ".png"}, "speech": {".wav"}, "movement": {".mp4"}}

def analyze_uploaded_files(face_file, speech_file, movement_file):
    uploads = {"face": face_file, "speech": speech_file, "movement": movement_file}
    for name, uploaded in uploads.items():
        if uploaded is None:
            return {"status": "error", "failed_modality": name, "message": "Missing upload", "fusion": None}
        if Path(uploaded.name).suffix.lower() not in EXTENSIONS[name]:
            return {"status": "error", "failed_modality": name, "message": "Unsupported file type", "fusion": None}
    with TemporaryDirectory(prefix="strokesense_") as directory:
        paths = {}
        for name, uploaded in uploads.items():
            data = uploaded.getvalue()
            if not data or len(data) > MAX_SIZE:
                return {"status": "error", "failed_modality": name, "message": "Invalid file size", "fusion": None}
            suffix = Path(uploaded.name).suffix.lower()
            path = Path(directory) / (name + suffix)
            path.write_bytes(data)
            paths[name] = path
        return analyze_stroke_signs(paths["face"], paths["speech"], paths["movement"])
