"""StrokeSense backend interface for Streamlit."""

from pathlib import Path
from src.fusion.run_pipeline import run_pipeline


def analyze_stroke_signs(image_path, audio_path, video_path):
    """
    Run experimental face, speech and movement analysis.

    Returns a dictionary for Streamlit.
    This is not a clinical stroke diagnosis.
    """

    paths = {
        "face": image_path,
        "speech": audio_path,
        "movement": video_path,
    }

    for name, path in paths.items():
        if not path or not Path(path).is_file():
            return {
                "status": "error",
                "failed_modality": name,
                "message": f"Missing or invalid {name} file.",
                "face": None,
                "speech": None,
                "movement": None,
                "fusion": None,
            }

    try:
        result = run_pipeline(
            str(image_path),
            str(audio_path),
            str(video_path),
        )
    except Exception as error:
        return {
            "status": "error",
            "message": f"Analysis failed: {type(error).__name__}",
            "face": None,
            "speech": None,
            "movement": None,
            "fusion": None,
        }

    return {
        "status": result.get("status", "error"),
        "face": result.get("face"),
        "speech": result.get("speech"),
        "movement": result.get("movement"),
        "fusion": result.get("fusion"),
        "failed_modality": result.get("failed_modality"),
        "message": result.get("reason"),
        "disclaimer": (
            "Experimental screening support only. "
            "Not a stroke probability or medical diagnosis."
        ),
    }
