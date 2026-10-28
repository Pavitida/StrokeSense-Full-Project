"""Bounded decoder validation for local, untrusted media uploads."""
from pathlib import Path
import warnings
import cv2
import soundfile as sf
from PIL import Image

LIMITS = {"image": 5 * 1024**2, "audio": 10 * 1024**2, "video": 20 * 1024**2}
EXTENSIONS = {"image": {".jpg", ".jpeg", ".png"}, "audio": {".wav"}, "video": {".mp4", ".webm", ".mov"}}

class InvalidMedia(ValueError):
    pass

def validate_media(kind, filename, contents, folder):
    suffix = Path(filename).suffix.lower()
    if kind not in LIMITS or suffix not in EXTENSIONS[kind]:
        raise InvalidMedia("ชนิดไฟล์ไม่รองรับ")
    if not contents or len(contents) > LIMITS[kind]:
        raise InvalidMedia(f"ไฟล์ว่างหรือใหญ่เกิน {LIMITS[kind] // 1024**2} MB")
    # Never use a client filename as a filesystem path.
    path = Path(folder) / (kind + suffix)
    path.write_bytes(contents)
    path.chmod(0o600)
    try:
        if kind == "image":
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    if image.format not in {"JPEG", "PNG"} or max(image.size) > 4096 or min(image.size) < 128:
                        raise InvalidMedia("รูปต้องเป็น JPEG/PNG ขนาดด้านละ 128–4096 พิกเซล")
                    image.verify()
        elif kind == "audio":
            info = sf.info(path)
            if info.format not in {"WAV", "WAVEX"} or not 1 <= info.duration <= 30 or info.channels > 2 or not 8000 <= info.samplerate <= 96000:
                raise InvalidMedia("เสียงต้องเป็น WAV ความยาว 1–30 วินาที ไม่เกิน 2 ช่องเสียง")
        else:
            # Reject renamed executables and containers outside the accepted formats.
            magic = contents[:16]
            if not (magic.startswith(b"\x1aE\xdf\xa3") or magic[4:8] == b"ftyp"):
                raise InvalidMedia("เนื้อหาไฟล์ไม่ใช่วิดีโอ MP4/MOV/WebM ที่รองรับ")
            capture = cv2.VideoCapture(str(path))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if not capture.isOpened() or not 1 <= fps <= 60 or not 128 <= width <= 1920 or not 128 <= height <= 1920 or width * height > 1920 * 1080:
                    raise InvalidMedia("วิดีโอต้องไม่เกิน 1080p และ 60 FPS")
                count = 0
                while True:
                    ok, _ = capture.read()
                    if not ok:
                        break
                    count += 1
                    if count > fps * 30 + 1:
                        raise InvalidMedia("วิดีโอต้องยาวไม่เกิน 30 วินาที")
                if count / fps < 1:
                    raise InvalidMedia("วิดีโออ่านไม่ได้หรือสั้นกว่า 1 วินาที")
            finally:
                capture.release()
    except InvalidMedia:
        raise
    except Exception as error:
        raise InvalidMedia("ไฟล์เสียหรือไม่สามารถอ่านเนื้อหาได้ กรุณาบันทึกใหม่") from error
    return path
