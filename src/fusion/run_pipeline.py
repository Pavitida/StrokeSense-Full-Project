"""Run independent modalities, preserving partial results when one module fails."""
import argparse
import json
import math
from src.face.face_inference import predict_face
from src.speech.speech_inference import predict_speech
from src.movement.movement_inference import predict_movement
from src.fusion.fusion_model import predict_fusion

def run_pipeline(image_path, audio_path, video_path):
    result = {"status": "incomplete", "fusion": None}
    for name, path, predict in [("face", image_path, predict_face), ("speech", audio_path, predict_speech), ("movement", video_path, predict_movement)]:
        if path is None:
            result[name] = {"status": "incomplete", "score": None, "reason": "ยังไม่มีข้อมูลสำหรับโมดูลนี้"}
            continue
        try:
            item = predict(path)
            if item.get("status") == "success":
                score = item.get("score")
                if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid model score")
            result[name] = item
        except FileNotFoundError:
            result[name] = {"status": "model_not_trained", "score": None, "reason": "ไม่พบไฟล์โมเดลหรือข้อมูลที่ต้องใช้"}
        except ValueError:
            result[name] = {"status": "incomplete", "score": None, "reason": "ข้อมูลไม่เหมาะสม อาจเป็นเสียงเงียบหรือไม่พบจุดร่างกาย กรุณาบันทึกใหม่"}
        except Exception:
            result[name] = {"status": "incomplete", "score": None, "reason": "โมเดลประมวลผลไม่สำเร็จ กรุณาลองใหม่หรือติดต่อผู้ดูแล"}
    failed = [name for name in ("face", "speech", "movement") if result[name].get("status") != "success"]
    if failed:
        result["failed_modality"] = failed[0]
        result["reason"] = result[failed[0]].get("reason", result[failed[0]]["status"])
        return result
    try:
        result["fusion"] = predict_fusion(*(result[name]["score"] for name in ("face", "speech", "movement")))
    except Exception:
        result["fusion"] = {"status": "incomplete", "score": None, "reason": "โมเดลรวมคะแนนยังไม่พร้อม"}
    if result["fusion"].get("status") == "success":
        result["status"] = "success"
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("image", "audio", "video"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.image, args.audio, args.video), indent=2))
