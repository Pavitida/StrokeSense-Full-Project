"""Browser video capture; no server-side camera access is required."""
import base64
import binascii
from pathlib import Path
import streamlit as st

MAX_VIDEO_BYTES = 20 * 1024 * 1024


def decode_video(payload):
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("ข้อมูลวิดีโอไม่ถูกต้อง กรุณาอัดใหม่")
    mime = str(payload.get("mime", "")).split(";")[0]
    if mime not in {"video/webm", "video/mp4"}:
        raise ValueError("รูปแบบวิดีโอไม่รองรับ กรุณาใช้ Chrome หรือ Edge")
    encoded = payload.get("base64", "")
    if not isinstance(encoded, str) or len(encoded) > (MAX_VIDEO_BYTES + 2) // 3 * 4:
        raise ValueError("วิดีโอใหญ่เกิน 20 MB กรุณาอัดให้สั้นลง")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("รับวิดีโอไม่ครบ กรุณาอัดใหม่") from error
    if not raw or len(raw) > MAX_VIDEO_BYTES:
        raise ValueError("วิดีโอว่างหรือใหญ่เกิน 20 MB กรุณาอัดใหม่")
    return raw, ".webm" if mime == "video/webm" else ".mp4"


def recorded_video():
    component = st.components.v2.component(
        "strokesense_video_recorder",
        html='''<div><video autoplay muted playsinline style="width:100%;max-height:320px;background:#161616;border-radius:8px"></video>
        <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
        <button id="start">เริ่มอัดวิดีโอ</button><button id="stop" disabled>หยุดบันทึก</button>
        <button id="clear">ลบวิดีโอ</button></div>
        <p id="status" role="status" aria-live="polite">กดเริ่มเพื่อเปิดกล้องและบันทึก</p></div>''',
        css="button {padding:10px 14px;border:1px solid #aaa;border-radius:8px;cursor:pointer} button:disabled {opacity:.5;cursor:default}",
        js=Path(__file__).with_name("video_recorder.js").read_text(encoding="utf-8"),
    )
    result = component(key="movement_recorder", default={"clip": None}, on_clip_change=lambda: None)
    return decode_video(result.clip)
