"""Authenticated, loopback-only multipart API; no patient files or results retained."""
import asyncio
import hmac
import os
import tempfile
import threading
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from src.media_validation import LIMITS, InvalidMedia, validate_media
from src.fusion.run_pipeline import run_pipeline

MAX_BODY = 36 * 1024**2
_busy = threading.Lock()

def authorized(request):
    token = os.environ.get("STROKESENSE_API_TOKEN", "")
    return bool(token) and hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token)

async def health(request):
    return JSONResponse({"status": "ready"}, headers={"Cache-Control": "no-store"})

def process(files):
    # TemporaryDirectory is private (0700); each file is 0600. All exits clean up.
    with tempfile.TemporaryDirectory(prefix="strokesense-") as folder:
        paths = {kind: validate_media(kind, filename, content, folder) for kind, (filename, content) in files.items()}
        return run_pipeline(paths.get("image"), paths.get("audio"), paths.get("video"))

async def analyze(request):
    if not authorized(request):
        return JSONResponse({"status": "error", "reason": "Unauthorized"}, status_code=401)
    if not _busy.acquire(blocking=False):
        return JSONResponse({"status": "busy", "reason": "กำลังประมวลผล กรุณารอแล้วลองใหม่"}, status_code=409)
    try:
        # Bound even chunked bodies before multipart decoding or spooling.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY:
                return JSONResponse({"status": "error", "reason": "ไฟล์รวมใหญ่เกินกำหนด"}, status_code=413)
            body.extend(chunk)
        request._body = bytes(body)
        files = {}
        async with request.form(max_files=3, max_fields=1, max_part_size=MAX_BODY) as form:
            if form.get("consent") != "true":
                return JSONResponse({"status": "error", "reason": "ต้องยินยอมประมวลผลก่อน"}, status_code=400)
            seen = set()
            for key, value in form.multi_items():
                if key in seen or key not in {"consent", *LIMITS}:
                    raise InvalidMedia("ช่องข้อมูลซ้ำหรือไม่รองรับ")
                seen.add(key)
                if key == "consent":
                    continue
                if not isinstance(value, UploadFile):
                    raise InvalidMedia("ต้องส่งไฟล์ในรูปแบบ multipart")
                contents = await value.read(LIMITS[key] + 1)
                if len(contents) > LIMITS[key]:
                    return JSONResponse({"status": "error", "reason": f"{key}: ไฟล์ใหญ่เกินกำหนด"}, status_code=413)
                files[key] = (value.filename or "", contents)
            target = request.path_params.get("modality")
            if target and (target not in {"face", "speech", "movement"} or set(files) != {{"face":"image", "speech":"audio", "movement":"video"}.get(target)}):
                raise InvalidMedia("ไฟล์ไม่ตรงกับโมดูลที่เลือก")
            if not files:
                raise InvalidMedia("กรุณาเพิ่มไฟล์อย่างน้อยหนึ่งรายการ")
            # Cancellation must not release the processing lock while the worker still runs.
            worker = asyncio.create_task(asyncio.to_thread(process, files))
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InvalidMedia as error:
        return JSONResponse({"status": "error", "reason": str(error)}, status_code=422)
    except Exception:
        return JSONResponse({"status": "error", "reason": "ประมวลผลไม่สำเร็จ กรุณาตรวจไฟล์แล้วลองใหม่"}, status_code=400)
    finally:
        _busy.release()

app = Starlette(routes=[Route("/health", health), Route("/api/analyze", analyze, methods=["POST"]), Route("/api/analyze/{modality}", analyze, methods=["POST"])])
