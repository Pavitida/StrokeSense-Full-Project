"""Start both local services with an ephemeral API token; stop children on exit."""
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for port in (8501, 8502):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Port {port} is in use. Stop the existing StrokeSense server first.")
os.environ["STROKESENSE_API_TOKEN"] = secrets.token_urlsafe(32)
os.environ["STROKESENSE_API_URL"] = "http://127.0.0.1:8502"
children = []

def stop(*_):
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
try:
    children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8502", "--no-access-log"]))
    for _ in range(120):
        if children[0].poll() is not None:
            raise RuntimeError("API failed to start")
        try:
            urllib.request.urlopen("http://127.0.0.1:8502/health", timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise RuntimeError("API startup timed out")
    children.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.maxUploadSize", "20", "--server.headless", "true", "--browser.gatherUsageStats", "false"]))
    while all(child.poll() is None for child in children):
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    stop()
