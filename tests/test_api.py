import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from PIL import Image
from starlette.testclient import TestClient
from src.api import app, _busy
from src.media_validation import validate_media, InvalidMedia, LIMITS

def photo():
    buf = io.BytesIO()
    Image.new("RGB", (256,256), "white").save(buf,format="JPEG")
    return buf.getvalue()

def wave(seconds=2):
    buf=io.BytesIO()
    sf.write(buf,np.zeros(16000*seconds),16000,format="WAV")
    return buf.getvalue()

class TestApi(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{"STROKESENSE_API_TOKEN":"test-token"})
        self.env.start();self.addCleanup(self.env.stop)
        self.client=TestClient(app)
        self.auth={"Authorization":"Bearer test-token"}

    def send(self,files,**kwargs):
        return self.client.post("/api/analyze",headers=self.auth,data={"consent":"true"},files=files,**kwargs)

    def test_auth_required(self):
        self.assertEqual(self.client.post("/api/analyze").status_code,401)

    def test_consent_required(self):
        r=self.client.post("/api/analyze",headers=self.auth,files={"image":("x.jpg",photo())})
        self.assertEqual(r.status_code,400)

    def test_bad_extension_and_corrupt_file(self):
        for name,contents in [("x.exe",photo()),("x.jpg",b"corrupt")]:
            self.assertEqual(self.send({"image":(name,contents)}).status_code,422)

    def test_size_limit(self):
        self.assertEqual(self.send({"image":("x.jpg",b"x"*(LIMITS['image']+1))}).status_code,413)

    def test_total_body_limit(self):
        with patch("src.api.MAX_BODY",10):
            self.assertEqual(self.send({"image":("x.jpg",photo())}).status_code,413)

    def test_duplicate_field(self):
        self.assertEqual(self.send([("image",("1.jpg",photo())),("image",("2.jpg",photo()))]).status_code,422)

    def test_busy_prevents_duplicate_work(self):
        _busy.acquire()
        try:
            self.assertEqual(self.send({"image":("x.jpg",photo())}).status_code,409)
        finally:
            _busy.release()

    def test_tempfiles_and_client_filename(self):
        seen=[]
        def predict(image,audio,video):
            self.assertTrue(image.exists());self.assertEqual(image.name,"image.jpg")
            self.assertEqual(image.stat().st_mode & 0o777,0o600)
            seen.append(image.parent)
            return {"status":"incomplete","face":{"status":"success","score":0.5}}
        with patch("src.api.run_pipeline",side_effect=predict):
            response=self.send({"image":("../../private.jpg",photo())})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers["cache-control"],"no-store")
        self.assertFalse(seen[0].exists())

    def test_cleanup_on_backend_exception(self):
        seen=[]
        def fail(image,*args):
            seen.append(image.parent);raise RuntimeError("private server path")
        with patch("src.api.run_pipeline",side_effect=fail):
            response=self.send({"image":("x.jpg",photo())})
        self.assertEqual(response.status_code,400)
        self.assertNotIn("private server path",response.text)
        self.assertFalse(seen[0].exists())

    def test_duration_and_renamed_video(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(InvalidMedia):validate_media('audio','x.wav',wave(31),folder)
            with self.assertRaises(InvalidMedia):validate_media('video','x.webm',b'MZ executable',folder)
            self.assertTrue(validate_media('audio','x.wav',wave(),folder).is_file())

    def test_modality_endpoint_mismatch(self):
        response=self.client.post('/api/analyze/face',headers=self.auth,data={'consent':'true'},files={'audio':('x.wav',wave())})
        self.assertEqual(response.status_code,422)

    def test_real_backend_negative_inputs(self):
        # Actual models and decoders; no mocked model responses here.
        response=self.send({'image':('blank.jpg',photo()),'audio':('silent.wav',wave())})
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['face']['status'],'no_face_detected')
        self.assertEqual(result['speech']['status'],'incomplete')
        self.assertIsNone(result['fusion'])

if __name__=='__main__':unittest.main()
