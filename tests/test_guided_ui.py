import base64
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.capture import decode_video

APP=Path(__file__).resolve().parents[1]/'app.py'

class TestGuidedUI(unittest.TestCase):
    def test_navigation_consent_and_reset(self):
        app=AppTest.from_file(str(APP)).run(timeout=60)
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Face').run()
        self.assertTrue(any('ความยินยอม' in item.value for item in app.info))
        app.sidebar.radio[0].set_value('วิธีใช้และความยินยอม').run()
        consent=next(x for x in app.checkbox if not x.disabled)
        consent.check().run()
        next(b for b in app.button if b.label=='ยืนยันและไป Face').click().run()
        self.assertEqual(app.session_state['nav'],'Face')
        for page in ['Speech','Movement','Result','ช่วยเหลือ / สิทธิ์อุปกรณ์','Home']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
        app.session_state['media']={'audio':('secret.wav',b'private')}
        next(b for b in app.sidebar.button if 'ยกเลิก' in b.label).click().run()
        self.assertFalse(app.session_state['consented'])
        self.assertEqual(app.session_state['media'],{})

    def test_partial_results_and_missing_fusion(self):
        app=AppTest.from_file(str(APP)).run(timeout=60)
        app.session_state['consented']=True
        app.session_state['result']={'status':'incomplete','face':{'status':'success','score':0.4},'speech':{'status':'incomplete'},'movement':{'status':'model_not_trained'},'fusion':{'status':'model_not_trained'}}
        app.sidebar.radio[0].set_value('Result').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric),1)
        self.assertTrue(any('model_not_trained' in i.value for i in app.info))

    def test_video_payload(self):
        data=base64.b64encode(b'video').decode()
        self.assertEqual(decode_video({'mime':'video/webm','base64':data}),(b'video','.webm'))
        for obj in [{'mime':'text/plain','base64':data},{'mime':'video/mp4','base64':'!!!'}]:
            with self.assertRaises(ValueError):decode_video(obj)

if __name__=='__main__':unittest.main()
