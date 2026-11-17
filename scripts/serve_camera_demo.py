"""Local camera capture and frozen GPU inference; bound to loopback only."""
import argparse
import base64,hashlib,json,os,re,subprocess,threading,uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock()
STORE=ROOT/'outputs/camera_demo_inference'

def linux_path(path):
    p=str(path.resolve()).replace('\\','/')
    return '/mnt/'+p[0].lower()+p[2:] if re.match(r'^[A-Za-z]:/',p) else p

def worker(action,folder):
    env='.venv-train-wsl' if action=='extract' else '.venv-face-gpu'
    command=[linux_path(ROOT/env/'bin/python'),'-u',linux_path(ROOT/'scripts/camera_case_worker.py'),action,linux_path(folder)]
    if os.name=='nt':command=['wsl.exe','-d','Ubuntu-22.04','-e']+command
    result=subprocess.run(command,capture_output=True,timeout=240)
    with (folder/'worker.log').open('ab') as f:f.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('วิเคราะห์ไม่สำเร็จ กรุณาตรวจ WSL/GPU หรือบันทึกใหม่ (รายละเอียดใน worker.log)')

class Handler(SimpleHTTPRequestHandler):
    def send_json(self,value,status=200):
        payload=json.dumps(value,ensure_ascii=False).encode('utf-8');self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_POST(self):
        try:
            host=self.headers.get('Host','')
            if host not in [f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}']:
                return self.send_json({'error':'Invalid host'},403)
            origin=self.headers.get('Origin')
            if origin and origin not in ['http://'+host]:return self.send_json({'error':'Invalid origin'},403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send_json({'error':'JSON required'},415)
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=28*1024*1024:return self.send_json({'error':'ไฟล์ใหญ่เกิน 20 MB'},413)
            request=json.loads(self.rfile.read(size));case=request.get('assessment_id','')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',case):raise ValueError('Invalid assessment id')
            STORE.mkdir(parents=True,exist_ok=True)
            if self.path=='/api/analyze':
                name=request.get('modality');mime=request.get('mime','').split(';')[0]
                allowed={'face':['image/png','image/jpeg'],'speech':['audio/webm','audio/mp4','audio/ogg'],'movement':['video/webm','video/mp4']}
                if name not in allowed or mime not in allowed[name]:raise ValueError('Unsupported media format')
                raw=base64.b64decode(request['content'],validate=True)
                if not 0<len(raw)<=20*1024*1024:raise ValueError('Empty or oversized file')
                folder=STORE/uuid.uuid4().hex;folder.mkdir()
                (folder/'input.bin').write_bytes(raw)
                (folder/'request.json').write_text(json.dumps({'assessment_id':case,'modality':name,'mime':mime,
                    'collection_type':request.get('collection_type','self_test'),'reference_label':None,'sha256':hashlib.sha256(raw).hexdigest()}),encoding='utf-8')
                with LOCK:worker('extract',folder);worker('predict',folder)
                result=json.loads((folder/'result.json').read_text(encoding='utf-8'));result['result_id']=folder.name
                self.send_json(result)
            elif self.path=='/api/fuse':
                refs=request['results'];results={}
                for name in ['face','speech','movement']:
                    ref=refs[name]
                    if not re.fullmatch('[a-f0-9]{32}',ref):raise ValueError('Invalid result id')
                    r=json.loads((STORE/ref/'result.json').read_text(encoding='utf-8'))
                    if r['assessment_id']!=case or r['modality']!=name:raise ValueError('Assessment mismatch')
                    results[name]=r
                folder=STORE/uuid.uuid4().hex;folder.mkdir()
                (folder/'fusion_request.json').write_text(json.dumps({'assessment_id':case,'results':results}),encoding='utf-8')
                with LOCK:worker('fuse',folder)
                self.send_json(json.loads((folder/'fusion_result.json').read_text(encoding='utf-8')))
            else:self.send_json({'error':'Not found'},404)
        except (ValueError,KeyError,FileNotFoundError) as e:self.send_json({'error':str(e)},400)
        except subprocess.TimeoutExpired:self.send_json({'error':'การวิเคราะห์ใช้เวลานานเกินกำหนด กรุณาลองใหม่'},504)
        except Exception as e:self.send_json({'error':str(e)},500)
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Permissions-Policy','camera=(self), microphone=(self)')
        super().end_headers()
    def list_directory(self,path):
        self.send_error(404)
    def log_message(self,format,*args):
        pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    folder=Path(__file__).resolve().parents[1]/'web/camera_demo'
    if not (folder/'index.html').is_file():raise SystemExit('Missing web/camera_demo/index.html')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(folder)))
    print(f'Camera demo: http://localhost:{args.port}',flush=True)
    print('Local model inference enabled. Media saved under outputs/camera_demo_inference. Ctrl+C stops.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
