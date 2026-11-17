"""Local camera inference worker. No fitting, no reference-label generation."""
import argparse,json,os,pickle,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/combined'))
CONFIG=ROOT/'experiments/fusion/fusion_v2_sources.json'

def extract(folder):
    import combined_training as common
    import imageio_ffmpeg
    request=json.loads((folder/'request.json').read_text());name=request['modality'];path=folder/'input.bin'
    details={}
    try:
        if name=='face':
            import cv2,mediapipe as mp
            from src.face.face_features import extract_aligned_features
            bgr=cv2.imread(str(path))
            if bgr is None:raise ValueError('อ่านภาพไม่ได้ กรุณาถ่ายใหม่')
            opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ROOT/'models/mediapipe/face_landmarker.task'),delegate=mp.tasks.BaseOptions.Delegate.GPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=2,min_face_detection_confidence=.5,min_face_presence_confidence=.5)
            with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as detector:
                detected=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
            if len(detected.face_landmarks)!=1:raise ValueError('ต้องเห็นใบหน้าชัดเจนเพียงหนึ่งคน กรุณาถ่ายใหม่')
            features=extract_aligned_features(detected.face_landmarks[0]);details={'faces':1,'unit':'single_image'}
            if features is None:raise ValueError('สกัดจุดใบหน้าไม่ได้ กรุณาถ่ายใหม่')
        else:
            target=folder/('decoded.wav' if name=='speech' else 'decoded.mp4')
            args=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-t','20']
            if name=='speech':args+=['-vn','-ac','1','-ar','16000','-c:a','pcm_s16le']
            else:args+=['-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart']
            subprocess.run(args+[str(target)],check=True,capture_output=True,timeout=90)
            if name=='speech':features,details=common.speech_features(target)
            else:features,details=common.video_features(target,'GPU')
        features={k:float(v) for k,v in features.items()}
        if not np.isfinite(list(features.values())).all():raise ValueError('ข้อมูลไม่ครบสำหรับการวิเคราะห์ กรุณาบันทึกใหม่')
        result={'status':'ok','modality':name,'features':features,'quality':details}
    except (ValueError,subprocess.CalledProcessError) as e:
        reason=str(e)
        if 'low_pose_coverage' in reason:reason='ตรวจพบท่าทางไม่เพียงพอ ให้เห็นไหล่ ศอก และข้อมือทั้งสองข้าง แล้วอัดใหม่'
        if isinstance(e,subprocess.CalledProcessError):reason='แปลงเสียงหรือวิดีโอไม่สำเร็จ กรุณาบันทึกใหม่'
        result={'status':'unavailable','modality':name,'message':reason}
    (folder/'features.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')

def predict(folder):
    import cupy as cp
    import hashlib
    data=json.loads((folder/'features.json').read_text());request=json.loads((folder/'request.json').read_text())
    data.update(assessment_id=request['assessment_id'],modality=request['modality'],reference_label=None)
    if data['status']=='ok':
        config=json.loads(CONFIG.read_text());name=data['modality'];source=config['sources'][name];path=ROOT/source['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise RuntimeError('Frozen model checksum changed')
        with path.open('rb') as f:pkg=pickle.load(f)
        cols=pkg['columns' if name=='face' else 'feature_columns']
        x=np.array([[data['features'][c] for c in cols]],dtype='float32')
        scale=pkg['scale' if name=='face' else 'std'];z=(x-pkg['mean'])/scale
        if not np.isfinite(z).all():raise ValueError('Invalid normalized features')
        device=cp.asarray(z,dtype=cp.float32)
        if name=='face':
            score=float(cp.asnumpy(pkg['model'].decision_function(device)).reshape(-1)[0]);label=int(score>=0)
            labels=['unaffected','affected'];score_type='svm_margin';threshold=0
        else:
            classes=cp.asnumpy(cp.asarray(pkg['model'].classes_)).reshape(-1)
            index=int(np.flatnonzero(classes==1)[0]);score=float(cp.asnumpy(pkg['model'].predict_proba(device))[0,index]);label=int(score>=.5)
            labels=['healthy','dysarthria'] if name=='speech' else ['correct / complete','incorrect / incomplete'];score_type='model_probability';threshold=.5
        data.update(score=score,score_type=score_type,prediction=label,label=labels[label],threshold=threshold,
                    feature_count=len(cols),model=source['model'],source_sha256=source['sha256'],
                    max_abs_standardized_feature=float(np.max(np.abs(z))))
        del data['features']
    (folder/'result.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')

def fuse(folder):
    import cupy as cp
    request=json.loads((folder/'fusion_request.json').read_text());config=json.loads(CONFIG.read_text())
    scores=[]
    for name in ['face','speech','movement']:
        r=request['results'][name]
        if r['status']!='ok' or r['assessment_id']!=request['assessment_id'] or r['source_sha256']!=config['sources'][name]['sha256']:
            raise ValueError('Unmatched assessment or model')
        scores.append(r['score'])
    import hashlib
    artifact=config['synthetic_fusion'];path=ROOT/artifact['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
        raise ValueError('Synthetic model checksum changed')
    with path.open('rb') as f:pkg=pickle.load(f)
    if pkg.get('synthetic_demo') is not True:raise ValueError('Unexpected demo model provenance')
    # Packaging changes paths, never the frozen source model identities.
    original=pkg['source_models']
    if any(original.get(k)!=config.get(k) for k in ['version','fusion_model']):
        raise ValueError('Unexpected demo configuration')
    for name in ['face','speech','movement']:
        if any(original['sources'][name][k]!=config['sources'][name][k] for k in ['model','sha256']):
            raise ValueError('Unexpected demo source model')
    x=(np.asarray([scores],dtype='float32')-pkg['mean'])/pkg['std']
    probability=float(cp.asnumpy(pkg['model'].predict_proba(cp.asarray(x)))[0,1])
    (folder/'fusion_result.json').write_text(json.dumps({'status':'ok','synthetic_demo':True,'demo_score':probability,
        'demo_class':int(probability>=.5),'assessment_id':request['assessment_id'],
        'message':'คะแนนชั้นจำลองจากโมเดลที่เทรนด้วยข้อมูลสังเคราะห์ ไม่ใช่โอกาสเป็น Stroke'},ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','predict','fuse']);p.add_argument('folder',type=Path);a=p.parse_args()
    os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
    {'extract':extract,'predict':predict,'fuse':fuse}[a.action](a.folder)
