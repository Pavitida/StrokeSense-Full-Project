"""Evaluate model inference on available labeled data without retraining or fusion."""
import argparse
from pathlib import Path
import os
import json
import csv
import tempfile
import time
import hashlib

parser=argparse.ArgumentParser()
parser.add_argument('--module',choices=['face','speech','movement'],required=True)
parser.add_argument('--device',choices=['cpu','gpu'],default='gpu')
args=parser.parse_args()
os.environ['STROKESENSE_DEVICE']=args.device
os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME']='NVIDIA'
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Data'
OUT=ROOT/'outputs/model_evaluation'/f'{args.module}_{args.device}'
OUT.mkdir(parents=True,exist_ok=True)
import cv2
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score

if args.module=='face':
    from src.face.face_inference import predict_face as predict
    labels=('unaffected','affected');model_name='face_landmark_model_v2.pkl'
elif args.module=='speech':
    from src.speech.speech_inference import predict_speech as predict
    labels=('healthy','dysarthria');model_name='speech_model.pkl'
else:
    from src.movement.movement_inference import predict_movement as predict
    labels=('Complete','Incomplete');model_name='movement_model_final.pkl'

rows=[]
started=time.monotonic()
def evaluate(path,label,source,**metadata):
    row=dict(source=source,expected=label,**metadata)
    before=time.monotonic()
    try:
        result=predict(path)
        row.update(result)
        if result.get('status')=='success':
            score=result['score']
            row['predicted']=labels[int(score>=0.5)]
            row['correct']=row['predicted']==label
    except Exception as exc:
        row.update(status='error',reason=f'{type(exc).__name__}: {exc}')
    row['seconds']=round(time.monotonic()-before,3)
    rows.append(row)
    with (OUT/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if len(rows)%10==0 or len(rows)==1:
        print(f'{args.module}/{args.device}: {len(rows)} evaluated, success={sum(r.get("status")=="success" for r in rows)}, elapsed={time.monotonic()-started:.1f}s',flush=True)

# This is a fresh evaluation, not a resume mixing previous model versions.
(OUT/'predictions.jsonl').write_text('')
with tempfile.TemporaryDirectory(prefix='strokesense-eval-') as folder:
    tmp=Path(folder)
    if args.module=='face':
        for label in labels:
            for video in sorted((DATA/'face'/label).glob('*.mp4'),key=lambda p:int(p.stem)):
                capture=cv2.VideoCapture(str(video))
                capture.set(cv2.CAP_PROP_POS_MSEC,1000)
                ok,frame=capture.read();capture.release()
                if not ok:
                    rows.append(dict(source=str(video.relative_to(ROOT)),expected=label,status='decode_error'))
                    continue
                path=tmp/'face.jpg';cv2.imwrite(str(path),frame)
                evaluate(path,label,str(video.relative_to(ROOT)),frame_seconds=1,split='unverified; potential training overlap')
    elif args.module=='movement':
        files=sorted((DATA/'movement').rglob('Test/*.mp4'))
        for video in files:
            label=video.parent.parent.name
            if label not in labels:raise ValueError(f'Unexpected label: {label}')
            evaluate(video,label,str(video.relative_to(ROOT)),exercise=video.parent.parent.parent.name,split='dataset Test folder')
    else:
        held_out={'FC01','MC04','M05'}
        for shard in sorted((DATA/'speech').rglob('*.parquet')):
            index=0
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
                for item in batch.to_pylist():
                    row_index=index;index+=1
                    name=Path(item['audio']['path']).name
                    speaker=name.split('_')[0]
                    if speaker not in held_out:continue
                    path=tmp/'speech.wav';path.write_bytes(item['audio']['bytes'])
                    evaluate(path,item['speech_status'],f'{shard.relative_to(ROOT)} row {row_index} ({name})',speaker=speaker,duration=item['duration'],split='test speaker per repository split')

valid=[r for r in rows if r.get('status')=='success' and 'predicted' in r]
y=[int(r['expected']==labels[1]) for r in valid]
pred=[int(r['predicted']==labels[1]) for r in valid]
summary={'module':args.module,'device':args.device if args.module!='speech' else 'cpu','threshold':0.5,'selected':len(rows),'evaluated':len(valid),'failed':len(rows)-len(valid),'label_order':labels,'metric_scope':'sample accuracy; split unverified' if args.module=='face' else 'all available repository-defined test samples','elapsed_seconds':round(time.monotonic()-started,2),'model_sha256':hashlib.sha256((ROOT/'models'/model_name).read_bytes()).hexdigest()}
if valid:
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    summary.update(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),positive_precision=precision,positive_recall=recall,positive_f1=f1,confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),roc_auc=roc_auc_score(y,[r['score'] for r in valid]) if len(set(y))==2 else None)
fields=list(dict.fromkeys(k for r in rows for k in r))
with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('DONE '+json.dumps(summary),flush=True)
