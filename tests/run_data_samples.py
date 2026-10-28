"""Deterministic sample-level smoke test; does not train or combine participants."""
import os
os.environ['STROKESENSE_DEVICE'] = 'gpu'
os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
import csv
import json
import time
from pathlib import Path
from collections import Counter
import cv2
import pyarrow.parquet as pq
from src.face.face_inference import predict_face
from src.speech.speech_inference import predict_speech
from src.movement.movement_inference import predict_movement

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data'
OUT = ROOT / 'outputs' / 'data_sample_test'
SAMPLES = OUT / 'samples'
SAMPLES.mkdir(parents=True, exist_ok=True)
rows = []

def run(modality, path, label, provenance, split):
    start = time.monotonic()
    row = dict(modality=modality, input=str(path.relative_to(ROOT)), source=provenance,
               expected=label, split=split)
    try:
        result = {'face':predict_face, 'speech':predict_speech, 'movement':predict_movement}[modality](path)
        row.update(result)
        if result.get('score') is not None:
            positive, negative = {'face':('affected','unaffected'), 'speech':('dysarthria','healthy'), 'movement':('Incomplete','Complete')}[modality]
            row['predicted_at_0_5'] = positive if result['score'] >= 0.5 else negative
            row['matches_label'] = row['predicted_at_0_5'] == label
    except Exception as exc:
        row.update(status='error', reason=f'{type(exc).__name__}: {exc}')
    row['seconds'] = round(time.monotonic() - start, 3)
    rows.append(row)
    (OUT/'results.json').write_text(json.dumps(rows, indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)

# Face inputs are videos; use one fixed frame at one second, before seeing predictions.
# No original held-out manifest was supplied for these videos.
for label in ['affected','unaffected']:
    for video in sorted((DATA/'face'/label).glob('*.mp4'),key=lambda p:int(p.stem))[:2]:
        cap=cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC,1000)
        ok, frame=cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot read frame at 1 second: {video}')
        target=SAMPLES/f'face_{label}_{video.stem}_1s.jpg'
        if not cv2.imwrite(str(target),frame): raise RuntimeError('Could not save frame')
        run('face',target,label,str(video.relative_to(ROOT))+' @ 1.0s','unknown; possible training overlap')

# Speakers declared as test speakers in experiments/speech/04_split_speakers.py.
remaining={'FC01':1,'MC04':1,'M05':2}
for shard in sorted((DATA/'speech').rglob('*.parquet')):
    metadata=pq.read_table(shard,columns=['audio.path','speech_status','duration']).to_pylist()
    chosen={}
    for idx, item in enumerate(metadata):
        speaker=Path(item['path']).name.split('_')[0]
        if remaining.get(speaker,0)>0 and float(item['duration'])>=2:
            chosen[idx]=item
            remaining[speaker]-=1
    if not chosen: continue
    offset=0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=32,columns=['audio']):
        for local,item in enumerate(batch.to_pylist()):
            idx=offset+local
            if idx not in chosen: continue
            meta=chosen[idx]
            target=SAMPLES/('speech_'+Path(meta['path']).name)
            target.write_bytes(item['audio']['bytes'])
            run('speech',target,meta['speech_status'],f'{shard.relative_to(ROOT)} row {idx} ({meta["path"]})','test speaker per repository split')
        offset+=batch.num_rows
        if offset>max(chosen): break
if any(remaining.values()): raise RuntimeError(f'Missing speech selections: {remaining}')

# One Test video for each exercise and class; never use Train here.
exercises=DATA/'movement'/'An upper limb stroke rehabilitation exercise video'/'Exercise'
for exercise in sorted(exercises.iterdir()):
    if not exercise.is_dir(): continue
    for label in ['Complete','Incomplete']:
        files=sorted((exercise/label/'Test').glob('*.mp4'))
        if not files: raise RuntimeError(f'No Test video: {exercise.name}/{label}')
        run('movement',files[0],label,str(files[0].relative_to(ROOT)),'dataset Test folder')

fields=list(dict.fromkeys(key for row in rows for key in row))
with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
lines=['# Data sample test','',
       'Scope: 4 face frames, 4 speech recordings, 8 movement videos. NVIDIA MediaPipe GPU delegate; speech/classifiers use CPU.',
       'Sampling was fixed before predictions. Classification comparison uses threshold 0.5.',
       'Face split membership is unknown and may overlap training. Speech speakers follow the repository split; movement uses Test folders.',
       'These small samples are execution checks, not independent accuracy estimates or clinical validation.',
       'Modalities are from unrelated datasets/people. No fused score was computed or training performed.','',
       '| Module | Successful | Selected | Matches at 0.5 |','|---|---:|---:|---:|']
for name in ['face','speech','movement']:
    group=[r for r in rows if r['modality']==name]
    lines.append(f'| {name} | {sum(r["status"]=="success" for r in group)} | {len(group)} | {sum(r.get("matches_label",False) for r in group)} |')
lines+=['','See results.csv/results.json for exact paths, scores, failures, and timings.',
        'Extracted face JPGs and speech WAVs are in samples/. Movement files are read directly from Data; their paths are in the report.']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('DONE',OUT,flush=True)

from src.accuracy_summary import write_summary
write_summary(rows, OUT)
