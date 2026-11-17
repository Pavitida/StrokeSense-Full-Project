"""Additional-data experiments. CLI separates CPU/GPU feature extraction from cuML fitting."""
import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / 'datasets/prepared/additional_20261116T063713088496Z'
VERSION = 'combined_v1'
SPEECH_COLS = [f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
SIGNALS = ['left_elbow','right_elbow','left_shoulderline','right_shoulderline',
           'left_extension','right_extension','elbow_asymmetry','shoulder_asymmetry','wrist_distance']
MOVE_COLS = [f'{s}_{stat}' for s in SIGNALS for stat in ('mean','std','min','max','rom')]


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def csv(path):
    return pd.read_csv(path, dtype={'person_id':str,'candidate_person_id':str})


def inputs(modality):
    if modality == 'speech':
        old = csv(ROOT/'training_data_v2/speech/manifest.csv')
        old = old[old.status.eq('accepted')].copy()
        old['label'] = old.label.map({'healthy':0,'dysarthria':1})
        old['source']='torgo'; old['path']=old.source_path
        old['person_id']='torgo:'+old.person_id
        old['kind']='audio'
        new = csv(PREPARED/'easycall/manifest.csv')
        new['path']=new.path.map(lambda p: str((PREPARED/'easycall'/p).relative_to(ROOT)))
        new['source']='easycall';new['person_id']='easycall:'+new.person_id;new['kind']='audio'
    else:
        old=csv(ROOT/'training_data_v2/movement/pose_gpu_v2/split.csv')
        old['label']=old.label.map({'Complete':0,'Incomplete':1})
        old['source']='original';old['path']=old.source_path
        old['person_id']='original:'+old.candidate_person_id;old['kind']='video'
        new=csv(PREPARED/'intellirehabds/manifest.csv')
        new=new[new.gesture.between(0,6)].copy()  # 7/8 are leg exercises, not arms.
        new['path']=new.path.map(lambda p:str((PREPARED/'intellirehabds'/p).relative_to(ROOT)))
        new['source']='intellirehabds';new['person_id']='intellirehabds:'+new.person_id
        new['exercise']=new.gesture.map(lambda g:'IRDS_'+str(g));new['kind']='skeleton'
    fields=['source','person_id','split','label','path','kind']
    if modality=='movement': fields+=['exercise']
    result=pd.concat([old[fields],new[fields]],ignore_index=True)
    if result[fields].isna().any().any(): raise ValueError('Missing input metadata')
    result['label']=result.label.astype(int)
    if set(result.label)!={0,1}: raise ValueError('Unexpected label')
    if not set(result.split)<= {'train','validation','test'}: raise ValueError('Unexpected split')
    if result.groupby('person_id').split.nunique().max()!=1: raise ValueError('Person leakage')
    if result.path.duplicated().any(): raise ValueError('Duplicate paths')
    for p in result.path:
        if not (ROOT/p).is_file(): raise FileNotFoundError(ROOT/p)
    return result


def speech_features(path):
    import librosa
    import soundfile as sf
    y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all(): raise ValueError('invalid_audio')
    if sr!=16000: y=librosa.resample(y,orig_sr=sr,target_sr=16000,res_type='soxr_hq')
    if np.max(np.abs(y))<=1e-8: raise ValueError('silent_audio')
    y=y/np.max(np.abs(y))
    content=hashlib.sha256(y.astype('<f4').tobytes()).hexdigest()
    mfcc=librosa.feature.mfcc(y=y,sr=16000,n_mfcc=13,n_fft=2048,hop_length=512,
                             n_mels=128,fmin=0,fmax=8000,htk=False,dct_type=2,norm='ortho',
                             lifter=0,center=True,pad_mode='constant',window='hann',power=2.0)
    values=np.stack([mfcc.mean(axis=1),mfcc.std(axis=1)],axis=1).ravel()
    return dict(zip(SPEECH_COLS,map(float,values))),{'canonical_sha256':content}


def common_movement(x):
    """Joint order LS,LE,LW,RS,RE,RW, all in a common metric 3D space.
    Angles/distance ratios are invariant to translation, global rotation/reflection and scale.
    No image-normalized MediaPipe z or frame-rate-dependent derivatives are used.
    """
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(6,3) or len(x)<3: raise ValueError('insufficient_joints')
    def norm(a): return np.linalg.norm(a,axis=-1)
    def angle(a,b,c):
        u=a-b;v=c-b;den=norm(u)*norm(v)
        if (den<1e-8).any(): raise ValueError('degenerate_joint')
        return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/den,-1,1)))
    ls,le,lw,rs,re,rw=x.transpose(1,0,2)
    shoulder=norm(ls-rs)
    left_len=norm(ls-le)+norm(le-lw);right_len=norm(rs-re)+norm(re-rw)
    if not np.isfinite(x).all() or (shoulder<1e-8).any() or (left_len<1e-8).any() or (right_len<1e-8).any():
        raise ValueError('invalid_skeleton_geometry')
    a,b=angle(ls,le,lw),angle(rs,re,rw)
    # Upper-arm angle relative to the shoulder line; does not require occluded hips.
    c,d=angle(rs,ls,le),angle(ls,rs,re)
    signals=[a,b,c,d,norm(ls-lw)/left_len,norm(rs-rw)/right_len,np.abs(a-b),np.abs(c-d),norm(lw-rw)/shoulder]
    values=[]
    for s in signals: values.extend([s.mean(),s.std(),s.min(),s.max(),np.ptp(s)])
    return dict(zip(MOVE_COLS,map(float,values)))


def skeleton_features(path):
    with np.load(path) as f: x=f['xyz'][:,[4,5,6,8,9,10],:]
    # Fixed 64 points over normalized clip duration, matching video aggregation.
    x=x[np.linspace(0,len(x)-1,min(64,len(x))).round().astype(int)]
    return common_movement(x),{'valid_frames':len(x),'canonical_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()}


def video_features(path,delegate='GPU'):
    import cv2
    import mediapipe as mp
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0 or total<3:
        cap.release();raise ValueError('invalid_video_metadata')
    frames=set(np.linspace(0,total-1,min(64,total)).round().astype(int).tolist())
    indices=[11,13,15,12,14,16];points=[];sampled=0;visibility=[]
    options=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/mediapipe/pose_landmarker.task'),
                                         delegate=getattr(mp.tasks.BaseOptions.Delegate,delegate)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
            i=0;last=-1
            while True:
                ok,frame=cap.read()
                if not ok: break
                if i in frames:
                    sampled+=1;stamp=max(last+1,round(i/fps*1000));last=stamp
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=model.detect_for_video(image,stamp)
                    if result.pose_world_landmarks and result.pose_landmarks:
                        visible=result.pose_landmarks[0]
                        # Match original extractor's pose-detection criterion. Occluded joints
                        # are model estimates; log confidence rather than claiming observation.
                        world=result.pose_world_landmarks[0]
                        points.append([[world[j].x,world[j].y,world[j].z] for j in indices])
                        visibility.append([visible[j].visibility for j in indices])
                i+=1
    finally: cap.release()
    rate=len(points)/max(sampled,1)
    if sampled!=len(frames) or len(points)<3 or rate<.4:
        raise ValueError(f'low_pose_coverage(valid={len(points)},sampled={sampled},expected={len(frames)})')
    return common_movement(points),{'valid_frames':len(points),'sampled_frames':sampled,'pose_rate':rate,
                                  'mean_joint_visibility':float(np.mean(visibility)),
                                  'fraction_all_joints_visible':float(np.mean(np.all(np.asarray(visibility)>=.5,axis=1))),
                                  'canonical_sha256':sha(path)}


def prepare(modality,delegate):
    data=inputs(modality)
    cache=ROOT/'training_data_v2/combined'/VERSION/modality;cache.mkdir(parents=True,exist_ok=True)
    data.to_csv(cache/'input_manifest.csv',index=False)
    data[data.split.eq('test')].to_csv(cache/'reserved_test.csv',index=False)
    dev=data[data.split.isin(['train','validation'])]
    records=[];audit=[];clips=cache/'clips';clips.mkdir(exist_ok=True)
    # Include implementation and library versions in cache keys; interrupted runs can resume.
    import importlib.metadata as metadata
    libs={n:metadata.version(n) for n in (['numpy','librosa','soundfile'] if modality=='speech' else ['numpy','mediapipe'])}
    if modality=='movement':
        import cv2
        libs['opencv']=cv2.__version__
    signature=sha(__file__)+json.dumps(libs,sort_keys=True)+delegate
    for i,row in enumerate(dev.to_dict('records'),1):
        path=ROOT/row['path'];source_hash=sha(path)
        key=hashlib.sha256((signature+row['path']+source_hash).encode()).hexdigest();entry=clips/(key+'.json')
        if entry.exists(): result=json.loads(entry.read_text())
        else:
            try:
                fn=speech_features if row['kind']=='audio' else skeleton_features if row['kind']=='skeleton' else None
                feat,quality=fn(path) if fn else video_features(path,delegate)
                if not np.isfinite(list(feat.values())).all(): raise ValueError('nonfinite_features')
                result=dict(status='accepted',**feat,**quality)
            except ValueError as e: result=dict(status='excluded',reason=str(e))
            # Runtime/GPU/dependency errors deliberately stop; never silently drop a whole source.
            tmp=entry.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(entry)
        audit.append(dict(row,**result))
        if result['status']=='accepted': records.append(dict(row,**result))
        if i%100==0 or i==len(dev): print(f'{modality}: {i}/{len(dev)} features',flush=True)
    frame=pd.DataFrame(records)
    if frame.empty: raise RuntimeError('No usable features')
    duplicate=frame.canonical_sha256.duplicated(keep=False)
    frame[duplicate].to_csv(cache/'duplicate_exclusions.csv',index=False)
    frame=frame[~duplicate]
    frame.to_csv(cache/'features.csv',index=False)
    pd.DataFrame(audit).to_csv(cache/'extraction_audit.csv',index=False)
    (cache/'protocol.json').write_text(json.dumps({'version':VERSION,'libraries':libs,'script_sha256':sha(__file__),
        'manifest_sha256':sha(cache/'input_manifest.csv'),'features_sha256':sha(cache/'features.csv'),
        'test_scored':False,'delegate':delegate,'features':SPEECH_COLS if modality=='speech' else MOVE_COLS,
        'movement_target':'exploratory exercise correctness; task definitions differ between sources',
        'movement_pose_quality':'pose coverage >=0.4; occluded joints are estimates; visibility logged, not filtered',
        'original_movement_person_ids':'filename groups; participant identity not verified'},indent=2))
    validate(frame,modality)
    print(frame.groupby(['source','split','label']).size().to_string(),flush=True)
    print('Prepared:',cache,flush=True)


def validate(frame,modality):
    cols=SPEECH_COLS if modality=='speech' else MOVE_COLS
    if not np.isfinite(frame[cols].to_numpy()).all(): raise ValueError('Nonfinite features')
    if frame.groupby('person_id').split.nunique().max()!=1: raise ValueError('Person leakage')
    if frame.canonical_sha256.duplicated().any(): raise ValueError('Duplicate content')
    expected={'torgo','easycall'} if modality=='speech' else {'original','intellirehabds'}
    if set(frame.source)!=expected: raise ValueError('Missing source')
    for source in expected:
        for split in ['train','validation']:
            if set(frame.loc[frame.source.eq(source)&frame.split.eq(split),'label'])!={0,1}:
                raise ValueError(f'Missing class in {source}/{split}')


def train(modality):
    import cupy as cp
    import cuml
    from cuml.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,recall_score,confusion_matrix,roc_auc_score
    cache=ROOT/'training_data_v2/combined'/VERSION/modality
    protocol=json.loads((cache/'protocol.json').read_text())
    if protocol['script_sha256']!=sha(__file__) or protocol['features_sha256']!=sha(cache/'features.csv'):
        raise ValueError('Code/features changed; rerun prepare')
    if protocol['manifest_sha256']!=sha(cache/'input_manifest.csv'): raise ValueError('Manifest changed')
    data=csv(cache/'features.csv');validate(data,modality)
    if set(data.split)!={'train','validation'}: raise ValueError('Test must remain reserved')
    cols=SPEECH_COLS if modality=='speech' else MOVE_COLS
    out=ROOT/'outputs/combined_gpu'/modality/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    old='torgo' if modality=='speech' else 'original';val=data[data.split.eq('validation')]
    metrics=[];predictions=[]
    print('CUDA:',cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),flush=True)
    for name in ['old_only','combined']:
        fit=data[data.split.eq('train') & (data.source.eq(old) if name=='old_only' else True)].copy()
        x=fit[cols].to_numpy(dtype='float32');mean=x.mean(0);std=x.std(0);std=np.where(std<1e-8,1,std)
        # Equal total training weight per source and label prevents the larger source dominating.
        counts=fit.groupby(['source','label']).label.transform('size').to_numpy()
        weight=1/counts;weight=weight/weight.mean()
        model=LogisticRegression(C=1,max_iter=2000,tol=1e-4,output_type='cupy')
        print('Training on CUDA:',modality,name,'LogisticRegression',len(fit),flush=True)
        model.fit(cp.asarray((x-mean)/std),cp.asarray(fit.label.to_numpy(),dtype=cp.int32),sample_weight=cp.asarray(weight,dtype=cp.float32))
        prob=cp.asnumpy(model.predict_proba(cp.asarray((val[cols].to_numpy(dtype='float32')-mean)/std)))[:,1]
        pred=val[['source','person_id','split','label','path']].copy();pred['experiment']=name
        pred['probability']=prob;pred['prediction']=(prob>=.5).astype(int);predictions.append(pred)
        for source in ['ALL']+sorted(val.source.unique()):
            rows=pred if source=='ALL' else pred[pred.source.eq(source)]
            y=rows.label.to_numpy();p=rows.prediction.to_numpy()
            metrics.append(dict(experiment=name,source=source,n=len(rows),accuracy=accuracy_score(y,p),
                balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),
                positive_recall=recall_score(y,p,zero_division=0),roc_auc=roc_auc_score(y,rows.probability),
                confusion_matrix=json.dumps(confusion_matrix(y,p,labels=[0,1]).tolist())))
        with (out/(name+'_LogisticRegression.pkl')).open('wb') as f:
            pickle.dump(dict(model=model,mean=mean,std=std,feature_columns=cols,modality=modality,
                label_map={'healthy':0,'dysarthria':1} if modality=='speech' else {'correct_or_complete':0,'incorrect_or_incomplete':1},
                feature_version=VERSION,training_sources=sorted(fit.source.unique()),threshold=.5),f)
    pd.DataFrame(metrics).to_csv(out/'validation_metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'validation_predictions.csv',index=False)
    protocol.update(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cuml=cuml.__version__,
                    input_features=str(cache),test_scored=False,weighting='equal source x class totals',
                    baseline='old-only retrained with identical feature extraction, not old saved model')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False));print('Saved:',out,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','prepare','train','smoke'])
    parser.add_argument('modality',choices=['speech','movement','all']);parser.add_argument('--pose-delegate',choices=['GPU','CPU'],default='GPU')
    args=parser.parse_args();os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
    for modality in (['speech','movement'] if args.modality=='all' else [args.modality]):
        if args.action=='train': train(modality)
        elif args.action=='prepare': prepare(modality,args.pose_delegate)
        else:
            data=inputs(modality);print(modality,data.groupby(['source','split']).size().to_dict())
            if args.action=='smoke':
                for source,group in data[data.split.eq('train')].groupby('source'):
                    # Cover different recordings rather than ten adjacent repeats of one task.
                    samples=group.iloc[np.linspace(0,len(group)-1,min(10,len(group))).round().astype(int)]
                    for row in samples.itertuples():
                        path=ROOT/row.path
                        try:
                            feat,_=speech_features(path) if modality=='speech' else skeleton_features(path) if row.kind=='skeleton' else video_features(path,args.pose_delegate)
                        except ValueError as e:
                            print('Smoke quality exclusion:',path.name,str(e),flush=True)
                            continue
                        assert np.isfinite(list(feat.values())).all()
                        print('Feature smoke PASS:',source,len(feat),flush=True)
                        break
                    else: raise RuntimeError('No passing smoke sample in 10 spread recordings: '+source)


if __name__=='__main__': main()
