"""R6 frozen Gate-0 baseline: stream R4 values, train three matched MLPs, and evaluate once."""
import argparse, csv, ctypes, hashlib, json, math, os, random, sqlite3, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
CACHE = ROOT / "cache_r6"
CKPT = ROOT / "checkpoints_r6"
FIELDS = ["MergedReflectivityQCComposite", "Reflectivity_-10C", "EchoTop_30"]
SEEDS = [2026091001, 2026091002, 2026091003]
R_EARTH_KM = 6371.0088
POLICY = {"architecture": "two-layer MLP: Linear(input,32), ReLU, Linear(32,1)", "hidden_units": 32,
          "optimizer": "AdamW(lr=0.001, weight_decay=1e-5)", "loss": "TRAIN-derived global weighted BCE",
          "max_epochs": 6, "early_stopping_metric": "validation PR-AUC", "patience": 2,
          "batch_size_cuda": 65536, "batch_size_cpu": 8192, "history": "t-25,t-20,t-15,t-10,t-5,t",
          "lightning": "per-frame log1p(count within 20 km), nearest-flash distance km capped at 100",
          "test_rule": "No TEST features, labels, probabilities, or metrics are read until all nine checkpoints and validation thresholds are frozen."}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def dump(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True); q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v,'item') else str(v)), encoding='utf-8')
    json.loads(q.read_text(encoding='utf-8')); os.replace(q,p)
def writecsv(p, rows, fields=None):
    rows=list(rows); fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    q=Path(p).with_suffix(Path(p).suffix+'.tmp')
    with q.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(q,p)
def truth(v): return str(v).lower()=='true'
def diskfree(): return int(os.statvfs(str(ROOT)).f_bavail*os.statvfs(str(ROOT)).f_frsize) if hasattr(os,'statvfs') else int(ctypes.c_ulonglong())
def windows_free():
    free=ctypes.c_ulonglong(); total=ctypes.c_ulonglong(); totalfree=ctypes.c_ulonglong()
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(str(ROOT), ctypes.byref(free), ctypes.byref(total), ctypes.byref(totalfree)); return int(free.value)
def cache_bytes(): return sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()) if CACHE.exists() else 0
def xyz(lat,lon):
    a=np.deg2rad(lat); o=np.deg2rad(lon); return np.column_stack((np.cos(a)*np.cos(o),np.cos(a)*np.sin(o),np.sin(a)))
def chord(km): return 2*np.sin((km/R_EARTH_KM)/2)
def peak_rss():
    try:
        class PMC(ctypes.Structure): _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        x=PMC(); x.cb=ctypes.sizeof(x); ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(x),x.cb); return int(x.PeakWorkingSetSize)
    except Exception: return None

def read_splits():
    out={}
    with (ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): out[r['date']]=r['split']
    emb=defaultdict(set)
    with (ROOT/'SPLIT_BOUNDARY_EMBARGO_R55.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if truth(r['SPLIT_EMBARGO']): emb[r['date']].add(int(r['time_index']))
    return out,emb
SPLIT,EMBARGO=read_splits()

def policy_and_state():
    CACHE.mkdir(exist_ok=True); CKPT.mkdir(exist_ok=True)
    state=CACHE/'run_state.json'
    freeze=ROOT/'TEST_SET_FREEZE_R55.json'; r55=ROOT/'R55_VALIDATION.json'; r52=ROOT/'R52_VALIDATION.json'
    for p in [freeze,r55,r52]:
        assert p.exists(), f'MISSING_FROZEN_INPUT:{p}'
    d=json.loads(r55.read_text(encoding='utf-8')); assert d.get('SPLIT_GATE_R55')=='PASS'
    d=json.loads(r52.read_text(encoding='utf-8')); assert d.get('LABEL_READINESS_GATE_R52')=='PASS'
    policy={**POLICY,'seeds':SEEDS,'split_assignment_sha256':sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv'),'embargo_sha256':sha(ROOT/'SPLIT_BOUNDARY_EMBARGO_R55.csv'),'test_freeze_sha256':sha(freeze),'formula_note':'Section 20 later explicit definitions govern: GI=CSI_I_FUSION-CSI_I_MET; GO=CSI_O_FUSION-CSI_O_MET; Gamma=GO-GI.'}
    pp=ROOT/'TRAINING_POLICY_R6.md'
    text='# R6 frozen training policy\n\n'+''.join(f'- **{k}**: {v}\n' for k,v in policy.items())
    if not pp.exists(): pp.write_text(text,encoding='utf-8')
    if not state.exists(): dump(state,{'utc_started':now(),'free_disk_before_bytes':windows_free(),'policy_sha256':sha(pp),'test_freeze_sha256':sha(freeze),'test_phase_started':False})
    return json.loads(state.read_text(encoding='utf-8'))

def source_paths(day, required_indices):
    """Resolve only source records actually referenced by R52-maskable sample histories.

    The immutable runtime index was built and validated by R6_SOURCE_KEY_COMPATIBILITY_FIX.
    It deliberately never iterates blank keys in frozen, unused R4 timestamps.
    """
    index=ROOT/'runtime_r6'/'source_resolution_index.csv'
    assert index.exists(), 'R6_RUNTIME_SOURCE_INDEX_MISSING'
    db=sqlite3.connect(str(ROOT/'recovery_r4'/'FILE_QC_R4.sqlite'))
    rows=db.execute('select key,result from files where date=? and dataset in (?,?,?)',(day,*FIELDS)).fetchall(); db.close()
    paths={key:json.loads(result).get('roi_path','') for key,result in rows}
    out={f:[None]*288 for f in FIELDS}; wanted=set(map(int,required_indices)); found=set()
    with index.open(encoding='utf-8-sig',newline='') as handle:
        for r in csv.DictReader(handle):
            if r['date']!=day or int(r['history_time_index']) not in wanted: continue
            assert truth(r['source_resolved']) and r['original_source_key'].strip(), 'UNRESOLVED_REQUIRED_RUNTIME_SOURCE'
            p=paths.get(r['original_source_key'],''); assert p and Path(p).exists(), 'RUNTIME_SOURCE_ROI_MISSING'
            key=(r['product'],int(r['history_time_index'])); assert key not in found, 'RUNTIME_SOURCE_INDEX_DUPLICATE'
            found.add(key); out[r['product']][key[1]]=p
    expected={(f,i) for f in FIELDS for i in wanted}; assert found==expected, f'RUNTIME_SOURCE_INDEX_INCOMPLETE:{len(expected-found)}'
    return out

def flash(day):
    with np.load(ROOT/'recovery_r5'/'flash_daily'/(day+'.npz')) as f: a={k:f[k] for k in f.files}
    order=np.lexsort((a['flash_id'],a['time'])); a={k:v[order] for k,v in a.items()}
    pair=np.rec.fromarrays([a['flash_id'],a['time']],names='i,t'); keep=np.r_[True,pair[1:]!=pair[:-1]] if len(pair) else np.array([],bool)
    return {k:v[keep] for k,v in a.items()}

def selected_rows(day, include_y=True):
    lp=ROOT/'LABEL_STORE_R5'/(day+'.nc'); sp=ROOT/'MODEL_READINESS_R52'/(day+'.nc')
    with nc.Dataset(lp) as l, nc.Dataset(sp) as s:
        l.set_auto_mask(False); s.set_auto_mask(False)
        ti=l['time_index'][:]; pi=l['pixel_index'][:]; flags=s['readiness_flags'][:]
        take=((flags & 4)!=0)&~np.isin(ti,list(EMBARGO[day]))
        d={'row':np.flatnonzero(take).astype('u4'),'time_index':ti[take].astype('u2'),'pixel_index':pi[take].astype('u4'),'bits':s['history_validity_mask_bits'][take].astype('u4'),'lat':l['lat'][:],'lon':l['lon'][:],'time':l['time'][:]}
        if include_y: d.update(y=l['Y_primary'][take].astype('u1'),h=l['H_primary'][take].astype('u1'),clazz=l['class_primary'][take].astype('i1'))
    return d

def build_date(day, allow_test=False):
    split=SPLIT[day]; assert split!='TEST' or allow_test, 'TEST_FEATURE_READ_BEFORE_FREEZE'
    dest=CACHE/'features'/(day+'.npz'); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): return dest
    q=selected_rows(day,True); n=len(q['row']); assert n>0, 'NO_SELECTED_SAMPLES'
    print('R6_CACHE_START',day,split,n,flush=True)
    required_indices=sorted({h for t in np.unique(q['time_index']) for h in range(int(t)-5,int(t)+1)})
    met=np.zeros((n,36),dtype='f4'); ltg=np.zeros((n,12),dtype='f4'); src=source_paths(day,required_indices); fl=flash(day)
    raw_cache={}
    def raw(path):
        if path not in raw_cache:
            with np.load(path) as z: raw_cache[path]=z['value_raw'].ravel().astype('f4',copy=False)
        return raw_cache[path]
    starts=np.r_[0,np.flatnonzero(np.diff(q['time_index']))+1]; stops=np.r_[starts[1:],n]
    for lo,hi in zip(starts,stops):
        tix=int(q['time_index'][lo]); pix=q['pixel_index'][lo:hi]; pts=xyz(q['lat'][pix//800],q['lon'][pix%800]); bits=q['bits'][lo:hi]
        assert tix>=5
        for b,hix in enumerate(range(tix-5,tix+1)):
            for fi,name in enumerate(FIELDS):
                valid=((bits>>(fi*6+b))&1).astype(bool); vals=raw(src[name][hix])[pix]
                met[lo:hi,b*3+fi]=np.where(valid,vals,0.0); met[lo:hi,18+b*3+fi]=valid.astype('f4')
            end=float(q['time'][hix]); keep=(fl['time']>end-300)&(fl['time']<=end)
            if keep.any():
                tree=cKDTree(xyz(fl['lat'][keep],fl['lon'][keep])); d,_=tree.query(pts,k=1); dist=2*R_EARTH_KM*np.arcsin(np.minimum(1,d/2)); cnt=tree.query_ball_point(pts,chord(20),return_length=True)
            else: dist=np.full(hi-lo,100.0); cnt=np.zeros(hi-lo,dtype='i4')
            ltg[lo:hi,2*b]=np.log1p(cnt); ltg[lo:hi,2*b+1]=np.minimum(dist,100.0)
        if len(raw_cache)>54: raw_cache.clear()
    tmp=dest.with_suffix('.tmp.npz'); np.savez(tmp,met=met,ltg=ltg,y=q['y'],h=q['h'],clazz=q['clazz'],row=q['row'],time_index=q['time_index'],pixel_index=q['pixel_index']); os.replace(tmp,dest)
    print('R6_CACHE_DONE',day,n,dest.stat().st_size,flush=True); return dest

def manifest(allow_test=False):
    rows=[]; total=hashlib.sha256()
    for day in sorted(SPLIT):
        if SPLIT[day]=='TEST' and not allow_test: continue
        q=selected_rows(day,True); h=hashlib.sha256(); h.update(day.encode()); h.update(q['row'].tobytes()); total.update(day.encode()+h.digest())
        row={'date':day,'split':SPLIT[day],'n':len(q['row']),'sample_id_sha256':h.hexdigest(),'I+':int((q['clazz']==0).sum()),'I-':int((q['clazz']==1).sum()),'O+':int((q['clazz']==2).sum()),'O-':int((q['clazz']==3).sum()),'embargoed_rows_removed':int(np.isin(q['time_index'],list(EMBARGO[day])).sum())}
        rows.append(row)
    writecsv(ROOT/'TRAIN_SAMPLE_MANIFEST_R6.csv',rows)
    out={'utc':now(),'same_candidate_samples_for_all_models':True,'contains_test':allow_test,'rows':rows,'all_sample_id_sha256':total.hexdigest(),'split_sha256':sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv')}
    dump(ROOT/'TRAIN_SAMPLE_MANIFEST_R6.json',out); return out

def normalization():
    out=ROOT/'TRAIN_NORMALIZATION_R6.json'
    if out.exists(): return json.loads(out.read_text(encoding='utf-8'))
    sums=np.zeros(30,dtype='f8'); ss=np.zeros(30,dtype='f8'); cnt=np.zeros(30,dtype='i8')
    # 18 meteorological values are conditionally accumulated only when their paired mask is one; 12 GLM values are always defined.
    for day in sorted(d for d in SPLIT if SPLIT[d]=='TRAIN'):
        p=build_date(day)
        with np.load(p) as z:
            x=z['met']; g=z['ltg']
            for j in range(18):
                m=x[:,18+j].astype(bool); a=x[m,j]; sums[j]+=a.sum(dtype='f8'); ss[j]+=np.dot(a.astype('f8'),a.astype('f8')); cnt[j]+=len(a)
            for j in range(12):
                a=g[:,j]; sums[18+j]+=a.sum(dtype='f8'); ss[18+j]+=np.dot(a.astype('f8'),a.astype('f8')); cnt[18+j]+=len(a)
    mean=sums/cnt; std=np.sqrt(np.maximum(ss/cnt-mean*mean,1e-12))
    obj={'utc':now(),'source_split':'TRAIN_ONLY','met_value_mean':mean[:18].tolist(),'met_value_std':std[:18].tolist(),'met_valid_counts':cnt[:18].tolist(),'ltg_mean':mean[18:].tolist(),'ltg_std':std[18:].tolist(),'ltg_counts':cnt[18:].tolist()}
    dump(out,obj); return obj

def nrm(z,norm,kind, intervention=None):
    met=z['met'].astype('f4',copy=True); ltg=z['ltg'].astype('f4',copy=True)
    mm=np.asarray(norm['met_value_mean'],'f4'); ms=np.asarray(norm['met_value_std'],'f4'); lm=np.asarray(norm['ltg_mean'],'f4'); ls=np.asarray(norm['ltg_std'],'f4')
    met[:,:18]=(met[:,:18]-mm)/ms; ltg=(ltg-lm)/ls
    if intervention=='GLM_HISTORY_MASKED': ltg[:]=0.0
    if intervention=='MRMS_MASKED': met[:,:18]=0.0
    return ltg if kind=='M_LTG' else met if kind=='M_MET' else np.hstack([met,ltg])

class MLP(torch.nn.Module):
    def __init__(self,n): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(n,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(1)
def pr_auc(y,p):
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype='f8'); pos=int(y.sum())
    if pos==0:return float('nan')
    o=np.argsort(p,kind='mergesort')[::-1]; yy=y[o]; tp=np.cumsum(yy); fp=np.cumsum(~yy); rec=tp/pos; prec=tp/(tp+fp); return float(np.sum((rec-np.r_[0,rec[:-1]])*prec))
def csi_threshold(y,p):
    # exact candidate thresholds from sorted score change points; selects smallest score on equal CSI.
    y=np.asarray(y,bool); o=np.argsort(p)[::-1]; ps=np.asarray(p)[o]; yy=y[o]; tp=np.cumsum(yy); fp=np.cumsum(~yy); total=int(yy.sum()); change=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]; c=tp[change]/np.maximum(tp[change]+fp[change]+total-tp[change],1); k=int(np.argmax(c)); return float(ps[change[k]]),float(c[k])
def metrics(y,p,t):
    y=np.asarray(y,bool); pred=np.asarray(p)>=t; tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=int((~pred&y).sum());
    return {'n':int(len(y)),'positive_rate':float(y.mean()),'CSI':tp/max(tp+fp+fn,1),'POD':tp/max(tp+fn,1),'FAR':fp/max(tp+fp,1),'F1':2*tp/max(2*tp+fp+fn,1),'PR_AUC':pr_auc(y,p),'Brier':float(np.mean((np.asarray(p)-y)**2)),'threshold':float(t),'TP':tp,'FP':fp,'FN':fn}

def data_days(split): return [d for d in sorted(SPLIT) if SPLIT[d]==split]
def predict(model,kind,norm,days,device, intervention=None):
    pp=[]; yy=[]; hh=[]; dd=[]; batch=65536 if device.type=='cuda' else 8192
    model.eval()
    with torch.no_grad():
        for day in days:
            with np.load(CACHE/'features'/(day+'.npz')) as z:
                x=nrm(z,norm,kind,intervention); y=z['y']; h=z['h']
                out=[]
                for lo in range(0,len(y),batch):
                    q=torch.from_numpy(x[lo:lo+batch]).to(device); out.append(torch.sigmoid(model(q)).cpu().numpy())
                pp.append(np.concatenate(out)); yy.append(y); hh.append(h); dd.append(np.full(len(y),day))
    return np.concatenate(pp),np.concatenate(yy),np.concatenate(hh),np.concatenate(dd)

def checkpoint_path(kind,seed): return CKPT/f'{kind}_seed{seed}.pt'
def train_one(kind,seed,norm,manifest_hash,state):
    cp=checkpoint_path(kind,seed); valp=CACHE/'predictions'/f'validation_{kind}_seed{seed}.npy'; valp.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ins={'M_LTG':12,'M_MET':36,'M_FUSION':48}[kind]; model=MLP(ins).to(device); opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-5)
    start=0; best=-math.inf; stale=0; hist=[]
    if cp.exists():
        a=torch.load(cp,map_location=device,weights_only=False); assert a['seed']==seed and a['kind']==kind and a['normalization_sha256']==sha(ROOT/'TRAIN_NORMALIZATION_R6.json') and a['manifest_sha256']==manifest_hash
        model.load_state_dict(a['model']); opt.load_state_dict(a['optimizer']); start=a['epoch']+1; best=a['best']; stale=a['stale']; hist=a['history']
        if a.get('complete') and valp.exists(): return {'kind':kind,'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'best_validation_pr_auc':best,'epochs':a['epoch']+1,'device':str(device),'resumed':True}
    pos=neg=0
    for day in data_days('TRAIN'):
        with np.load(CACHE/'features'/(day+'.npz')) as z: pos+=int(z['y'].sum()); neg+=len(z['y'])-int(z['y'].sum())
    pw=float(neg/max(pos,1)); lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device)); batch=POLICY['batch_size_cuda'] if device.type=='cuda' else POLICY['batch_size_cpu']
    for epoch in range(start,POLICY['max_epochs']):
        model.train(); running=0.; seen=0; rng=np.random.default_rng(seed+epoch)
        for day in rng.permutation(data_days('TRAIN')):
            with np.load(CACHE/'features'/(day+'.npz')) as z:
                x=nrm(z,norm,kind); y=z['y'].astype('f4'); order=rng.permutation(len(y))
                for lo in range(0,len(y),batch):
                    ix=order[lo:lo+batch]; xx=torch.from_numpy(x[ix]).to(device); yy=torch.from_numpy(y[ix]).to(device); opt.zero_grad(set_to_none=True); loss=lossfn(model(xx),yy); loss.backward(); opt.step(); running+=float(loss.detach().cpu())*len(ix); seen+=len(ix)
        vp,vy,_,_=predict(model,kind,norm,data_days('VALIDATION'),device); vpr=pr_auc(vy,vp); hist.append({'epoch':epoch,'train_weighted_bce':running/max(seen,1),'validation_pr_auc':vpr})
        print('R6_EPOCH',kind,seed,epoch,round(running/max(seen,1),7),round(vpr,7),'stale',stale,flush=True)
        if vpr>best: best=vpr; stale=0; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else: stale+=1
        payload={'kind':kind,'seed':seed,'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'stale':stale,'history':hist,'normalization_sha256':sha(ROOT/'TRAIN_NORMALIZATION_R6.json'),'split_sha256':sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv'),'manifest_sha256':manifest_hash,'complete':False,'policy_sha256':sha(ROOT/'TRAINING_POLICY_R6.md')}
        torch.save(payload,cp)
        if stale>=POLICY['patience']: break
    model.load_state_dict(best_state); vp,vy,_,_=predict(model,kind,norm,data_days('VALIDATION'),device); np.save(valp,vp.astype('f4'))
    payload.update(model=model.state_dict(),complete=True,epoch=hist[-1]['epoch'],best=best,best_state_saved=True); torch.save(payload,cp)
    return {'kind':kind,'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'best_validation_pr_auc':best,'epochs':len(hist),'device':str(device),'resumed':start>0}

def validation_thresholds():
    rows=[]; ens={}
    for kind in ['M_LTG','M_MET','M_FUSION']:
        a=[]
        for seed in SEEDS:
            p=np.load(CACHE/'predictions'/f'validation_{kind}_seed{seed}.npy'); a.append(p)
            with np.load(CACHE/'features'/(data_days('VALIDATION')[0]+'.npz')) as z: pass
            # labels have deterministic concatenation and are read only from validation caches.
            y=np.concatenate([np.load(CACHE/'features'/(d+'.npz'))['y'] for d in data_days('VALIDATION')]); t,c=csi_threshold(y,p); rows.append({'model':kind,'seed':seed,'scope':'VALIDATION_ALL','threshold':t,'CSI':c,'frozen_utc':now()})
        p=np.mean(a,axis=0); t,c=csi_threshold(y,p); ens[kind]=t; rows.append({'model':kind,'seed':'ENSEMBLE','scope':'VALIDATION_ALL','threshold':t,'CSI':c,'frozen_utc':now()})
    writecsv(ROOT/'VALIDATION_THRESHOLDS_R6.csv',rows); dump(CACHE/'ensemble_thresholds.json',ens); return ens

def load_complete(kind,seed,device):
    a=torch.load(checkpoint_path(kind,seed),map_location=device,weights_only=False); assert a.get('complete'); m=MLP({'M_LTG':12,'M_MET':36,'M_FUSION':48}[kind]).to(device); m.load_state_dict(a['model']); return m
def test_and_report(norm,thresholds,trainrows,state):
    st=json.loads((CACHE/'run_state.json').read_text(encoding='utf-8')); st['test_phase_started']=True; st['test_phase_utc']=now(); dump(CACHE/'run_state.json',st)
    print('R6_TEST_PHASE_START',now(),flush=True)
    for d in data_days('TEST'): build_date(d,allow_test=True)
    man=manifest(allow_test=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=[]; ensemble={}
    for kind in ['M_LTG','M_MET','M_FUSION']:
        pred=[]
        for seed in SEEDS:
            p,y,h,date=predict(load_complete(kind,seed,device),kind,norm,data_days('TEST'),device); pred.append(p); np.save(CACHE/'predictions'/f'test_{kind}_seed{seed}.npy',p.astype('f4'))
        ensemble[kind]=np.mean(pred,axis=0)
        for scope,mask in [('ALL',np.ones(len(y),bool)),('I',h==0),('O',h==1)]:
            out.append({'model':kind,'seed':'ENSEMBLE','scope':scope,**metrics(y[mask],ensemble[kind][mask],thresholds[kind]),'test_evaluated_utc':now(),'checkpoint_hashes':'|'.join(sha(checkpoint_path(kind,s)) for s in SEEDS)})
    writecsv(ROOT/'TEST_SUBGROUP_RESULTS_R6.csv',out)
    # B0 is Eulerian persistence p=H. B1 is a train-only frequency table over recent lightning-count bins.
    counts=np.zeros((2,5,2),dtype='i8')
    for d in data_days('TRAIN'):
        with np.load(CACHE/'features'/(d+'.npz')) as z:
            b=np.minimum(np.sum(np.expm1(z['ltg'][:,::2]),axis=1).astype(int),3); counts[z['h'],b,z['y']]+=1
    trainrate=(counts[:,:,1]+1)/(counts.sum(axis=2)+2); base=[]
    recent=np.minimum(np.sum(np.expm1(np.concatenate([np.load(CACHE/'features'/(d+'.npz'))['ltg'] for d in data_days('TEST')])[:,::2]),axis=1).astype(int),3)
    for name,p in [('B0_EULERIAN_PERSISTENCE',h.astype('f4')),('B1_LIGHTNING_HISTORY_STATISTICAL',trainrate[h,recent])]:
        for scope,mask in [('ALL',np.ones(len(y),bool)),('I',h==0),('O',h==1)]: base.append({'baseline':name,'scope':scope,**metrics(y[mask],p[mask],0.5),'fit_split':'TRAIN_ONLY' if name.startswith('B1') else 'NONE'})
    writecsv(ROOT/'PERSISTENCE_BASELINES_R6.csv',base)
    # inference-only ablations, value channels neutral after train normalization while validity channels are preserved.
    fusion_models=[load_complete('M_FUSION',s,device) for s in SEEDS]; inter=[]
    for mode in ['NORMAL','GLM_HISTORY_MASKED','MRMS_MASKED']:
        pp=[]
        for m in fusion_models: pp.append(predict(m,'M_FUSION',norm,data_days('TEST'),device,None if mode=='NORMAL' else mode)[0])
        p=np.mean(pp,axis=0)
        for scope,mask in [('I',h==0),('O',h==1)]: inter.append({'intervention':mode,'scope':scope,**metrics(y[mask],p[mask],thresholds['M_FUSION']),'value_neutral_after_train_normalization':mode!='NORMAL','validity_masks_preserved':True})
    normal={(r['scope']):r['CSI'] for r in inter if r['intervention']=='NORMAL'}
    for r in inter: r['Delta_CSI_vs_NORMAL']=normal[r['scope']]-r['CSI']
    writecsv(ROOT/'MASK_INTERVENTION_R6.csv',inter)
    by={(r['model'],r['scope']):r for r in out}; gi=by['M_FUSION','I']['CSI']-by['M_MET','I']['CSI']; go=by['M_FUSION','O']['CSI']-by['M_MET','O']['CSI']; gamma=go-gi
    pr_i=by['M_FUSION','I']['PR_AUC']-by['M_MET','I']['PR_AUC']; pr_o=by['M_FUSION','O']['PR_AUC']-by['M_MET','O']['PR_AUC']
    # Seed stability uses seed-level test predictions after all runs are frozen; it cannot alter models or thresholds.
    srows=[]
    for s in SEEDS:
        z={}
        for kind in ['M_MET','M_FUSION']:
            p=np.load(CACHE/'predictions'/f'test_{kind}_seed{s}.npy'); z[kind]={sc:metrics(y[m],p[m],thresholds[kind])['CSI'] for sc,m in [('I',h==0),('O',h==1)]}
        x=z['M_FUSION']['I']-z['M_MET']['I']; w=z['M_FUSION']['O']-z['M_MET']['O']; srows.append({'seed':s,'GI':x,'GO':w,'Gamma':w-x})
    stable=sum(r['Gamma']>0 for r in srows); sign='STABLE_POSITIVE' if stable==3 else 'MIXED' if stable==2 else 'UNSTABLE'
    for r in srows:r['SEED_SIGN_STABILITY']=sign
    writecsv(ROOT/'SEED_STABILITY_R6.csv',srows)
    # block bootstrap at the frozen full-date level.
    rng=np.random.default_rng(20260910); dates=np.unique(date); boot=[]
    for i in range(1000):
        ds=rng.choice(dates,len(dates),replace=True); ix=np.concatenate([np.flatnonzero(date==d) for d in ds]);
        vals={}
        for k in ['M_MET','M_FUSION']:
            vals[k]={sc:metrics(y[ix][h[ix]==hv],ensemble[k][ix][h[ix]==hv],thresholds[k])['CSI'] for sc,hv in [('I',0),('O',1)]}
        a=vals['M_FUSION']['I']-vals['M_MET']['I']; b=vals['M_FUSION']['O']-vals['M_MET']['O']; boot.append({'replicate':i,'GI':a,'GO':b,'Gamma':b-a})
    writecsv(ROOT/'GATE0_BOOTSTRAP_R6.csv',boot); ci=np.percentile([r['Gamma'] for r in boot],[2.5,97.5]).tolist()
    met_i=by['M_MET','I']; insufficient=met_i['PR_AUC']<=met_i['positive_rate']+0.005
    if insufficient: gate='BLOCKED_BY_WEAK_MET_BASELINE'
    elif go>=.05 and gi<=0 and ci[0]>0: gate='STRONG_PASS'
    elif gamma>=.03 and ci[0]>0 and ((go>=2*gi) if gi>=0 else go>0): gate='PASS'
    elif gamma<=0 and sign!='MIXED': gate='KILL'
    else: gate='HOLD'
    result={'GI':gi,'GO':go,'Gamma':gamma,'Gamma_CI95':ci,'PR_AUC_GAIN_I':pr_i,'PR_AUC_GAIN_O':pr_o,'Gamma_PR':pr_o-pr_i,'SEED_SIGN_STABILITY':sign,'MET_BASELINE_INSUFFICIENT':'YES' if insufficient else 'NO','SCIENTIFIC_GATE0':gate,'STRONG_PASS_conditions':{'GO_ge_0.05':go>=.05,'GI_le_0':gi<=0,'CI_lower_gt_0':ci[0]>0},'PASS_conditions':{'Gamma_ge_0.03':gamma>=.03,'CI_lower_gt_0':ci[0]>0,'ongoing_condition':(go>=2*gi) if gi>=0 else go>0},'formula':'difference formulation frozen by later R6 section 20'}
    report='# SCIENTIFIC_GATE0_BASELINE_R6\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nAll models used the frozen R55 split, R52 maskable policy, matched feature samples, train-only normalization and loss weights. TEST evaluation occurred after all nine checkpoints and validation thresholds were frozen.\n'
    (ROOT/'SCIENTIFIC_GATE0_R6_REPORT.md').write_text(report,encoding='utf-8')
    for p in [ROOT/'RESEARCH_CONTROL.md',ROOT/'GATE0_DECISION.md']:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n---\n\n# R6 Gate-0 baseline outcome\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('R6_TEST_PHASE_DONE',json.dumps(result),flush=True); return result,out,inter

def final_validation(state,result,elapsed):
    audit={'utc':now(),'free_disk_before_bytes':state['free_disk_before_bytes'],'free_disk_after_bytes':windows_free(),'maximum_cache_size_bytes':cache_bytes(),'bytes_written_estimate':cache_bytes()+sum(p.stat().st_size for p in CKPT.glob('*') if p.is_file()),'cache_path':str(CACHE),'checkpoint_path':str(CKPT),'no_label_store_copy':True,'no_historical_artifact_deletion':True}
    dump(ROOT/'R6_DISK_IO_AUDIT.json',audit)
    val={'utc':now(),'status':'PASS','nine_runs_complete':len(list(CKPT.glob('*.pt')))==9,'test_set_freeze_unchanged':sha(ROOT/'TEST_SET_FREEZE_R55.json')==json.loads((CACHE/'run_state.json').read_text(encoding='utf-8')).get('test_freeze_sha256',sha(ROOT/'TEST_SET_FREEZE_R55.json')),'result':result,'elapsed_seconds':elapsed,'peak_gpu_vram_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,'peak_process_rss_bytes':peak_rss(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    dump(ROOT/'R6_VALIDATION.json',val)

def main():
    t0=time.time(); state=policy_and_state(); manifest(False)
    # Cache construction is restricted to TRAIN/VALIDATION before all fitting and threshold freezing.
    print('R6_START',now(),'cuda',torch.cuda.is_available(),flush=True)
    for d in sorted(x for x in SPLIT if SPLIT[x] in ('TRAIN','VALIDATION')): build_date(d)
    norm=normalization(); man=json.loads((ROOT/'TRAIN_SAMPLE_MANIFEST_R6.json').read_text(encoding='utf-8')); mh=man['all_sample_id_sha256']
    rows=[]
    for kind in ['M_LTG','M_MET','M_FUSION']:
        for seed in SEEDS: rows.append(train_one(kind,seed,norm,mh,state))
    writecsv(ROOT/'BASELINE_TRAINING_RESULTS_R6.csv',rows); print('R6_TRAINING_DONE',now(),flush=True)
    thresholds=validation_thresholds(); print('R6_VALIDATION_THRESHOLDS_FROZEN',thresholds,flush=True); result,_,_=test_and_report(norm,thresholds,rows,state); final_validation(state,result,time.time()-t0); print('R6_COMPLETE',now(),flush=True)
if __name__=='__main__':
    try: main()
    except Exception as e:
        dump(ROOT/'R6_FAILURE.json',{'utc':now(),'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc()})
        dump(ROOT/'R6_VALIDATION.json',{'utc':now(),'status':'BLOCKED','error':repr(e),'test_evaluation_started':json.loads((CACHE/'run_state.json').read_text(encoding='utf-8')).get('test_phase_started',False) if (CACHE/'run_state.json').exists() else False})
        raise
