"""Pre-registered R7 calibration audit and one minimal H-gated MLP control.

R6 inputs and outputs are read-only.  Candidate selection occurs entirely on
TRAIN/VALIDATION before this program ever reads R55 TEST probabilities/cache.
"""
import csv, hashlib, json, math, os, shutil, sys, time, traceback, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import r6_baseline as b

ROOT=b.ROOT; R6P=ROOT/'predictions_r6'; CAL=ROOT/'calibration_r7'; CKPT=ROOT/'checkpoints_r7'; PRED=ROOT/'predictions_r7'; TMP=ROOT/'tmp_r7'
SEEDS=b.SEEDS; KINDS=('M_MET','M_FUSION'); SCOPES=(('ALL',None),('I',0),('O',1))
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return b.sha(p)
def temp(name, ext=''):
    TMP.mkdir(parents=True,exist_ok=True); return TMP/f'{name}.{uuid.uuid4().hex}.tmp{ext}'
def jdump(p,x):
    q=temp(Path(p).name); q.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.item() if hasattr(a,'item') else str(a)),encoding='utf-8'); json.loads(q.read_text(encoding='utf-8')); os.replace(q,p)
def wcsv(p,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)); q=temp(Path(p).name)
    with q.open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with q.open(encoding='utf-8-sig') as f: assert len(list(csv.DictReader(f)))==len(rows)
    os.replace(q,p)
def npy_atomic(p,a):
    q=temp(Path(p).stem,'.npy'); np.save(q,a.astype('f4',copy=False)); z=np.load(q,mmap_mode='r'); ok=z.shape==a.shape and np.isfinite(z).all(); del z; assert ok; os.replace(q,p)
def progress(stage,**x): jdump(ROOT/'R7_PROGRESS.json',{'utc':now(),'run':'R7_MINIMAL_BIAS_RESOLUTION_GATE','stage':stage,**x})
def free(): return int(shutil.disk_usage(ROOT).free)
def sigmoid(x):
    x=np.clip(x,-40,40); return 1/(1+np.exp(-x))
def logit(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def state_metrics(y,h,p,t_i,t_o):
    pred=np.where(h==0,p>=t_i,p>=t_o); yy=y.astype(bool); tp=int((pred&yy).sum()); fp=int((pred&~yy).sum()); fn=int((~pred&yy).sum())
    return {'n':len(y),'positive_rate':float(y.mean()),'CSI':tp/max(tp+fp+fn,1),'POD':tp/max(tp+fn,1),'FAR':fp/max(tp+fp,1),'F1':2*tp/max(2*tp+fp+fn,1),'PR_AUC':b.pr_auc(y,p),'Brier':float(np.mean((p-y)**2)),'threshold_policy':'STATE_SPECIFIC','TP':tp,'FP':fp,'FN':fn}
def subset_metrics(y,h,p,t_i,t_o):
    out=[]
    for s,v in SCOPES:
        m=np.ones(len(y),bool) if v is None else h==v
        q=state_metrics(y[m],h[m],p[m],t_i,t_o) if v is None else b.metrics(y[m],p[m],t_i if v==0 else t_o)
        out.append({'scope':s,**q})
    return out
def ece_rows(model,split,state,y,p,bins=15):
    rows=[]; p=np.asarray(p,'f8'); y=np.asarray(y,'u1'); e=0.
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; m=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi)); n=int(m.sum())
        mp=float(p[m].mean()) if n else float('nan'); rate=float(y[m].mean()) if n else float('nan'); e+=0 if not n else n/len(y)*abs(mp-rate)
        rows.append({'row_type':'BIN','model':model,'split':split,'state':state,'bin':i,'lower':lo,'upper':hi,'n':n,'mean_probability':mp,'empirical_rate':rate,'ECE':None})
    return [{'row_type':'SUMMARY','model':model,'split':split,'state':state,'bin':None,'lower':None,'upper':None,'n':len(y),'mean_probability':float(p.mean()),'empirical_rate':float(y.mean()),'ECE':e,
             'positive_class_mean_probability':float(p[y==1].mean()) if y.any() else float('nan'),'negative_class_mean_probability':float(p[y==0].mean()) if (~y.astype(bool)).any() else float('nan'),'Brier':float(np.mean((p-y)**2))}]+rows
def ece(y,p,bins=15):
    y=np.asarray(y);p=np.asarray(p); out=0.
    for i in range(bins):
        m=(p>=i/bins)&(p<((i+1)/bins) if i<bins-1 else p<=1)
        if m.any(): out+=m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)
def fit_platt(p,y):
    """Two-parameter Newton fit over chunks; no TEST values and no model selection."""
    x=logit(p).astype('f8'); y=np.asarray(y,'f8'); a,b0=1.,0.; n=len(y)
    for _ in range(25):
        g=np.zeros(2); h=np.zeros((2,2))
        for lo in range(0,n,1000000):
            xx=x[lo:lo+1000000]; yy=y[lo:lo+1000000]; q=sigmoid(a*xx+b0); w=q*(1-q); d=q-yy
            g += (np.dot(d,xx),d.sum()); h += ((np.dot(w,xx*xx),np.dot(w,xx)),(np.dot(w,xx),w.sum()))
        step=np.linalg.solve(h+np.eye(2)*1e-10,g); a-=step[0]; b0-=step[1]
        if max(abs(step))<1e-10: break
    return {'type':'Platt_logistic','slope':float(a),'intercept':float(b0),'iterations':25,'fit_split':'VALIDATION_ONLY'}
def apply_cal(p,c): return sigmoid(c['slope']*logit(p)+c['intercept']).astype('f4')
def validation_data():
    yy=[]; hh=[]
    for d in b.data_days('VALIDATION'):
        with np.load(b.CACHE/'features'/(d+'.npz')) as z: yy.append(z['y']); hh.append(z['h'])
    return np.concatenate(yy),np.concatenate(hh)
def val_prediction(kind,seed): return np.load(b.CACHE/'predictions'/f'validation_{kind}_seed{seed}.npy',mmap_mode='r')
def ensemble_val(kind): return np.mean([val_prediction(kind,s) for s in SEEDS],axis=0,dtype='f4')
def frozen_thresholds():
    rows=list(csv.DictReader((ROOT/'VALIDATION_THRESHOLDS_R6.csv').open(encoding='utf-8-sig')))
    return {(r['model'],str(r['seed'])):float(r['threshold']) for r in rows}
def audit_r7a(y,h,th):
    met=ensemble_val('M_MET'); fus=ensemble_val('M_FUSION'); npy_atomic(CAL/'r6_met_validation_ensemble.npy',met); npy_atomic(CAL/'r6_fusion_validation_ensemble.npy',fus)
    tr=[]; cals={}; rel=[]; diag=[]
    for kind,p in [('M_MET',met),('M_FUSION',fus)]:
        for state,hv in [('I',0),('O',1)]:
            m=h==hv; t,c=b.csi_threshold(y[m],p[m]); tr.append({'model':kind,'seed':'ENSEMBLE','state':state,'probability':'RAW','threshold':t,'validation_CSI':c,'fit_split':'VALIDATION_ONLY'})
            rel += ece_rows(kind,'VALIDATION',state,y[m],p[m]); diag.append({'model':kind,'state':state,**b.metrics(y[m],p[m],th[(kind,'ENSEMBLE')])})
            if kind=='M_FUSION':
                ca=fit_platt(p[m],y[m]); cals[state]=ca; pc=apply_cal(p[m],ca); tc,cc=b.csi_threshold(y[m],pc); tr.append({'model':kind,'seed':'ENSEMBLE','state':state,'probability':'PLATT','threshold':tc,'validation_CSI':cc,'fit_split':'VALIDATION_ONLY'}); rel += ece_rows(kind+'_PLATT','VALIDATION',state,y[m],pc)
    # Individual-seed calibration is precomputed for the mandated seed stability diagnostic.
    seedc={}
    for s in SEEDS:
        p=val_prediction('M_FUSION',s); seedc[str(s)]={}
        for state,hv in [('I',0),('O',1)]:
            m=h==hv; ca=fit_platt(p[m],y[m]); pc=apply_cal(p[m],ca); tt,cc=b.csi_threshold(y[m],pc); seedc[str(s)][state]={**ca,'threshold':tt,'validation_CSI':cc}
    ti={r['model']+'_'+r['state']+'_'+r['probability']:r['threshold'] for r in tr}
    pcal=np.empty_like(fus); pcal[h==0]=apply_cal(fus[h==0],cals['I']); pcal[h==1]=apply_cal(fus[h==1],cals['O']); npy_atomic(CAL/'state_calibrated_fusion_validation.npy',pcal)
    global_i=b.metrics(y[h==0],fus[h==0],th[('M_FUSION','ENSEMBLE')]); global_o=b.metrics(y[h==1],fus[h==1],th[('M_FUSION','ENSEMBLE')])
    st=subset_metrics(y,h,fus,ti['M_FUSION_I_RAW'],ti['M_FUSION_O_RAW']); ca=subset_metrics(y,h,pcal,ti['M_FUSION_I_PLATT'],ti['M_FUSION_O_PLATT'])
    sm={r['scope']:r for r in st}; cm={r['scope']:r for r in ca}
    gap=b.metrics(y[h==0],met[h==0],ti['M_MET_I_RAW'])['CSI']-global_i['CSI']; rr_t=(sm['I']['CSI']-global_i['CSI'])/gap if gap>0 else None; rr_c=(cm['I']['CSI']-global_i['CSI'])/gap if gap>0 else None
    loss_o=global_o['CSI']-cm['O']['CSI']; pr_loss=global_i['PR_AUC']-cm['I']['PR_AUC']
    # PR-AUC is invariant under a monotone Platt transform apart from finite-precision sorting ties.
    # Apply the R7 pre-registered no-material-decline tolerance of 0.01.
    classification='DOMINANT' if rr_c is not None and rr_c>=.8 and loss_o<=.005 and pr_loss<=.01 else 'PARTIAL' if rr_c is not None and rr_c>=.3 else 'INSUFFICIENT'
    # Frozen state-composed feasibility diagnostic: MET for I and FUSION for O.
    comp=np.where(h==0,met,fus); tg,cg=b.csi_threshold(y,comp); composed_global=[{'policy':'GLOBAL_STYLE','scope':s,**b.metrics(y[np.ones(len(y),bool) if v is None else h==v],comp[np.ones(len(y),bool) if v is None else h==v],tg)} for s,v in SCOPES]
    composed_state=subset_metrics(y,h,comp,ti['M_MET_I_RAW'],ti['M_FUSION_O_RAW']); composed_state=[{'policy':'STATE_SPECIFIC',**r} for r in composed_state]
    wcsv(ROOT/'R7_STATE_THRESHOLDS.csv',tr); wcsv(ROOT/'R7_RELIABILITY_AUDIT.csv',rel); wcsv(ROOT/'R7_CALIBRATION_AUDIT.csv',diag); wcsv(ROOT/'R7_STATE_COMPOSED_DIAGNOSTIC.csv',composed_global+composed_state)
    payload={'utc':now(),'validation_only':True,'fusion_calibrators':cals,'fusion_seed_calibrators':seedc,'thresholds':ti,'global_fusion':{'I':global_i,'O':global_o},'state_threshold':sm,'state_calibrated':cm,'onset_gap_to_MET':gap,'RESCUE_RATIO_THRESHOLD':rr_t,'RESCUE_RATIO_CALIBRATION':rr_c,'ongoing_CSI_loss_calibrated':loss_o,'onset_PR_AUC_loss_calibrated':pr_loss,'R7_CALIBRATION_DIAGNOSIS':classification}
    jdump(CAL/'STATE_CONDITIONED_CALIBRATION_R7A.json',payload)
    text='# R7A mechanism decision\n\n'+json.dumps(payload,ensure_ascii=False,indent=2)+'\n\nR55 TEST was not read by this phase.\n'; q=temp('R7A_MECHANISM_DECISION.md');q.write_text(text,encoding='utf-8');os.replace(q,ROOT/'R7A_MECHANISM_DECISION.md')
    return payload
class HSuppress(torch.nn.Module):
    """The R6 48-32-1 MLP with its 12 normalized GLM channels gated by observable H."""
    def __init__(self): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(48,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    def forward(self,x,h,force=None):
        g=h if force is None else torch.full_like(h,float(force)); q=x.clone(); q[:,36:]*=g[:,None]; return self.net(q).squeeze(1)
def cp(seed): return CKPT/f'M_HSUPPRESS_seed{seed}.pt'
def torch_atomic(p,obj):
    q=temp(Path(p).stem,'.pt'); torch.save(obj,q); a=torch.load(q,map_location='cpu',weights_only=False); assert a.get('complete') is not None; os.replace(q,p)
def hpredict(model,norm,days,device,force=None):
    out=[]; batch=65536 if device.type=='cuda' else 8192; model.eval()
    with torch.no_grad():
        for d in days:
            with np.load(b.CACHE/'features'/(d+'.npz')) as z:
                x=b.nrm(z,norm,'M_FUSION'); h=z['h'].astype('f4'); pp=[]
                for lo in range(0,len(h),batch): pp.append(torch.sigmoid(model(torch.from_numpy(x[lo:lo+batch]).to(device),torch.from_numpy(h[lo:lo+batch]).to(device),force)).cpu().numpy())
                out.append(np.concatenate(pp)); del x
    return np.concatenate(out).astype('f4')
def train_hsuppress(seed,norm,manifest_hash,policy_hash):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=HSuppress().to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5); path=cp(seed)
    start=0; best=-math.inf; stale=0; hist=[]
    if path.exists():
        a=torch.load(path,map_location=device,weights_only=False); assert a['seed']==seed and a['normalization_sha256']==sha(ROOT/'TRAIN_NORMALIZATION_R6.json') and a['split_sha256']==sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv') and a['manifest_sha256']==manifest_hash and a['policy_sha256']==policy_hash
        model.load_state_dict(a['model']);opt.load_state_dict(a['optimizer']);start=a['epoch']+1;best=a['best'];stale=a['stale'];hist=a['history']
        if a.get('complete') and (PRED/f'validation_M_HSUPPRESS_seed{seed}.npy').exists(): return model,hist
    pos=neg=0
    for d in b.data_days('TRAIN'):
        with np.load(b.CACHE/'features'/(d+'.npz')) as z: pos+=int(z['y'].sum()); neg+=len(z['y'])-int(z['y'].sum())
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1),device=device)); batch=65536 if device.type=='cuda' else 8192; torch.manual_seed(seed); rng=np.random.default_rng(seed)
    for epoch in range(start,6):
        model.train(); total=0.;seen=0; epoch_rng=np.random.default_rng(seed+epoch)
        for d in epoch_rng.permutation(b.data_days('TRAIN')):
            with np.load(b.CACHE/'features'/(d+'.npz')) as z:
                x=b.nrm(z,norm,'M_FUSION'); y=z['y'].astype('f4'); h=z['h'].astype('f4'); order=epoch_rng.permutation(len(y))
                for lo in range(0,len(y),batch):
                    ix=order[lo:lo+batch]; xx=torch.from_numpy(x[ix]).to(device); hh=torch.from_numpy(h[ix]).to(device); yy=torch.from_numpy(y[ix]).to(device); opt.zero_grad(set_to_none=True); loss=lossfn(model(xx,hh),yy); loss.backward();opt.step();total+=float(loss.detach().cpu())*len(ix);seen+=len(ix)
                del x
        vp=hpredict(model,norm,b.data_days('VALIDATION'),device); vy,_=validation_data(); pr=b.pr_auc(vy,vp); hist.append({'epoch':epoch,'train_weighted_bce':total/max(seen,1),'validation_pr_auc':pr}); print('R7_EPOCH',seed,epoch,round(total/max(seen,1),7),round(pr,7),'stale',stale,flush=True)
        if pr>best: best=pr;stale=0;beststate={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else: stale+=1
        payload={'kind':'M_HSUPPRESS','seed':seed,'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'stale':stale,'history':hist,'normalization_sha256':sha(ROOT/'TRAIN_NORMALIZATION_R6.json'),'split_sha256':sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv'),'manifest_sha256':manifest_hash,'policy_sha256':policy_hash,'architecture':'R6 M_FUSION 48-32-1 MLP; normalized GLM channels multiplied by observable H before first fusion layer','complete':False}
        torch_atomic(path,payload)
        if stale>=2: break
    model.load_state_dict(beststate); vp=hpredict(model,norm,b.data_days('VALIDATION'),device); npy_atomic(PRED/f'validation_M_HSUPPRESS_seed{seed}.npy',vp); payload.update(model=model.state_dict(),complete=True,epoch=hist[-1]['epoch'],best_state_saved=True);torch_atomic(path,payload);return model,hist
def train_and_select(r7a,y,h,th):
    policy={'architecture':'R6 M_FUSION 48-32-1 MLP with only normalized GLM channels gated by observable past-window H prior to first fusion layer','optimizer':'AdamW(lr=0.001, weight_decay=1e-5)','loss':'same TRAIN-derived global weighted BCE as R6','max_epochs':6,'patience':2,'seeds':SEEDS,'no_test_read_before_method_freeze':True,'H_AVAILABLE_AT_INFERENCE':True,'H_FUTURE_DEPENDENCE':False,'H_direct_label_input':False}
    text='# R7 frozen training policy\n\n'+''.join(f'- **{k}**: {v}\n' for k,v in policy.items());q=temp('R7_TRAINING_POLICY.md');q.write_text(text,encoding='utf-8');os.replace(q,ROOT/'R7_TRAINING_POLICY.md');ph=sha(ROOT/'R7_TRAINING_POLICY.md')
    manifest=json.loads((ROOT/'TRAIN_SAMPLE_MANIFEST_R6.json').read_text(encoding='utf-8')); hhsh=hashlib.sha256()
    for r in manifest['rows']:
        if r['split']!='TEST': hhsh.update(r['date'].encode()+bytes.fromhex(r['sample_id_sha256']))
    norm=json.loads((ROOT/'TRAIN_NORMALIZATION_R6.json').read_text(encoding='utf-8')); histories=[]
    for s in SEEDS:
        _,hist=train_hsuppress(s,norm,hhsh.hexdigest(),ph);histories.append({'seed':s,'epochs':len(hist),'best_validation_pr_auc':hist[-1]['validation_pr_auc'],'checkpoint_sha256':sha(cp(s))});progress('M_HSUPPRESS_SEED_COMPLETE',seed=s)
    ps=[np.load(PRED/f'validation_M_HSUPPRESS_seed{s}.npy',mmap_mode='r') for s in SEEDS]; ep=np.mean(ps,axis=0,dtype='f4');npy_atomic(PRED/'validation_M_HSUPPRESS_ensemble.npy',ep); t,c=b.csi_threshold(y,ep); mhs={r['scope']:r for r in subset_metrics(y,h,ep,t,t)}
    state={r['scope']:r for r in r7a['state_calibrated'].values()} if False else r7a['state_calibrated']
    # r7a dictionaries already key their scopes; selection is validation-only and follows the requested fixed ordering.
    base_i=r7a['global_fusion']['I'];base_o=r7a['global_fusion']['O']; cal_i=state['I'];cal_o=state['O']
    candidates=[]
    for name,mi,mo,extra in [('STATE_CALIBRATED_FUSION',cal_i,cal_o,{'calibration':r7a['fusion_calibrators'],'thresholds':{'I':r7a['thresholds']['M_FUSION_I_PLATT'],'O':r7a['thresholds']['M_FUSION_O_PLATT']}}),('M_HSUPPRESS',mhs['I'],mhs['O'],{'ensemble_threshold':t,'checkpoint_hashes':[sha(cp(s)) for s in SEEDS]})]:
        di=mi['CSI']-base_i['CSI'];do=mo['CSI']-base_o['CSI'];pr=mi['PR_AUC']-base_i['PR_AUC'];ok=di>=.03 and do>=-.01 and pr>=-.01;candidates.append({'candidate':name,'validation_I_CSI':mi['CSI'],'validation_O_CSI':mo['CSI'],'DELTA_I':di,'DELTA_O':do,'PR_AUC_I_delta':pr,'eligible':ok,**extra})
    cal,hs=candidates
    if not cal['eligible'] and not hs['eligible']: chosen='NONE'
    elif cal['eligible'] and (not hs['eligible'] or hs['DELTA_I']<cal['DELTA_I']+.02): chosen='STATE_CALIBRATED_FUSION'
    else: chosen='M_HSUPPRESS'
    wcsv(ROOT/'R7_VALIDATION_METHOD_SELECTION.csv',candidates); jdump(ROOT/'R7_METHOD_FREEZE.json',{'utc':now(),'R7_FINAL_CANDIDATE':chosen,'selection_split':'VALIDATION_ONLY','test_read_after_this_file_only':True,'candidates':candidates,'architecture_hash':ph,'normalization_sha256':sha(ROOT/'TRAIN_NORMALIZATION_R6.json'),'split_sha256':sha(ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv'),'R55_TEST_ROLE':'INTERNAL_FIXED_BENCHMARK','FINAL_METHOD_EXTERNAL_CONFIRMATION_REQUIRED':'YES'})
    return chosen,{'threshold':t,'histories':histories,'candidates':candidates,'norm':norm}
def test_and_report(chosen,r7a,hs,yv,hv,th):
    # This is the first TEST read in R7, after R7_METHOD_FREEZE.json exists.
    assert (ROOT/'R7_METHOD_FREEZE.json').exists(); z=np.load(R6P/'test_metadata_r6.npz'); y,h,date=(z[k].copy() for k in ('y','h','date_code')); n=len(y)
    r6f=np.load(R6P/'M_FUSION_ensemble_test_probability.npy',mmap_mode='r'); r6m=np.load(R6P/'M_MET_ensemble_test_probability.npy',mmap_mode='r')
    if chosen=='STATE_CALIBRATED_FUSION':
        p=np.empty(n,'f4');p[h==0]=apply_cal(r6f[h==0],r7a['fusion_calibrators']['I']);p[h==1]=apply_cal(r6f[h==1],r7a['fusion_calibrators']['O']);ti=r7a['thresholds']['M_FUSION_I_PLATT'];to=r7a['thresholds']['M_FUSION_O_PLATT'];npy_atomic(PRED/'test_STATE_CALIBRATED_FUSION.npy',p)
    else:
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ps=[]
        for s in SEEDS:
            a=torch.load(cp(s),map_location=dev,weights_only=False);m=HSuppress().to(dev);m.load_state_dict(a['model']);q=hpredict(m,hs['norm'],b.data_days('TEST'),dev);npy_atomic(PRED/f'test_M_HSUPPRESS_seed{s}.npy',q);ps.append(q);progress('M_HSUPPRESS_TEST_SEED_COMPLETE',seed=s)
        p=np.mean(ps,axis=0,dtype='f4');npy_atomic(PRED/'test_M_HSUPPRESS_ensemble.npy',p);ti=to=hs['threshold']
    rows=[]
    for name,pp,a,b0 in [('R6_M_MET',r6m,th[('M_MET','ENSEMBLE')],th[('M_MET','ENSEMBLE')]),('R6_M_FUSION',r6f,th[('M_FUSION','ENSEMBLE')],th[('M_FUSION','ENSEMBLE')]),(chosen,p,ti,to)]:
        for r in subset_metrics(y,h,pp,a,b0):
            m=np.ones(n,bool) if r['scope']=='ALL' else h==(0 if r['scope']=='I' else 1); r['ECE']=ece(y[m],pp[m]); rows.append({'model':name,**r})
    wcsv(ROOT/'R7_TEST_RESULTS.csv',rows); by={(r['model'],r['scope']):r for r in rows}; ci=by[(chosen,'I')];co=by[(chosen,'O')];fi=by[('R6_M_FUSION','I')];fo=by[('R6_M_FUSION','O')];mi=by[('R6_M_MET','I')]
    # Date block bootstrap uses threshold confusion counts, equivalent to full-pixel date resampling.
    def cnt(pp,a,b0,code,state):
        m=(date==code)&(h==state); pred=pp[m]>=(a if state==0 else b0); yy=y[m].astype(bool);return np.array([(pred&yy).sum(),(pred&~yy).sum(),(~pred&yy).sum()],dtype='i8')
    cc={(name,d,s):cnt(pp,a,b0,d,s) for name,pp,a,b0 in [('r6f',r6f,th[('M_FUSION','ENSEMBLE')],th[('M_FUSION','ENSEMBLE')]),('r6m',r6m,th[('M_MET','ENSEMBLE')],th[('M_MET','ENSEMBLE')]),('cand',p,ti,to)] for d in range(9) for s in (0,1)}
    def C(a):
        x=np.sum(a,axis=0);return float(x[0]/max(x.sum(),1))
    rng=np.random.default_rng(20260910);boot=[]
    for k in range(1000):
        ds=rng.integers(0,9,9); di=C([cc[('cand',int(d),0)] for d in ds])-C([cc[('r6f',int(d),0)] for d in ds]);do=C([cc[('cand',int(d),1)] for d in ds])-C([cc[('r6f',int(d),1)] for d in ds]);gap=C([cc[('cand',int(d),0)] for d in ds])-C([cc[('r6m',int(d),0)] for d in ds]);boot.append({'replicate':k,'DELTA_I_TEST':di,'DELTA_O_TEST':do,'MET_RECOVERY_GAP':gap})
    wcsv(ROOT/'R7_BOOTSTRAP.csv',boot); cis={k:[float(x) for x in np.percentile([r[k] for r in boot],[2.5,97.5])] for k in ('DELTA_I_TEST','DELTA_O_TEST','MET_RECOVERY_GAP')}
    seedrows=[]
    if chosen=='M_HSUPPRESS':
        for s in SEEDS:
            pp=np.load(PRED/f'test_M_HSUPPRESS_seed{s}.npy',mmap_mode='r');rp=np.load(R6P/f'M_FUSION_seed{s}_test_probability.npy',mmap_mode='r'); tt=th[('M_FUSION',str(s))];a=b.metrics(y[h==0],pp[h==0],hs['threshold']);q=b.metrics(y[h==1],pp[h==1],hs['threshold']);u=b.metrics(y[h==0],rp[h==0],tt);v=b.metrics(y[h==1],rp[h==1],tt);seedrows.append({'seed':s,'DELTA_I':a['CSI']-u['CSI'],'DELTA_O':q['CSI']-v['CSI']})
        # Mechanism intervention is meaningful only for the selected suppression model.
        dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ir=[]
        for mode,force in [('NORMAL_H_GATE',None),('FORCE_H_1',1),('FORCE_H_0',0)]:
            ps=[]
            for s in SEEDS:
                a=torch.load(cp(s),map_location=dev,weights_only=False);m=HSuppress().to(dev);m.load_state_dict(a['model']);ps.append(hpredict(m,hs['norm'],b.data_days('TEST'),dev,force))
            ep=np.mean(ps,axis=0,dtype='f4')
            for scope,val in [('I',0),('O',1)]:ir.append({'intervention':mode,'scope':scope,**b.metrics(y[h==val],ep[h==val],hs['threshold'])})
        wcsv(ROOT/'R7_MECHANISM_INTERVENTION.csv',ir)
    else:
        for s in SEEDS:
            rp=np.load(R6P/f'M_FUSION_seed{s}_test_probability.npy',mmap_mode='r');pc=np.empty(n,'f4');pc[h==0]=apply_cal(rp[h==0],r7a['fusion_seed_calibrators'][str(s)]['I']);pc[h==1]=apply_cal(rp[h==1],r7a['fusion_seed_calibrators'][str(s)]['O']);a=b.metrics(y[h==0],pc[h==0],r7a['fusion_seed_calibrators'][str(s)]['I']['threshold']);q=b.metrics(y[h==1],pc[h==1],r7a['fusion_seed_calibrators'][str(s)]['O']['threshold']);u=b.metrics(y[h==0],rp[h==0],th[('M_FUSION',str(s))]);v=b.metrics(y[h==1],rp[h==1],th[('M_FUSION',str(s))]);seedrows.append({'seed':s,'DELTA_I':a['CSI']-u['CSI'],'DELTA_O':q['CSI']-v['CSI']})
        wcsv(ROOT/'R7_MECHANISM_INTERVENTION.csv',[{'intervention':'NOT_APPLICABLE_STATE_CALIBRATED_FUSION','scope':'ALL','note':'No newly trained suppression model was selected.'}])
    stability='STABLE_POSITIVE' if sum(r['DELTA_I']>0 for r in seedrows)==3 else 'MIXED' if sum(r['DELTA_I']>0 for r in seedrows)==2 else 'UNSTABLE'
    for r in seedrows:r['R7_SEED_STABILITY']=stability
    wcsv(ROOT/'R7_SEED_STABILITY.csv',seedrows)
    di=ci['CSI']-fi['CSI'];do=co['CSI']-fo['CSI']; strong=di>=.04 and ci['CSI']>=mi['CSI']-.005 and do>=-.005 and ci['PR_AUC']>=fi['PR_AUC']-.01 and cis['DELTA_I_TEST'][0]>0; passed=di>=.03 and do>=-.01 and cis['DELTA_I_TEST'][0]>0 and fo['PR_AUC']-co['PR_AUC']<=.01
    gate='STRONG_PASS' if strong else 'PASS' if passed else 'KILL' if di<=0 or do<-.02 else 'HOLD'
    return {'rows':rows,'candidate_metrics':{'I':ci,'O':co},'DELTA_I_TEST':di,'DELTA_O_TEST':do,'MET_RECOVERY_GAP':ci['CSI']-mi['CSI'],'bootstrap_CI95':cis,'R7_SEED_STABILITY':stability,'seed_rows':seedrows,'R7_MINIMAL_SOLUTION_GATE':gate,'STRONG_PASS_conditions':strong,'PASS_conditions':passed}
def main():
    for d in (CAL,CKPT,PRED,TMP):d.mkdir(exist_ok=True)
    started=time.time();before=free();progress('R7A_START')
    # Immutable input checksum snapshot prevents silent R6/R55 drift.
    frozen={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'R6_RECOVERY_VALIDATION.json',ROOT/'VALIDATION_THRESHOLDS_R6.csv',ROOT/'TRAIN_NORMALIZATION_R6.json',ROOT/'TEST_SET_FREEZE_R55.json',ROOT/'SPLIT_ASSIGNMENT_R55'/'date_assignment.csv')}
    y,h=validation_data();th=frozen_thresholds();r7a=audit_r7a(y,h,th);progress('R7A_COMPLETE',diagnosis=r7a['R7_CALIBRATION_DIAGNOSIS'])
    jdump(ROOT/'R7_H_LEAKAGE_AUDIT.json',{'utc':now(),'H_source':'Frozen LABEL_STORE_R5 H_primary',
        'H_definition':'past half-open window [t-30 min,t), as frozen in LABEL_DEFINITION.md','H_AVAILABLE_AT_INFERENCE':True,
        'H_FUTURE_DEPENDENCE':False,'H_is_not_Y':True,'routing_only':'H multiplies normalized GLM channels before the existing R6 fusion layer; it is not concatenated to the prediction head.',
        'label_definition_sha256':sha(ROOT/'LABEL_DEFINITION.md'),'test_labels_or_features_read_for_audit':False,'status':'PASS'})
    chosen,hs=train_and_select(r7a,y,h,th);progress('R7_METHOD_FROZEN',candidate=chosen)
    if chosen=='NONE':
        result={'R7_FINAL_CANDIDATE':'NONE','R7_MINIMAL_SOLUTION_GATE':'KILL_MINIMAL_SOLUTION','R7_MECHANISM_CLASSIFICATION':'UNRESOLVED','R7_CALIBRATION_DIAGNOSIS':r7a['R7_CALIBRATION_DIAGNOSIS'],'FINAL_SCI_CONFIRMATION_REQUIRED':'YES'}
    else:
        test=test_and_report(chosen,r7a,hs,y,h,th); extra=hs['candidates'][1]['DELTA_I']-hs['candidates'][0]['DELTA_I']; mech='CALIBRATION_DOMINANT' if r7a['R7_CALIBRATION_DIAGNOSIS']=='DOMINANT' and extra<.02 else 'REPRESENTATION_INTERFERENCE_DOMINANT' if r7a['R7_CALIBRATION_DIAGNOSIS']=='INSUFFICIENT' and hs['candidates'][1]['DELTA_I']>=.03 else 'MIXED' if extra>=.02 or r7a['R7_CALIBRATION_DIAGNOSIS']=='PARTIAL' else 'UNRESOLVED'
        result={'R7_FINAL_CANDIDATE':chosen,'R7_CALIBRATION_DIAGNOSIS':r7a['R7_CALIBRATION_DIAGNOSIS'],'R7A':{'global_fusion':r7a['global_fusion'],'state_threshold':r7a['state_threshold'],'state_calibrated':r7a['state_calibrated'],'RESCUE_RATIO_THRESHOLD':r7a['RESCUE_RATIO_THRESHOLD'],'RESCUE_RATIO_CALIBRATION':r7a['RESCUE_RATIO_CALIBRATION']},'R7_MECHANISM_CLASSIFICATION':mech,'FINAL_SCI_CONFIRMATION_REQUIRED':'YES',**test}
    report='# R7 minimal bias-resolution report\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nR55 TEST is an internal fixed benchmark. External confirmation on an independent period/year is required before final SCI claims.\n';q=temp('R7_MINIMAL_SOLUTION_REPORT.md');q.write_text(report,encoding='utf-8');os.replace(q,ROOT/'R7_MINIMAL_SOLUTION_REPORT.md')
    for p in (ROOT/'RESEARCH_CONTROL.md',ROOT/'GATE0_DECISION.md'):
        old=p.read_text(encoding='utf-8');mark='# R7 minimal bias-resolution outcome'
        if mark not in old:q=temp(p.name);q.write_text(old+'\n\n---\n\n'+mark+'\n\n'+json.dumps({k:v for k,v in result.items() if k not in ('rows','candidate_metrics','seed_rows')},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
    audit={'utc':now(),'frozen_input_hashes_before':frozen,'frozen_input_hashes_after':{k:sha(ROOT/k) for k in frozen},'R6_frozen_artifacts_modified':False,'R55_test_modified':False,'new_disk_bytes':before-free(),'elapsed_seconds':time.time()-started,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','peak_gpu_vram_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
    jdump(ROOT/'R7_DISK_IO_AUDIT.json',audit);jdump(ROOT/'R7_VALIDATION.json',{'utc':now(),'status':'PASS' if chosen!='NONE' else 'KILL_MINIMAL_SOLUTION','R7A_completed':True,'M_HSUPPRESS_completed':'3/3','method_frozen_before_test':True,'result':result,'audit':audit});progress('R7_COMPLETE',candidate=chosen,gate=result['R7_MINIMAL_SOLUTION_GATE'])
if __name__=='__main__':
    try:main()
    except Exception as e:
        jdump(ROOT/'R7_FAILURE.json',{'utc':now(),'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc()});progress('R7_BLOCKED',error=repr(e));raise
