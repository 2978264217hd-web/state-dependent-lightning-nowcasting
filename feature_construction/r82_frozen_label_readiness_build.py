"""R82 data-only frozen label/readiness build; no model imports or fitting."""
import argparse, ast, collections, csv, datetime as dt, hashlib, json, math, os, traceback
from collections import deque
from pathlib import Path
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
REC=OUT/'glm_precision_recovery'
UTC=dt.timezone.utc
FIELDS=('MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30')
DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
CLASSES=('I+','I-','O+','O-')
LAT=38.995-np.arange(800)*.01;LON=-99.995+np.arange(800)*.01
SPEC={'time_index':'u2','pixel_index':'u4','candidate':'u1','H_primary':'i1','Y_primary':'i1','class_primary':'i1','label_valid':'u1','H_qf0':'i1','Y_qf0':'i1','class_qf0':'i1','readiness_flags':'u1','history_validity_mask_bits':'u4'}
NORM_SHA='4dbcd95dde8d4f183a33bce5d7d04fa66b0a34b36c0dd6e5e69583f773f30e09'
def now():return dt.datetime.now(UTC).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def stamp(t):return dt.datetime.fromtimestamp(float(t),UTC).isoformat()
def truth(x):return str(x).lower()=='true'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def safe(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x
def dump(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(safe(x),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def install():
 env=dict(np=np,cKDTree=cKDTree,R=6371008.8,math=math,collections=collections,json=json,to_json_safe=safe,OUT=REC,sha=sha,read=read,epoch=epoch,stamp=stamp)
 registry={}
 sources={'scripts/r5_common.py':['xyz','chord'],'scripts/r5_build_labels.py':['near','open_flash','UF','onset_flashes','cluster_events','edge_margin_km'],'scripts/analyze_semantic_pilot_r2.py':['classify','test_semantics'],'scripts/r3_policy.py':['primary_candidate']}
 for rel,names in sources.items():
  source=(ROOT/rel).read_text(encoding='utf-8-sig');nodes={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  for name in names:
   s=ast.get_source_segment(source,nodes[name]);exec(compile(s,rel+'::'+name,'exec'),env)
   registry[rel+'::'+name]=hashlib.sha256(s.encode()).hexdigest()
 return env,registry
F,REGISTRY=install()
xyz=F['xyz'];chord=F['chord'];near=F['near'];open_flash=F['open_flash'];classify=F['classify'];primary_candidate=F['primary_candidate']
GY,GX=np.meshgrid(LAT,LON,indexing='ij');GRIDXYZ=xyz(GY.ravel(),GX.ravel());del GY,GX
def frozen_check():
 base=read(OUT/'R82_PREBUILD_BASELINE.json');checks={}
 for x in base['frozen_artifacts']:
  name=x['relative_path']
  if name in ('RESEARCH_CONTROL.md','GATE0_DECISION.md'):continue
  actual=sha(ROOT/name);assert actual==x['sha256'],name;checks[name]=actual
 assert sha(ROOT/'TRAIN_NORMALIZATION_R6.json')==NORM_SHA
 return checks
def preflight():
 frozen_check();F['test_semantics']();assert F['edge_margin_km']()>20
 # Boundary regression compares exact frozen near queries to independent
 # pairwise great-circle threshold predicates; synthetic points are tests only.
 pixels=np.array([0,320400,639999]);p=GRIDXYZ[pixels]
 fl={'lat':LAT[pixels//800].copy(),'lon':LON[pixels%800].copy(),'time':np.array([-1800.,0.,1800.]),'qf':np.zeros(3,'i1')}
 for t in [0.]:
  hm=(fl['time']>=t-1800)&(fl['time']<=t);ym=(fl['time']>t)&(fl['time']<=t+1800)
  assert not np.any(hm&ym)
  for mask,radius in [(hm,20),(ym,10)]:
   distances=2*6371.0088*np.arcsin(np.clip(np.linalg.norm(p[:,None,:]-xyz(fl['lat'][mask],fl['lon'][mask])[None,:,:],axis=2)/2,0,1))
   assert np.array_equal(near(GRIDXYZ,pixels,fl,mask,radius),np.any(distances<=radius,axis=1))
 # Exact R52 maskable semantics: local ambiguity/no-coverage is retained;
 # only unexplained decode/source-unavailable excludes a candidate row.
 q=np.array([0,1,2,3,4],dtype='u1');z=np.array([0,5,1,2,8],dtype='u1');e=np.array([0,6,7,2,8],dtype='u1')
 unexpected=np.isin(q,[3,4])|np.isin(z,[3,8])|np.isin(e,[3,8])
 assert unexpected.tolist()==[False,False,False,True,True]
 valid=[(q==0)|(q==1),(z==0)|(z==5),(e==0)|(e==6)]
 assert [v.tolist() for v in valid]==[[True,True,False,False,False],[True,True,False,False,False],[True,True,False,False,False]]
 dump(OUT/'R82_LABEL_READINESS_IMPLEMENTATION_PREFLIGHT.json',{'utc':now(),'status':'PASS','frozen_function_hashes':REGISTRY,'script_sha256':sha(__file__),'normalization_sha256':NORM_SHA,'synthetic_regressions_only':True,'models_loaded':False})
def load_roi(key,raw=False):
 directory=OUT/('raw_composite_audit_reference/raw_roi' if raw else 'mrms_roi_cache')
 path=directory/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
 meta=read(path.with_suffix('.json'))
 assert sha(path)==meta['roi_sha256' if raw else 'npz_sha256']
 with np.load(path,allow_pickle=False) as z:
  vals=z['value_raw'];base=z['base_class']
  assert vals.dtype==np.dtype('f8') and base.dtype==np.dtype('u1') and vals.shape==base.shape==(800,800)
  assert np.allclose(z['lat'],LAT,rtol=0,atol=1e-10) and np.allclose(z['lon'],LON,rtol=0,atol=1e-10)
 if raw:assert meta['source']['key']==key and meta['model_input'] is False
 else:assert meta['key']==key
 return vals.ravel(),base.ravel(),meta
def sources(day):
 with (OUT/'temporal_daily'/(day+'.mrms_sources.csv')).open(encoding='utf-8-sig',newline='') as h:
  rows=list(csv.DictReader(h))
 source={p:{} for p in FIELDS}
 for r in rows:source[r['product']][int(r['time_index'])]=r
 raw=read(OUT/'raw_composite_audit_reference/daily'/(day+'.json'))
 assert raw['status']=='PASS' and raw['model_input'] is False
 return source,{int(r['time_index']):r for r in raw['anchors']}
def frame(day,i,t,src,rawsrc,audit):
 vals=[];bases=[];causals=[]
 for field in FIELDS:
  r=src[field][i];good=truth(r['causal_valid']) and truth(r['cache_available'])
  # 00:00 lacks an authorized cached preceding-day object; never consumed by
  # the first same-day label-valid six-frame window, which starts at 00:05.
  if truth(r['causal_valid']) and not good:assert i==0
  if good:
   ts=epoch(r['source_utc']);assert ts<=t and 0<=t-ts<=180 and t-ts==float(r['age_seconds'])
   v,b,m=load_roi(r['source_key']);assert epoch(m['start_utc'])==ts
   audit['model_sources_checked']+=1
  else:v=np.full(640000,np.nan);b=np.full(640000,3,'u1')
  vals.append(v);bases.append(b);causals.append(good)
 rr=rawsrc[i];rawgood=bool(rr['causal_valid'])
 if rawgood:
  ts=epoch(rr['source_time']);assert ts<=t and 0<=t-ts<=180
  c,cb,m=load_roi(rr['source_key'],True);assert epoch(m['content_timestamp'])==ts
  audit['raw_audit_sources_checked']+=1
 else:c=np.full(640000,np.nan);cb=np.full(640000,3,'u1')
 q,z,e=vals;qb,zb,eb=bases;zs,es,cs=classify(z,e,c,zb,eb,cb)
 qb=qb.copy()
 if not causals[0]:qb[:]=4
 if not causals[1]:zs[:]=8
 if not causals[2]:es[:]=8
 valid=((qb==0)|(qb==1),(zs==0)|(zs==5),(es==0)|(es==6))
 unexpected=np.isin(qb,[3,4])|np.isin(zs,[3,8])|np.isin(es,[3,8])
 cand=np.flatnonzero(primary_candidate(q,qb,causals[0])==1).astype('u4')
 # Preserve frozen R6 categorical encoding while separating physical state.
 # No sentinel is relabeled PHYSICAL; invalid values always have mask0.
 sem=(qb,zs,es)
 for fi,(v,b,s,mask) in enumerate(zip(vals,bases,sem,valid)):
  physical=s==0;assert np.all(np.isfinite(v[physical]))
  sentinel=(-1.,-3.) if fi==2 else (-99.,-999.)
  assert not np.any(np.isin(v[physical],sentinel))
  audit['semantic_valid_categorical_pixel_frames'][fi]+=int((mask&~physical).sum())
 return {'values':vals,'valid':valid,'joint':valid[0]&valid[1]&valid[2],'unexpected':unexpected,'causal':all(causals),'candidate':cand}
def counts(c,mask):return {k:int(np.sum(mask&(c==i))) for i,k in enumerate(CLASSES)}
def build_day(day,temporal):
 dest=OUT/'labels_r82'/(day+'.nc');mp=OUT/'label_readiness_daily'/(day+'.json')
 if dest.exists() and mp.exists():
  m=read(mp)
  if m['store_sha256']==sha(dest) and m['temporal_mask_sha256']==sha(OUT/'R82_GLM_TEMPORAL_MASK.csv') and m['frozen_function_hashes']==REGISTRY:return m
  raise RuntimeError('EXISTING_R82_LABEL_CHECKPOINT_FAILED_VERIFICATION:'+day)
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.nc')
 src,rawsrc=sources(day);fl=open_flash(day);day0=int(epoch(day+'T00:00:00+00:00'));times=day0+np.arange(288,dtype='i8')*300
 history=deque();unexpected_count=np.zeros(640000,'u1');row=0
 total={name:collections.Counter() for name in ('LABEL_VALID','MODEL_READY_TEMPORAL','MODEL_READY_MASKABLE','LEGACY_MODEL_READY_FULL')}
 qmatrix=np.zeros((4,4),'i8');audit={'model_sources_checked':0,'raw_audit_sources_checked':0,'semantic_valid_categorical_pixel_frames':[0,0,0],'history_max_le_t':True,'future_min_gt_t':True,'same_flash_both':0,'future_mrms_reads':0,'mask_depends_on_Y':False}
 with nc.Dataset(tmp,'w',format='NETCDF4') as n:
  n.createDimension('sample',None);n.createDimension('time',288);n.createDimension('lat',800);n.createDimension('lon',800)
  for name,data,typ,dim in [('time',times,'i8','time'),('lat',LAT,'f8','lat'),('lon',LON,'f8','lon')]:n.createVariable(name,typ,(dim,))[:]=data
  variables={name:n.createVariable(name,typ,('sample',),zlib=True,complevel=4,chunksizes=(262144,),fill_value=False) for name,typ in SPEC.items()}
  n.title='R82 frozen R5 labels with row-aligned R52 readiness, complete unsampled candidates';n.date=day
  n.history_interval='[t-1800,t]';n.future_interval='(t,t+1800]';n.class_legend='-1 INVALID;0 I+;1 I-;2 O+;3 O-'
  n.readiness_flag_legend='1 LABEL_VALID;2 MODEL_READY_TEMPORAL;4 MODEL_READY_MASKABLE;8 LEGACY_FULL_AUDIT_ONLY'
  n.mask_bits_legend='bits0-5 QC;6-11 Z10;12-17 ET30, chronological t-25..t'
  n.placeholder=0.;n.no_split=1;n.models_loaded=0;n.raw_composite_role='QC_AUDIT_REFERENCE_ONLY'
  n.semantic_encoding_note='Frozen R4/R6 validity includes corroborated no-echo categories; retained raw codes are categorical encodings, not physical measurements.'
  for i,t in enumerate(times):
   current=frame(day,i,t,src,rawsrc,audit);history.append(current);unexpected_count+=current['unexpected']
   if len(history)>6:unexpected_count-=history.popleft()['unexpected']
   p=current['candidate'];size=len(p);tv=temporal[(day,i)];lv=truth(tv['glm_label_temporal_valid'])
   six=len(history)==6;causal6=six and all(x['causal'] for x in history)
   assert causal6==truth(tv['mrms_six_frame_causal_valid']) or i<=5
   maskable=np.full(size,lv and causal6,dtype=bool)&(unexpected_count[p]==0)
   tempready=np.full(size,lv and causal6,dtype=bool)&current['joint'][p]
   legacy=np.full(size,lv and six,dtype=bool)
   if six:
    for h in history:legacy &= h['joint'][p]
   bits=np.zeros(size,'u4')
   if six:
    for b,h in enumerate(history):
     for fi,v in enumerate(h['valid']):bits |= v[p].astype('u4')<<(fi*6+b)
   hp=np.full(size,-1,'i1');yp=hp.copy();hq=hp.copy();yq=hp.copy();cp=hp.copy();cq=hp.copy()
   if lv and size:
    ht=(fl['time']>=t-1800)&(fl['time']<=t);yt=(fl['time']>t)&(fl['time']<=t+1800)
    assert not np.any(ht&yt);audit['same_flash_both']+=int(np.sum(ht&yt))
    audit['history_max_le_t'] &= not ht.any() or fl['time'][ht].max()<=t
    audit['future_min_gt_t'] &= not yt.any() or fl['time'][yt].min()>t
    hp=near(GRIDXYZ,p,fl,ht,20).astype('i1');yp=near(GRIDXYZ,p,fl,yt,10).astype('i1')
    hq=near(GRIDXYZ,p,fl,ht&(fl['qf']==0),20).astype('i1');yq=near(GRIDXYZ,p,fl,yt&(fl['qf']==0),10).astype('i1')
    cp=(2*hp+1-yp).astype('i1');cq=(2*hq+1-yq).astype('i1')
    qmatrix+=np.bincount((cp*4+cq).astype('i4'),minlength=16).reshape(4,4)
   selected={'LABEL_VALID':np.full(size,lv,bool),'MODEL_READY_TEMPORAL':tempready,'MODEL_READY_MASKABLE':maskable,'LEGACY_MODEL_READY_FULL':legacy}
   for state,selected_mask in selected.items():total[state].update(counts(cp,selected_mask))
   readiness=(np.full(size,lv,'u1')|(tempready.astype('u1')<<1)|(maskable.astype('u1')<<2)|(legacy.astype('u1')<<3))
   payload={'time_index':np.full(size,i,'u2'),'pixel_index':p,'candidate':np.ones(size,'u1'),'H_primary':hp,'Y_primary':yp,'class_primary':cp,'label_valid':np.full(size,lv,'u1'),'H_qf0':hq,'Y_qf0':yq,'class_qf0':cq,'readiness_flags':readiness,'history_validity_mask_bits':bits}
   for name,v in payload.items():variables[name][row:row+size]=v
   row+=size
   if i%12==0:
    n.sync();dump(OUT/'R82_LABEL_READINESS_PROGRESS.json',{'utc':now(),'phase':'FROZEN_R5_LABELS_R52_MASKABLE','pid':os.getpid(),'date':day,'anchor_index':i,'candidate_rows_written':row,'dates_completed':DAYS.index(day),'models_loaded':False})
  n.source_float64_flash_sha256=sha(REC/'flash_daily'/(day+'.npz'));n.source_temporal_mask_sha256=sha(OUT/'R82_GLM_TEMPORAL_MASK.csv')
 # Independent recount of actual written rows, formulas and readiness subsets.
 with nc.Dataset(tmp) as n:
  n.set_auto_mask(False);assert len(n.dimensions['sample'])==row
  for start in range(0,row,500000):
   sl=slice(start,min(row,start+500000));cp=n['class_primary'][sl];hp=n['H_primary'][sl];yp=n['Y_primary'][sl];lv=n['label_valid'][sl].astype(bool);rf=n['readiness_flags'][sl]
   assert np.all(cp[~lv]==-1) and np.all(cp[lv]==2*hp[lv]+1-yp[lv])
   assert np.all(((rf&4)!=0)<=lv) and np.all((rf&1)==lv.astype('u1'))
   assert np.all(n['history_validity_mask_bits'][sl] < (1<<18))
 os.replace(tmp,dest)
 meta={'utc':now(),'date':day,'status':'PASS','store_path':str(dest),'store_sha256':sha(dest),'sample_rows':row,'class_counts':total,'qf_transition_matrix':qmatrix.tolist(),'causal_and_endpoint_audit':audit,'frozen_function_hashes':REGISTRY,'temporal_mask_sha256':sha(OUT/'R82_GLM_TEMPORAL_MASK.csv'),'normalization_sha256':NORM_SHA,'models_loaded':False,'no_split':True,'raw_reference_not_model_input':True}
 dump(mp,meta);print('R82_LABEL_DAY',day,row,total['MODEL_READY_MASKABLE'],flush=True);return meta
def main(limit,preflight_only):
 preflight()
 if preflight_only:return
 for name in ['R82_CACHE_FINAL_VALIDATION.json','R82_GLM_PRECISION_RECOVERY_COMPLETE.json','R82_RAW_COMPOSITE_AUDIT_COMPLETE.json','R82_TEMPORAL_VALIDITY_SUMMARY.json']:assert read(OUT/name)['status']=='PASS'
 with (OUT/'R82_GLM_TEMPORAL_MASK.csv').open(encoding='utf-8-sig',newline='') as h:temporal={(r['date'],int(r['time_index'])):r for r in csv.DictReader(h)}
 days=DAYS[:limit] if limit else DAYS;daily=[]
 for day in days:
  daily.append(build_day(day,temporal))
  dump(OUT/'R82_PROGRESS.json',{'utc':now(),'phase':'FROZEN_R5_LABELS_R52_MASKABLE','date':day,'dates_completed':len(daily),'dates_required':61,'pid':os.getpid(),'labels_constructed':True,'models_loaded':False})
 if not limit:
  dump(OUT/'R82_LABEL_READINESS_COMPLETE.json',{'utc':now(),'status':'PASS','dates':61,'sample_rows':sum(m['sample_rows'] for m in daily),'label_directory':str(OUT/'labels_r82'),'metadata_directory':str(OUT/'label_readiness_daily'),'frozen_function_hashes':REGISTRY,'script_sha256':sha(__file__),'models_loaded':False,'feature_build_complete':False,'R82_final_gate':'PENDING'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--limit-days',type=int,default=0);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 try:main(args.limit_days,args.preflight)
 except Exception as e:
  dump(OUT/('R82_LABEL_READINESS_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
