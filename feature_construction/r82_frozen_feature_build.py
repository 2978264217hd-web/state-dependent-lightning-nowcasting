"""R82 raw feature cache using verbatim frozen R6 data-only functions."""
import argparse, ast, csv, datetime as dt, hashlib, json, os, traceback
from pathlib import Path
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
CACHE=OUT/'cache_r82';REC=OUT/'glm_precision_recovery';UTC=dt.timezone.utc
FIELDS=['MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30']
DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
NORM_SHA='4dbcd95dde8d4f183a33bce5d7d04fa66b0a34b36c0dd6e5e69583f773f30e09'
def now():return dt.datetime.now(UTC).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def selected_rows(day,include_y=True):
 lp=OUT/'labels_r82'/(day+'.nc');m=read(OUT/'label_readiness_daily'/(day+'.json'));assert sha(lp)==m['store_sha256']
 with nc.Dataset(lp) as n:
  n.set_auto_mask(False);ti=n['time_index'][:];pi=n['pixel_index'][:];flags=n['readiness_flags'][:]
  take=(flags&4)!=0
  d={'row':np.flatnonzero(take).astype('u4'),'time_index':ti[take].astype('u2'),'pixel_index':pi[take].astype('u4'),'bits':n['history_validity_mask_bits'][:][take].astype('u4'),'lat':n['lat'][:],'lon':n['lon'][:],'time':n['time'][:]}
  if include_y:d.update(y=n['Y_primary'][:][take].astype('u1'),h=n['H_primary'][:][take].astype('u1'),clazz=n['class_primary'][:][take].astype('i1'))
 return d
def source_paths(day,required_indices):
 with (OUT/'temporal_daily'/(day+'.mrms_sources.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 indices=set(required_indices);result={p:{} for p in FIELDS}
 for r in rows:
  i=int(r['time_index'])
  if i not in indices:continue
  assert r['causal_valid'].lower()=='true' and r['cache_available'].lower()=='true' and 0<=float(r['age_seconds'])<=180
  key=r['source_key'];p=OUT/'mrms_roi_cache'/(hashlib.sha256(key.encode()).hexdigest()+'.npz');m=read(p.with_suffix('.json'))
  assert sha(p)==m['npz_sha256'] and m['key']==key and m['start_utc']==r['source_utc']
  result[r['product']][i]=str(p)
 assert all(set(result[p])==indices for p in FIELDS)
 return result
def flash(day):
 p=REC/'flash_daily'/(day+'.npz');assert sha(p)==read(p.with_suffix('.json'))['npz_sha256']
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 order=np.lexsort((a['flash_id'],a['time']));a={k:v[order] for k,v in a.items()}
 pair=np.rec.fromarrays([a['flash_id'],a['time']],names='i,t');keep=np.r_[True,pair[1:]!=pair[:-1]] if len(pair) else np.array([],bool)
 return {k:v[keep] for k,v in a.items()}
def install():
 source=(ROOT/'scripts/r6_baseline.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
 nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 # AST functions only: no torch, models, checkpoints, training, normalization
 # fit, split files, thresholds, performance metrics or interventions imported.
 env=dict(np=np,Path=Path,os=os,cKDTree=cKDTree,CACHE=CACHE,ROOT=ROOT,FIELDS=FIELDS,R_EARTH_KM=6371.0088,
          selected_rows=selected_rows,source_paths=source_paths,flash=flash,
          SPLIT={d:'EXTERNAL_CONFIRMATION_SET' for d in DAYS})
 registry={}
 for name in ['xyz','chord','build_date']:
  body=ast.get_source_segment(source,nodes[name]);exec(compile(body,'r6_baseline.py::'+name,'exec'),env);registry[name]=hashlib.sha256(body.encode()).hexdigest()
 assert 'torch' not in env and 'CKPT' not in env and 'normalization' not in env and 'nrm' not in env
 return env,registry
F,REGISTRY=install()
SCHEMA={'model_input_fields':FIELDS,'history_minutes':[-25,-20,-15,-10,-5,0],'met_shape':['N',36],'ltg_shape':['N',12],'fusion_shape':['N',48],
        'met_channel_order':'18 time-major values, then 18 time-major paired masks','ltg_channel_order':'six chronological pairs log1p(count20km), min(nearest_geodesic_km,100)',
        'value_dtype':'float32','mask_dtype':'float32 binary in met; uint32 bit sidecar','roi_geometry':[800,800],'placeholder_before_normalization':0.0,
        'semantic_mask':'verbatim R4/R52 semantic-valid, including categorical no-echo codes; never classified as physical','normalization_source':'2024_R6_TRAIN_ONLY','normalization_sha256':NORM_SHA}
SCHEMA_SHA=hashlib.sha256(json.dumps(SCHEMA,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run_day(day):
 dest=CACHE/'features'/(day+'.npz');mp=OUT/'feature_daily'/(day+'.json')
 label_meta=read(OUT/'label_readiness_daily'/(day+'.json'))
 if dest.exists():
  if not mp.exists():raise RuntimeError('FEATURE_CACHE_WITHOUT_ATOMIC_VERIFICATION:'+day)
  old=read(mp);assert old['feature_sha256']==sha(dest) and old['source_label_sha256']==label_meta['store_sha256'] and old['feature_schema_sha256']==SCHEMA_SHA
  return old
 # Verbatim R6 build_date core, external date access and ready-row adapters.
 path=F['build_date'](day,False)
 assert path==dest
 norm=read(ROOT/'TRAIN_NORMALIZATION_R6.json')
 mm=np.asarray(norm['met_value_mean'],'f4');ms=np.asarray(norm['met_value_std'],'f4');lm=np.asarray(norm['ltg_mean'],'f4');ls=np.asarray(norm['ltg_std'],'f4')
 assert mm.shape==ms.shape==(18,) and lm.shape==ls.shape==(12,) and np.all(ms>0) and np.all(ls>0)
 q=selected_rows(day,True)
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)=={'met','ltg','y','h','clazz','row','time_index','pixel_index'}
  n=len(z['row']);assert z['met'].shape==(n,36) and z['ltg'].shape==(n,12) and z['met'].dtype==z['ltg'].dtype==np.dtype('f4')
  for name in ['row','time_index','pixel_index']:assert np.array_equal(z[name],q[name])
  for name,key in [('y','y'),('h','h'),('clazz','clazz')]:assert np.array_equal(z[name],q[key])
  met=z['met'];ltg=z['ltg'];masks=met[:,18:]
  assert np.all(np.isin(masks,[0.,1.])) and np.all(met[:,:18][masks==0]==0)
  assert np.all(np.isfinite(met)) and np.all(np.isfinite(ltg)) and np.all(ltg[:,1::2]<=100)
  # Validate actual channel/row alignment against six-frame bits, then apply
  # frozen TRAIN transforms in bounded chunks without retaining duplicates.
  bits=q['bits']
  for b in range(6):
   for fi in range(3):assert np.array_equal(masks[:,b*3+fi],((bits>>(fi*6+b))&1).astype('f4'))
  for start in range(0,n,100000):
   stop=min(n,start+100000)
   nm=(met[start:stop,:18]-mm)/ms;nl=(ltg[start:stop]-lm)/ls
   assert np.all(np.isfinite(nm)) and np.all(np.isfinite(nl))
  clazz=z['clazz'];cc={k:int(np.sum(clazz==i)) for i,k in enumerate(['I+','I-','O+','O-'])}
 assert cc==label_meta['class_counts']['MODEL_READY_MASKABLE']
 meta={'utc':now(),'date':day,'status':'PASS','samples':n,'class_counts':cc,'feature_path':str(path),'feature_sha256':sha(path),'feature_bytes':path.stat().st_size,
       'source_label_sha256':label_meta['store_sha256'],'feature_schema':SCHEMA,'feature_schema_sha256':SCHEMA_SHA,'frozen_r6_function_hashes':REGISTRY,
       'normalization_applied_for_validation_only':True,'normalization_fitted':False,'masks_do_not_depend_on_Y':True,'future_mrms_inputs':False,'models_loaded':False}
 dump(mp,meta);return meta
def main(limit):
 assert sha(ROOT/'TRAIN_NORMALIZATION_R6.json')==NORM_SHA
 assert read(OUT/'R82_LABEL_READINESS_IMPLEMENTATION_PREFLIGHT.json')['status']=='PASS'
 days=DAYS[:limit] if limit else DAYS;daily=[]
 for day in days:
  daily.append(run_day(day))
  dump(OUT/'R82_FEATURE_PROGRESS.json',{'utc':now(),'phase':'FROZEN_R6_FEATURE_CACHE','date':day,'dates_completed':len(daily),'dates_required':61,'pid':os.getpid(),'models_loaded':False,'normalization_fitted':False})
 if not limit:
  dump(OUT/'R82_FEATURE_SCHEMA_AUDIT.json',{'utc':now(),'status':'PASS','dates':61,'feature_schema':SCHEMA,'feature_schema_sha256':SCHEMA_SHA,'frozen_r6_function_hashes':REGISTRY,'samples':sum(x['samples'] for x in daily),
       'actual_partitions_verified':61,'field_channel_time_order_verified':True,'source_timestamp_age_rule':'latest <= frame, age<=180','normalization_fitted':False,'normalization_source':'2024_R6_TRAIN_ONLY','normalization_sha256':NORM_SHA,'models_loaded':False})
  dump(OUT/'R82_FEATURE_CACHE_COMPLETE.json',{'utc':now(),'status':'PASS','dates':61,'samples':sum(x['samples'] for x in daily),'feature_directory':str(CACHE/'features'),'feature_schema_sha256':SCHEMA_SHA,'models_loaded':False})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--limit-days',type=int,default=0);args=ap.parse_args()
 try:main(args.limit_days)
 except Exception as e:
  dump(OUT/('R82_FEATURE_BUILD_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
