"""R83 common I/O and exact frozen pure function extraction. No training."""
import ast, collections, csv, ctypes, datetime as dt, hashlib, json, os, shutil, struct, sys, zipfile
from pathlib import Path
import numpy as np
import netCDF4 as nc
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve();OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R83_FROZEN_INFERENCE';DATA=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
sys.path.insert(0,str(OUT/'runtime_dependencies'))
UTC=dt.timezone.utc;SEEDS=[2026091001,2026091002,2026091003];DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
def now():return dt.datetime.now(UTC).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8');os.replace(q,p)
def csvrows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,a):
 a=list(a);fields=list(dict.fromkeys(k for r in a for k in r));p=Path(p);q=p.with_name(p.name+'.tmp')
 with q.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(a)
 os.replace(q,p)
def diskcheck(extra=0):
 free=shutil.disk_usage(OUT).free
 if free-extra<80*2**30:raise RuntimeError('DISK_BELOW_80_GiB_PAUSE_LARGE_CACHE')
 return free
def pure(rel,names,env):
 s=(ROOT/rel).read_text(encoding='utf-8-sig');nodes={n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};registry={}
 for name in names:
  body=ast.get_source_segment(s,nodes[name]);registry[rel+'::'+name]=hashlib.sha256(body.encode()).hexdigest();exec(compile(body,rel+'::'+name,'exec'),env)
 return registry

def mmap_npz(path):
 result={}
 with zipfile.ZipFile(path) as z:
  for info in z.infolist():
   assert info.compress_type==zipfile.ZIP_STORED and info.filename.endswith('.npy')
   with Path(path).open('rb') as f:
    f.seek(info.header_offset);header=f.read(30);nlen,elen=struct.unpack('<HH',header[26:30]);f.seek(nlen+elen,1);version=np.lib.format.read_magic(f)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
    else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
    offset=f.tell();assert not dtype.hasobject
   result[info.filename[:-4]]=np.memmap(path,mode='r',dtype=dtype,shape=shape,offset=offset,order='F' if fortran else 'C')
 return result

def frozen_hashes():
 freeze=read(DATA/'R82_CONFIRMATORY_SET_FREEZE.json');v=read(DATA/'R82_VALIDATION.json');assert v['R82_CONFIRMATORY_DATA_GATE']=='PASS' and v['READY_FOR_R83_FROZEN_CONFIRMATORY_INFERENCE']=='YES' and sha(DATA/'R82_CONFIRMATORY_SET_FREEZE.json')==v['freeze_sha256']
 hashes={}
 pres=read(DATA/'R82_FROZEN_ARTIFACT_FINAL_AUDIT.json')['hashes']
 for rel,expected in pres.items():
  p=Path(rel);p=p if p.is_absolute() else ROOT/rel;h=sha(p);assert h==expected,rel;hashes[str(p)]=h
 for name in ('R82_CONFIRMATORY_SET_FREEZE.json','R82_VALIDATION.json','R82_FEATURE_SCHEMA_AUDIT.json','R82_GLM_TEMPORAL_MASK.csv','R82_TEMPORAL_VALIDITY_SUMMARY.json','R82_LABEL_LEAKAGE_AUDIT.json','R82_QF_SENSITIVITY.json','R82_ONSET_SUFFICIENCY_SUMMARY.json','R82_CACHE_FINAL_VALIDATION.json'):
  hashes[str(DATA/name)]=sha(DATA/name)
 assert hashes[str(DATA/'R82_GLM_TEMPORAL_MASK.csv')]==read(DATA/'R82_TEMPORAL_VALIDITY_SUMMARY.json')['mask_sha256']
 hashes[str(DATA/'R82_CONFIRMATORY_SAMPLE_MANIFEST/manifest.json')]=sha(DATA/'R82_CONFIRMATORY_SAMPLE_MANIFEST/manifest.json');assert hashes[str(DATA/'R82_CONFIRMATORY_SAMPLE_MANIFEST/manifest.json')]==freeze['manifest_sha256']
 for part in freeze['partitions']:
  day=part['date']
  for p,key in [(DATA/'labels_r82'/(day+'.nc'),'label_sha256'),(DATA/'cache_r82/features'/(day+'.npz'),'feature_sha256'),(DATA/'R82_CONFIRMATORY_SAMPLE_MANIFEST'/(day+'.nc'),'manifest_sha256')]:
   h=sha(p);assert h==part[key],str(p);hashes[str(p)]=h
 for r in csvrows(ROOT/'R6_CHECKPOINT_VALIDATION.csv'):
  import re
  match=re.search(r'checkpoint_sha256=([a-fA-F0-9]{64})',r['detail'])
  if match:assert sha(r['checkpoint'])==match.group(1)
 hashes[str(ROOT/'VALIDATION_THRESHOLDS_R6.csv')]=sha(ROOT/'VALIDATION_THRESHOLDS_R6.csv')
 assert freeze['normalization_hash']==sha(ROOT/'TRAIN_NORMALIZATION_R6.json') and freeze['R7_method_hash']==sha(ROOT/'R7_METHOD_FREEZE.json')
 return hashes,freeze

def config():
 method=read(ROOT/'R7_METHOD_FREEZE.json');assert method['R7_FINAL_CANDIDATE']=='STATE_CALIBRATED_FUSION';state=next(x for x in method['candidates'] if x['candidate']=='STATE_CALIBRATED_FUSION');norm=read(ROOT/'TRAIN_NORMALIZATION_R6.json');assert norm['source_split']=='TRAIN_ONLY'
 globalt=next(float(r['threshold']) for r in csvrows(ROOT/'VALIDATION_THRESHOLDS_R6.csv') if r['model']=='M_FUSION' and r['seed']=='ENSEMBLE')
 return norm,state,globalt

def runtime_schema():
 a=read(DATA/'R82_FEATURE_SCHEMA_AUDIT.json');s=a['feature_schema'];h=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==a['feature_schema_sha256']==read(DATA/'R82_CONFIRMATORY_SET_FREEZE.json')['feature_schema_hash']
 assert s['model_input_fields']==['MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30'] and s['history_minutes']==[-25,-20,-15,-10,-5,0] and s['met_shape']==['N',36] and s['ltg_shape']==['N',12] and s['fusion_shape']==['N',48]
 return h,s
