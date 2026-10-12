"""Read-only MRMS cache verification. New reports only; no GRIB download/decode."""
import csv, datetime as dt, hashlib, json, os, shutil, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
CACHE=OUT/'mrms_roi_cache'; CHECK=OUT/'cache_final_validation_daily'
DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
PRODUCTS=('MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def frozen_check():
 mismatches=[];verified=[]
 for r in read(OUT/'R82_PREBUILD_BASELINE.json')['frozen_artifacts']:
  # R8/R81 have already appended authorized control history since this baseline.
  if r['relative_path'] in ['RESEARCH_CONTROL.md','GATE0_DECISION.md']:continue
  p=ROOT/r['relative_path'];s=sha(p);verified.append({'path':str(p),'sha256':s,'matches_prebuild':s==r['sha256']})
  if s!=r['sha256']:mismatches.append(str(p))
 if mismatches:raise RuntimeError('FROZEN_ARTIFACT_HASH_MISMATCH:'+repr(mismatches))
 return verified
def verify(item):
 row,expected=item;key=row['key'];stem=hashlib.sha256(key.encode()).hexdigest()
 p=CACHE/(stem+'.npz');j=CACHE/(stem+'.json')
 result={'key':key,'date':row['date'],'dataset':row['dataset'],'start_utc':row['start_utc'],'path':str(p)}
 if not p.exists() or not j.exists():return {**result,'status':'MISSING','error':'atomic NPZ/JSON pair absent'}
 try:
  m=read(j);s=sha(p)
  for k in ['key','dataset','date','start_utc']:assert m[k]==row[k],k
  assert m['source_bytes']==int(row['bytes']) and m['cache_schema']=='R82_ROI_RAW_BASE_V1'
  assert s==m['npz_sha256'] and m['roi_sha256']==expected['roi_sha256']
  assert m['base_counts']==expected['base_counts']
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)=={'value_raw','base_class','lat','lon'},'fields'
   v=z['value_raw'];b=z['base_class'];lat=z['lat'];lon=z['lon']
   assert v.dtype==np.dtype('f8') and b.dtype==np.dtype('u1'),'value/mask dtype'
   assert v.shape==b.shape==(800,800) and lat.shape==lon.shape==(800,),'ROI dimensions'
   assert lat.dtype==lon.dtype==np.dtype('f8'),'coordinates dtype'
   assert np.isfinite(lat).all() and np.isfinite(lon).all()
   # Native north-to-south, west-to-east grid, identical frozen 0.01-degree centers.
   assert np.allclose(lat,38.995-np.arange(800)*.01,rtol=0,atol=1e-10),'latitude geometry'
   assert np.allclose(lon,-99.995+np.arange(800)*.01,rtol=0,atol=1e-10),'longitude geometry'
   assert hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==m['roi_sha256'],'decoded ROI SHA'
   counts={str(i):int(np.count_nonzero(b==i)) for i in range(4)}
   assert counts==m['base_counts'] and sum(counts.values())==640000,'base classes'
   missing,nocov=(-1,-3) if row['dataset']=='EchoTop_30' else (-99,-999)
   assert np.all(v[b==1]==missing) and np.all(v[b==2]==nocov),'sentinel classes'
  assert dt.datetime.fromisoformat(row['start_utc']).date().isoformat()==row['date']
  return {**result,'status':'PASS','npz_bytes':p.stat().st_size,'npz_sha256':s,'metadata_bytes':j.stat().st_size,'metadata_sha256':sha(j),'roi_sha256':m['roi_sha256'],'source_bytes':m['source_bytes']}
 except Exception as e:return {**result,'status':'MISMATCH','error':repr(e)}
def main():
 CHECK.mkdir(exist_ok=True);baseline=frozen_check()
 if (OUT/'R82_CACHE_FINAL_VALIDATION.json').exists():
  if read(OUT/'R82_CACHE_FINAL_VALIDATION.json').get('status')=='PASS':return
 inventory={}
 with (OUT/'R82_MRMS_OBJECT_INVENTORY.csv').open(encoding='utf-8-sig',newline='') as f:
  inventory={r['key']:r for r in csv.DictReader(f)}
 selected={};daily={}
 for day in DAYS:
  report=read(OUT/'mrms_content_qc'/(day+'.json'))
  assert report['status']=='PASS' and report['max_mrms_age_seconds']==180 and report['no_future_source'] is True
  rec={}
  for r in report['records']:
   row=inventory[r['key']];expected={'roi_sha256':r['roi_sha256'],'base_counts':r['base_counts']}
   assert row['dataset']==r['dataset'] and row['start_utc']==r['start_utc'] and int(row['bytes'])==r['bytes']
   item=(row,expected)
   if r['key'] in selected:assert selected[r['key']]==item
   selected[r['key']]=item;rec[r['key']]=item
  daily[day]=rec
 assert len(selected)==52500
 summaries=[];errors=[];seen=set();free_before=shutil.disk_usage(OUT).free
 for day in DAYS:
  items=[v for k,v in sorted(daily[day].items()) if k not in seen];seen.update(daily[day])
  dest=CHECK/(day+'.csv');meta=CHECK/(day+'.json')
  keyhash=hashlib.sha256('\n'.join(v[0]['key'] for v in items).encode()).hexdigest()
  if dest.exists() and meta.exists() and read(meta).get('status')=='PASS' and read(meta)['key_manifest_sha256']==keyhash and read(meta)['manifest_sha256']==sha(dest):
   summary=read(meta)
  else:
   with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,items))
   bad=[r for r in rows if r['status']!='PASS'];errors+=bad
   fields=list(dict.fromkeys(k for r in rows for k in r));tmp=dest.with_suffix('.tmp.csv')
   with tmp.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
   os.replace(tmp,dest)
   summary={'utc':now(),'date':day,'status':'PASS' if not bad else 'FAIL','objects':len(rows),'missing':sum(r['status']=='MISSING' for r in rows),'schema_mismatch':sum(r['status']=='MISMATCH' for r in rows),'products':dict(Counter(r['dataset'] for r in rows)),'npz_bytes':sum(r.get('npz_bytes',0) for r in rows),'manifest_sha256':sha(dest),'key_manifest_sha256':keyhash}
   dump(meta,summary)
  summaries.append(summary)
  dump(OUT/'R82_CACHE_FINAL_VALIDATION_PROGRESS.json',{'utc':now(),'phase':'CACHE_FINAL_READ_ONLY_VALIDATION','dates_completed':len(summaries),'dates_required':61,'objects_verified':sum(s['objects'] for s in summaries if s['status']=='PASS'),'objects_required':52500,'labels_constructed':False,'models_loaded':False})
  print('CACHE_FINAL',day,summary['status'],summary['objects'],flush=True)
 missing=sum(s['missing'] for s in summaries);mismatch=sum(s['schema_mismatch'] for s in summaries)
 result={'utc':now(),'status':'PASS' if not missing and not mismatch else 'FAIL','CACHE_OBJECT_MISSING':missing,'CACHE_SCHEMA_MISMATCH':mismatch,'required_objects':52500,'verified_objects':sum(s['objects'] for s in summaries if s['status']=='PASS'),'date_count':61,'products':list(PRODUCTS),'checks':['NPZ SHA256','metadata size/SHA256','official selected key/date/product/timestamp','decoded ROI SHA256 versus immutable content QC','field names/dtype/shape','exact native ROI coordinates','base class counts/sentinel interpretation'],'cache_writes':0,'cache_rebuilt':False,'content_qc_rerun':False,'partitioned_manifest_directory':str(CHECK),'npz_total_bytes':sum(s['npz_bytes'] for s in summaries),'daily':summaries,'errors':errors,'frozen_artifacts':baseline,'disk_free_before':free_before,'disk_free_after':shutil.disk_usage(OUT).free,'models_loaded':False}
 dump(OUT/'R82_CACHE_FINAL_VALIDATION.json',result)
 if result['status']!='PASS':raise RuntimeError('MRMS_CACHE_FINAL_VALIDATION_FAILED')
if __name__=='__main__':
 try:main()
 except Exception as e:
  dump(OUT/('R82_CACHE_FINAL_VALIDATION_FAILURE_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
