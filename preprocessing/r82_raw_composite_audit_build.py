"""User-authorized raw Composite audit reference; never a fourth model input.

Reuse frozen R4 MRMS decode verbatim. Existing three-product QC/cache untouched.
"""
import argparse, ast, bisect, concurrent.futures as cf, csv, datetime as dt
import gzip, hashlib, http.client, json, os, shutil, time, traceback, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
REF=OUT/'raw_composite_audit_reference';ROI=REF/'raw_roi';TMP=REF/'tmp'
PRODUCT='MergedReflectivityComposite';HOST='noaa-mrms-pds.s3.amazonaws.com'
UTC=dt.timezone.utc;DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
def now():return dt.datetime.now(UTC).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def install():
 global DECODE,DOWNLOAD
 work=TMP/('decode_'+str(os.getpid()));(work/'raw_roi').mkdir(parents=True,exist_ok=True)
 env={'dt':dt,'UTC':UTC,'OUT':work,'gzip':gzip,'hashlib':hashlib,'sha':sha,'QC':'MergedReflectivityQCComposite','SPECS':{PRODUCT:(11,1,-99,-999)}}
 source=(ROOT/'scripts/r4_io.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mrms')
 exec(compile(ast.get_source_segment(source,node),str(ROOT/'scripts/r4_io.py')+'::mrms','exec'),env);DECODE=env['mrms']
 env={'http':__import__('http'),'urllib':__import__('urllib'),'time':time,'hashlib':hashlib}
 source=(OUT/'r82_mrms_roi_cache_build.py').read_text(encoding='utf-8-sig');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='download')
 exec(compile(ast.get_source_segment(source,node),str(OUT/'r82_mrms_roi_cache_build.py')+'::download','exec'),env);DOWNLOAD=env['download']
def list_day(day):
 path=REF/'inventory'/(day+'.json')
 if path.exists():return read(path)['objects']
 prefix=f'CONUS/{PRODUCT}_00.50/{day.replace("-","")}/';rows=[];token=None;ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
 while True:
  query={'list-type':'2','prefix':prefix}
  if token:query['continuation-token']=token
  last=None
  for attempt in range(5):
   conn=http.client.HTTPSConnection(HOST,timeout=30)
   try:
    conn.request('GET','/?'+urllib.parse.urlencode(query));r=conn.getresponse();assert r.status==200
    tree=ET.fromstring(r.read());last=None;break
   except Exception as e:last=e;time.sleep(min(30,2**attempt))
   finally:conn.close()
  if last:raise last
  for e in tree.findall('s:Contents',ns):
   key=e.findtext('s:Key',namespaces=ns)
   if not key or not key.endswith('.grib2.gz'):continue
   token_time=key.rsplit('_',1)[1].split('.',1)[0];t=dt.datetime.strptime(token_time,'%Y%m%d-%H%M%S').replace(tzinfo=UTC)
   assert t.date().isoformat()==day
   rows.append({'dataset':PRODUCT,'date':day,'key':key,'url':'https://'+HOST+'/'+key,'bytes':int(e.findtext('s:Size',namespaces=ns)),'start_utc':t.isoformat(),'etag':e.findtext('s:ETag',namespaces=ns)})
  if tree.findtext('s:IsTruncated',namespaces=ns).lower()!='true':break
  token=tree.findtext('s:NextContinuationToken',namespaces=ns);assert token
 rows.sort(key=lambda r:(r['start_utc'],r['key']));dump(path,{'utc':now(),'date':day,'objects':rows});return rows
def paths(row):
 stem=hashlib.sha256(row['key'].encode()).hexdigest();return ROI/(stem+'.npz'),ROI/(stem+'.json')
def verified(row):
 p,j=paths(row)
 if not p.exists() or not j.exists():return False
 try:
  m=read(j);assert m['source']['key']==row['key'] and m['source']['bytes']==row['bytes']
  assert m['roi_sha256']==sha(p) and m['frozen_r4_io_sha256']==sha(ROOT/'scripts/r4_io.py')
  assert m['source']['start_utc']==m['content_timestamp']==row['start_utc']
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)=={'value_raw','base_class','lat','lon'}
   assert z['value_raw'].shape==z['base_class'].shape==(800,800)
   assert z['value_raw'].dtype==np.dtype('f8') and z['base_class'].dtype==np.dtype('u1')
   assert np.allclose(z['lat'],38.995-np.arange(800)*.01,rtol=0,atol=1e-10)
   assert np.allclose(z['lon'],-99.995+np.arange(800)*.01,rtol=0,atol=1e-10)
  return True
 except Exception:return False
def decode(row,body,payload_sha):
 result=DECODE(row,body)
 if not result.get('readable'):raise RuntimeError('RAW_COMPOSITE_FROZEN_R4_QC_FAILED:'+row['key']+':'+repr(result))
 p,j=paths(row);temp=Path(result['roi_path']);assert result['roi_sha256']==sha(temp)
 with np.load(temp,allow_pickle=False) as z:
  assert z['value_raw'].shape==(800,800) and z['value_raw'].dtype==np.dtype('f8')
  assert np.allclose(z['lat'],38.995-np.arange(800)*.01,rtol=0,atol=1e-10)
  assert np.allclose(z['lon'],-99.995+np.arange(800)*.01,rtol=0,atol=1e-10)
 os.replace(temp,p);result['roi_path']=str(p)
 result.update(utc=now(),source=row,source_payload_sha256=payload_sha,source_bytes=len(body),frozen_r4_io_sha256=sha(ROOT/'scripts/r4_io.py'),role='AUDIT_REFERENCE_RAW_COMPOSITE_ONLY',model_input=False)
 dump(j,result);return {'key':row['key'],'download_bytes':len(body)}
def process(day,workers,limit=0):
 inv=list_day(day);times=[epoch(r['start_utc']) for r in inv];start=int(epoch(day+'T00:00:00+00:00'));selected={};anchors=[]
 for i in range(288):
  t=start+i*300;ix=bisect.bisect_right(times,t)-1;row=inv[ix] if ix>=0 else None;age=t-times[ix] if ix>=0 else -1;causal=row is not None and 0<=age<=180
  anchors.append({'time_index':i,'anchor_utc':dt.datetime.fromtimestamp(t,UTC).isoformat(),'source_key':row['key'] if row else '', 'source_time':row['start_utc'] if row else '', 'age_seconds':age,'causal_valid':bool(causal)})
  if causal:selected[row['key']]=row
 rows=list(selected.values());pending=[r for r in rows if not verified(r)];done=len(rows)-len(pending)
 if limit:pending=pending[:limit]
 downloaded=0;iterator=iter(pending);pn={};pc={};end=False;last=0
 with cf.ThreadPoolExecutor(max_workers=workers) as net,cf.ProcessPoolExecutor(max_workers=2,initializer=install) as cpu:
  while not end or pn or pc:
   while not end and len(pn)<workers:
    try:row=next(iterator)
    except StopIteration:end=True;break
    pn[net.submit(DOWNLOAD,row)]=row
   for f in [f for f in pn if f.done()][:max(0,4-len(pc))]:
    row=pn.pop(f);body,raw_sha=f.result();pc[cpu.submit(decode,row,body,raw_sha)]=row
   key=None
   for f in [f for f in pc if f.done()]:
    row=pc.pop(f);r=f.result();done+=1;downloaded+=r['download_bytes'];key=r['key']
   if time.monotonic()-last>=20 or (end and not pn and not pc):
    state={'utc':now(),'phase':'RAW_COMPOSITE_FROZEN_QC_AUDIT_REFERENCE','pid':os.getpid(),'date':day,'objects_verified_in_day':done,'objects_required_in_day':len(rows),'download_bytes_in_day_this_run':downloaded,'current_key':key,'existing_three_product_cache_modified':False,'model_input_products_changed':False,'models_loaded':False}
    dump(OUT/'R82_RAW_COMPOSITE_AUDIT_PROGRESS.json',state);print(json.dumps(state),flush=True);last=time.monotonic()
   time.sleep(.02)
 if limit:return {'date':day,'preflight':'PASS','objects_verified':done,'download_bytes':downloaded}
 assert done==len(rows)
 meta={'utc':now(),'date':day,'status':'PASS','inventory_objects':len(inv),'selected_objects':len(rows),'selected_source_bytes':sum(r['bytes'] for r in rows),'download_bytes_this_run':downloaded,'sources':rows,'anchors':anchors,'model_input':False,'source_rule':'latest source <= anchor, age 0..180 seconds','R4_decode_script_sha256':sha(ROOT/'scripts/r4_io.py')}
 dump(REF/'daily'/(day+'.json'),meta);return meta
def main(workers,preflight):
 assert read(OUT/'R82_RAW_COMPOSITE_AUDIT_AUTHORIZATION.json')['human_authorized'] is True
 install();daily=[]
 for day in DAYS:
  p=REF/'daily'/(day+'.json')
  if p.exists() and read(p).get('status')=='PASS' and all(verified(r) for r in read(p)['sources']):daily.append(read(p));continue
  r=process(day,workers,limit=1 if preflight else 0)
  if preflight:print('RAW_COMPOSITE_PREFLIGHT_PASS',json.dumps(r),flush=True);return
  daily.append(r)
 assert len(daily)==61
 dump(OUT/'R82_RAW_COMPOSITE_AUDIT_COMPLETE.json',{'utc':now(),'status':'PASS','dates':61,'objects':sum(r['selected_objects'] for r in daily),'source_bytes':sum(r['selected_source_bytes'] for r in daily),'role':'AUDIT_REFERENCE_RAW_COMPOSITE_ONLY','model_input_products_changed':False,'frozen_R2_classification_rule_unchanged':True,'frozen_R4_decode_unchanged':True,'model_input_fields':['MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30'],'daily_directory':str(REF/'daily'),'roi_directory':str(ROI),'models_loaded':False})
if __name__=='__main__':
 for p in [REF,ROI,TMP,REF/'inventory',REF/'daily']:p.mkdir(parents=True,exist_ok=True)
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 try:main(a.workers,a.preflight)
 except Exception as e:
  dump(OUT/('R82_RAW_COMPOSITE_AUDIT_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'model_input_products_changed':False,'models_loaded':False});raise
