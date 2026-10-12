"""Recover source precision with verbatim R5 decode and R4 content QC.

Old flash_daily partitions are immutable. No MRMS reads/build, labels or models.
Atomic per-object recovery permits resume without discarding unfinished days.
"""
import argparse, ast, concurrent.futures as cf, csv, datetime as dt, hashlib
import http.client, json, os, re, shutil, tempfile, threading, time, traceback
import urllib.parse, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
import netCDF4 as nc

ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
REC=OUT/'glm_precision_recovery'; GRAN=REC/'granules'; DAILY=REC/'flash_daily';TMP=REC/'payload'
UTC=dt.timezone.utc;DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
HOST='noaa-goes19.s3.amazonaws.com';LOCAL=threading.local();NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
QF_WORDS='good_quality_qf degraded_due_to_flash_constituent_events_out_of_time_order_qf degraded_due_to_flash_constituent_event_count_exceeds_threshold_qf degraded_due_to_flash_duration_exceeds_threshold_qf'
def now():return dt.datetime.now(UTC).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def filename_epoch(key,tag):
 text=re.search('_'+tag+r'(\d+)',key).group(1)
 base=dt.datetime.strptime(text[:13],'%Y%j%H%M%S').replace(tzinfo=UTC).timestamp()
 return base+int(text[13:] or '0')/(10**len(text[13:]))
def stamp(t):return dt.datetime.fromtimestamp(t,UTC).isoformat()
def frozen_function(relative,name,env):
 source=(ROOT/relative).read_text(encoding='utf-8-sig');tree=ast.parse(source)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 text=ast.get_source_segment(source,node)
 exec(compile(text,str(ROOT/relative)+'::'+name,'exec'),env)
 return env[name]
def install_functions():
 global GLM_QC,R5_DECODE
 env={'UTC':UTC,'dt':dt,'Counter':Counter,'isotime':lambda s:dt.datetime.fromisoformat(s.replace('Z','+00:00'))}
 GLM_QC=frozen_function('scripts/stream_content_qc.py','glm_qc',env)
 env={'np':np,'Path':Path,'OUT':REC}
 R5_DECODE=frozen_function('scripts/r5_extract_glm.py','decode',env)
def get_source(row):
 key=row['key'];attempts=[];stem=hashlib.sha256(key.encode()).hexdigest();path=TMP/(stem+'.nc')
 for i in range(1,6):
  start=time.monotonic();rec={'utc':now(),'attempt':i};conn=getattr(LOCAL,'conn',None)
  try:
   if conn is None:conn=http.client.HTTPSConnection(HOST,timeout=30);LOCAL.conn=conn
   conn.request('GET','/'+urllib.parse.quote(key,safe='/'),headers={'User-Agent':'J1-R82-Frozen-R5-PrecisionRecovery/1.0'})
   r=conn.getresponse();rec.update(http_status=r.status,etag=r.getheader('ETag'));assert r.status==200
   h=hashlib.sha256();n=0
   with path.open('wb') as f:
    while True:
     if time.monotonic()-start>180:raise TimeoutError('TRANSPORT_TOTAL_DEADLINE_180_SECONDS')
     b=r.read(65536)
     if not b:break
     f.write(b);h.update(b);n+=len(b)
   assert n==int(row['bytes']),'OFFICIAL_SOURCE_SIZE_MISMATCH'
   if row.get('etag'):assert rec['etag']==row['etag'],'OFFICIAL_SOURCE_ETAG_MISMATCH'
   rec.update(bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start);attempts.append(rec)
   return row,str(path),attempts
  except Exception as e:
   rec.update(error=repr(e),seconds=time.monotonic()-start);attempts.append(rec)
   if conn:conn.close()
   LOCAL.conn=None
   if i<5:time.sleep(min(30,2**i))
 path.unlink(missing_ok=True)
 raise RuntimeError('GLM_OFFICIAL_DOWNLOAD_EXHAUSTED:'+key+':'+repr(attempts))
def list_day(day):
 path=REC/'inventory'/(day+'.json')
 if path.exists():return read(path)['objects']
 date=dt.date.fromisoformat(day);prefix=f'GLM-L2-LCFA/2025/{date.timetuple().tm_yday:03d}/';token=None;objects=[]
 while True:
  q={'list-type':'2','prefix':prefix,'max-keys':1000}
  if token:q['continuation-token']=token
  url='/?'+urllib.parse.urlencode(q);last=None
  for attempt in range(5):
   conn=http.client.HTTPSConnection(HOST,timeout=30)
   try:
    conn.request('GET',url);response=conn.getresponse();assert response.status==200
    root=ET.fromstring(response.read());last=None;break
   except Exception as e:last=e;time.sleep(min(30,2**attempt))
   finally:conn.close()
  if last:raise last
  for e in root.findall('s:Contents',NS):
   k=e.findtext('s:Key',namespaces=NS)
   if not k.endswith('.nc'):continue
   objects.append({'key':k,'bytes':int(e.findtext('s:Size',namespaces=NS)),'etag':e.findtext('s:ETag',namespaces=NS),'date':day,'dataset':'GLM-L2-LCFA','start_utc':stamp(filename_epoch(k,'s')),'end_utc':stamp(filename_epoch(k,'e')),'url':'https://'+HOST+'/'+k})
  if root.findtext('s:IsTruncated',default='false',namespaces=NS).lower()!='true':break
  token=root.findtext('s:NextContinuationToken',namespaces=NS);assert token
 objects.sort(key=lambda x:(x['start_utc'],x['key']))
 with (ROOT/'R8_EXTERNAL_CONFIRMATION/R81_PLATFORM_CONTINUITY/R81_GOES19_GLM_AVAILABILITY.csv').open(encoding='utf-8-sig',newline='') as f:
  old=next(r for r in csv.DictReader(f) if r['date']==day)
 assert len(objects)==int(old['object_count']) and sum(r['bytes'] for r in objects)==int(old['listed_bytes']),'R81_OFFICIAL_INVENTORY_CHANGED'
 dump(path,{'utc':now(),'date':day,'objects':objects,'R81_count_and_bytes_match':True});return objects
def paths(row):
 folder=GRAN/row['date'];folder.mkdir(exist_ok=True)
 stem=hashlib.sha256(row['key'].encode()).hexdigest();return folder/(stem+'.npz'),folder/(stem+'.json')
def verified(row):
 p,j=paths(row)
 if not p.exists() or not j.exists():return False
 try:
  m=read(j)
  assert m['key']==row['key'] and m['source_bytes']==row['bytes'] and m['npz_sha256']==sha(p)
  assert m['R5_decode_script_sha256']==sha(ROOT/'scripts/r5_extract_glm.py') and m['R4_qc_script_sha256']==sha(ROOT/'scripts/stream_content_qc.py')
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)=={'time','lat','lon','qf','flash_id'}
   for k in ['time','lat','lon']:assert z[k].dtype==np.dtype('f8')
   assert z['qf'].dtype==np.dtype('i1') and z['flash_id'].dtype==np.dtype('i8')
   assert all(z[k].shape==(m['support_flash_count'],) for k in z.files)
  return True
 except Exception:return False
def decode(item):
 row,path,attempts=item;body=Path(path).read_bytes()
 try:
  qc=GLM_QC(row,body);qc.pop('roi_flash_counts_by_epoch_bin',None);qc.pop('valid_for_observation',None)
  if any(x.startswith('MISSING_FIELDS:') or x.startswith('UNKNOWN_QUALITY_FLAG:') for x in qc.get('issues',[])):
   raise RuntimeError('BLOCKED_SCHEMA_CHANGE:'+repr(qc))
  with nc.Dataset('schema_check.nc',memory=body) as z:
   q=z['flash_quality_flag']
   if not np.array_equal(q.flag_values,[0,1,3,5]) or q.flag_meanings!=QF_WORDS:
    raise RuntimeError('BLOCKED_SCHEMA_CHANGE:G19_QF_SEMANTICS_CHANGED:'+row['key'])
  result=R5_DECODE(({'source':row},path,attempts));assert result['readable']
  p,j=paths(row);tmp=p.with_name(p.name+'.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,**{k:result[k] for k in ['time','lat','lon','qf','flash_id']})
  os.replace(tmp,p)
  a=epoch(qc['coverage_start']);b=epoch(qc['coverage_end']);issues=qc.get('issues',[])
  structurally_valid=qc['readable'] and b>a and 'CONTENT_FILENAME_TIME_MISMATCH' not in issues
  nonblocking=structurally_valid and qc['coverage_seconds']==20 and all(x=='DEGRADED_FLASH_IN_SUPPORT' for x in issues)
  offsets=[qc.get(k,{}).get('add_offset',0) for k in ['event_time_offset','group_time_offset','flash_time_offset_of_first_event']]
  uncertain=None if nonblocking else {'start':a+min([0]+offsets),'end':b,'key':row['key'],'reason':'FROZEN_R45_CONTENT_UNCERTAINTY_NOT_QF_OUTAGE'}
  meta={'utc':now(),'key':row['key'],'date':row['date'],'source_bytes':len(body),'raw_source_sha256':hashlib.sha256(body).hexdigest(),'npz_sha256':sha(p),'npz_bytes':p.stat().st_size,'support_flash_count':result['support_flash_count'],'full_flash_count':result['full_flash_count'],'qf_counts':result['qf_counts'],'R5_decode_script_sha256':sha(ROOT/'scripts/r5_extract_glm.py'),'R4_qc_script_sha256':sha(ROOT/'scripts/stream_content_qc.py'),'R4_QC':qc,'observation_usable':structurally_valid,'uncertain_interval':uncertain,'download_attempts':attempts}
  dump(j,meta)
  return {'key':row['key'],'support_flash_count':result['support_flash_count'],'download_bytes':len(body),'uncertain':uncertain is not None}
 finally:Path(path).unlink(missing_ok=True)
def publish_day(day,objects):
 arrays={k:[] for k in ['time','lat','lon','qf','flash_id']};records=[];h=hashlib.sha256()
 for row in objects:
  p,j=paths(row);meta=read(j);assert meta['npz_sha256']==sha(p)
  records.append(meta)
  with np.load(p,allow_pickle=False) as z:
   for k in arrays:arrays[k].append(z[k])
  h.update((row['key']+' '+meta['raw_source_sha256']+'\n').encode())
 arrays={k:np.concatenate(v) for k,v in arrays.items()};order=np.argsort(arrays['time'],kind='stable');arrays={k:v[order] for k,v in arrays.items()}
 dest=DAILY/(day+'.npz');q=dest.with_suffix('.tmp.npz');np.savez_compressed(q,**arrays);os.replace(q,dest)
 timeline=REC/'timeline'/(day+'.json');dump(timeline,{'date':day,'records':records})
 m={'utc':now(),'date':day,'status':'PASS','source_files':len(objects),'support_flash_count':len(arrays['time']),'support_qf_counts':{str(k):int((arrays['qf']==k).sum()) for k in [0,1,3,5]},'fractional_timestamps':int(np.count_nonzero(arrays['time']%1)),'npz_sha256':sha(dest),'timeline_sha256':sha(timeline),'source_manifest_sha256':h.hexdigest(),'dtype_schema':{k:str(v.dtype) for k,v in arrays.items()},'precision_matches_frozen_R5':True,'old_R82_cache_preserved':True}
 dump(DAILY/(day+'.json'),m);return m
def preflight():
 install_functions()
 sample=next((ROOT/'R8_EXTERNAL_CONFIRMATION/R81_PLATFORM_CONTINUITY/schema_samples').glob('G19_2025-06-01*'))
 row={'key':'GLM-L2-LCFA/2025/152/18/'+sample.name.split('G19_2025-06-01_',1)[1],'date':'2025-06-01','start_utc':stamp(filename_epoch(sample.name,'s')),'end_utc':stamp(filename_epoch(sample.name,'e')),'bytes':sample.stat().st_size}
 p=TMP/'preflight.nc';shutil.copyfile(sample,p)
 r=decode((row,str(p),[{'sha256':sha(sample),'bytes':sample.stat().st_size,'utc':now(),'local_R81_sample':True}]))
 meta=read(paths(row)[1])
 print(json.dumps({'preflight':r,'R4_QC_issues':meta['R4_QC']['issues'],'precision_match':True}),flush=True)
def main(workers):
 install_functions();completed=0;began=time.monotonic();downloaded=0
 for day in DAYS:
  dest=DAILY/(day+'.npz');m=DAILY/(day+'.json');timeline=REC/'timeline'/(day+'.json')
  if dest.exists() and m.exists() and timeline.exists():
   meta=read(m)
   if meta['npz_sha256']==sha(dest) and meta['timeline_sha256']==sha(timeline):completed+=1;continue
  objects=list_day(day);pending=[r for r in objects if not verified(r)];done=len(objects)-len(pending)
  def progress(current=None):
   state={'utc':now(),'phase':'GLM_FROZEN_R5_PRECISION_RECOVERY','status':'RUNNING','pid':os.getpid(),'date':day,'dates_completed':completed,'dates_required':61,'objects_completed':done,'objects_required':len(objects),'current_object_key':current,'download_bytes_this_run':downloaded,'network_workers':workers,'decode_workers':2,'elapsed_seconds':time.monotonic()-began,'old_flash_daily_preserved':True,'mrms_rebuilt':False,'labels_constructed':False,'models_loaded':False}
   dump(OUT/'R82_GLM_PRECISION_RECOVERY_PROGRESS.json',state);dump(OUT/'R82_PROGRESS.json',state)
  progress();iterator=iter(pending);pn={};pc={};end=False;last=time.monotonic()
  with cf.ThreadPoolExecutor(max_workers=workers) as net,cf.ProcessPoolExecutor(max_workers=2,initializer=install_functions) as cpu:
   while not end or pn or pc:
    while not end and len(pn)<workers:
     try:r=next(iterator)
     except StopIteration:end=True;break
     pn[net.submit(get_source,r)]=r
    for f in [f for f in pn if f.done()][:max(0,4-len(pc))]:
     r=pn.pop(f);pc[cpu.submit(decode,f.result())]=r
    current=None
    for f in [f for f in pc if f.done()]:
     r=pc.pop(f);result=f.result();done+=1;downloaded+=result['download_bytes'];current=r['key']
    if time.monotonic()-last>=20 or (end and not pn and not pc):
     progress(current);last=time.monotonic();print('GLM_PRECISION',day,done,len(objects),flush=True)
    time.sleep(.02)
  assert done==len(objects);meta=publish_day(day,objects);completed+=1;progress()
  print('GLM_PRECISION_DAY_PASS',day,meta['support_flash_count'],flush=True)
 assert completed==61
 dump(OUT/'R82_GLM_PRECISION_RECOVERY_COMPLETE.json',{'utc':now(),'status':'PASS','dates':61,'flash_daily':str(DAILY),'timeline':str(REC/'timeline'),'R5_precision_restored':True,'old_R82_flash_daily_preserved':True,'download_bytes_this_run':downloaded,'no_method_or_label_definition_change':True,'models_loaded':False})
if __name__=='__main__':
 for p in [REC,GRAN,DAILY,TMP,REC/'inventory',REC/'timeline']:p.mkdir(parents=True,exist_ok=True)
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 try:
  if a.preflight:preflight()
  else:main(a.workers)
 except Exception as e:
  status='BLOCKED_SCHEMA_CHANGE' if 'BLOCKED_SCHEMA_CHANGE' in repr(e) else 'BLOCKED_ENGINEERING'
  dump(OUT/('R82_GLM_PRECISION_RECOVERY_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'status':status,'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False,'old_R82_flash_daily_preserved':True});raise
