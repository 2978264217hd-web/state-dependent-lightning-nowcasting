"""Resumable, SHA-verified extraction of support-domain GLM flashes for R5."""
import argparse,collections,hashlib,http.client,json,os,sqlite3,sys,threading,time,traceback,urllib.parse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from pathlib import Path
from r5_common import ROOT,OUT,FIXED,now,sha,dump,epoch
sys.path.insert(0,str(ROOT/'runtime_qc'))
import numpy as np
LOCAL=threading.local()
def download(task):
 s=task['source'];u=urllib.parse.urlsplit(s['url']);attempts=[];path=OUT/'payload'/(hashlib.sha256(s['key'].encode()).hexdigest()+'.nc');path.parent.mkdir(exist_ok=True)
 for i in range(1,4):
  a={'utc':now(),'attempt':i};start=time.monotonic()
  try:
   if not getattr(LOCAL,'connections',None):LOCAL.connections={}
   c=LOCAL.connections.get(u.hostname)
   if c is None:c=http.client.HTTPSConnection(u.hostname,timeout=90);LOCAL.connections[u.hostname]=c
   c.request('GET',u.path,headers={'User-Agent':'J1-R5-LabelBuild/1.0'});r=c.getresponse();a['http_status']=r.status;assert r.status==200
   h=hashlib.sha256();n=0
   with path.open('wb') as f:
    while True:
     b=r.read(65536)
     if not b:break
     f.write(b);h.update(b);n+=len(b)
   a.update(bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start);assert n==int(s['bytes']);assert a['sha256']==task['old_sha256'],'R4_SOURCE_SHA_MISMATCH'
   attempts.append(a);return task,str(path),attempts
  except Exception as e:
   a.update(error=repr(e),seconds=time.monotonic()-start);attempts.append(a);c=getattr(LOCAL,'connections',{}).pop(u.hostname,None)
   if c:c.close()
   if i<3:time.sleep(2**i)
 return task,None,attempts
def decode(item):
 task,path,attempts=item;s=task['source'];out={'source':s,'attempts':attempts,'readable':False}
 if path is None:return out
 try:
  import netCDF4 as nc
  body=Path(path).read_bytes()
  with nc.Dataset('memory.nc',memory=body) as f:
   names=['flash_id','flash_lat','flash_lon','flash_time_offset_of_first_event','flash_quality_flag']
   assert all(n in f.variables for n in names)
   a={n:f[n][:] for n in names};n=len(a['flash_id']);assert all(x.shape==(n,) for x in a.values())
   assert not any(np.ma.count_masked(x) for x in a.values())
   lat=np.asarray(a['flash_lat'],dtype='f8');lon=np.asarray(a['flash_lon'],dtype='f8');q=np.asarray(a['flash_quality_flag'],dtype='i1');ids=np.asarray(a['flash_id'],dtype='i8')
   assert np.isfinite(lat).all() and np.isfinite(lon).all() and np.isin(q,[0,1,3,5]).all()
   base=nc.num2date([0],f['flash_time_offset_of_first_event'].units,only_use_cftime_datetimes=False)[0]
   base=base.replace(tzinfo=__import__('datetime').timezone.utc).timestamp();tt=base+np.asarray(a['flash_time_offset_of_first_event'],dtype='f8')
   keep=(lat>=30.7)&(lat<=39.3)&(lon>=-100.4)&(lon<=-91.6)
   out.update(readable=True,time=tt[keep],lat=lat[keep],lon=lon[keep],qf=q[keep],flash_id=ids[keep],full_flash_count=n,support_flash_count=int(keep.sum()),qf_counts={str(k):int((q==k).sum()) for k in [0,1,3,5]},raw_sha256=attempts[-1]['sha256'])
 finally:
  Path(path).unlink(missing_ok=True)
 return out
def tasks(day):
 c=sqlite3.connect('file:'+(ROOT/'recovery_r4/FILE_QC_R4.sqlite').as_posix()+'?mode=ro',uri=True);rows=[]
 for src,txt in c.execute('SELECT i.source,f.result FROM inventory i JOIN files f ON f.key=i.key WHERE i.date=? AND i.dataset=? ORDER BY i.timestamp,i.key',(day,'GLM-L2-LCFA')):
  s=json.loads(src);r=json.loads(txt);raw=r.get('raw_source_sha256') or next((a['sha256'] for a in reversed(r.get('attempts',[])) if a.get('sha256')),None)
  assert r.get('readable') and raw;rows.append({'source':s,'old_sha256':raw})
 c.close();return rows
def extract_day(day,workers):
 dest=OUT/'flash_daily'/(day+'.npz');meta=OUT/'flash_daily'/(day+'.json')
 if dest.exists() and meta.exists() and readmeta(meta).get('npz_sha256')==sha(dest):return readmeta(meta)
 todo=tasks(day);done=[]
 def run(todo,w):
  results=[];it=iter(todo);netp={};cpup={};end=False
  with ThreadPoolExecutor(w) as net,ProcessPoolExecutor(4) as cpu:
   while not end or netp or cpup:
    while not end and len(netp)<w:
     try:t=next(it)
     except StopIteration:end=True;break
     netp[net.submit(download,t)]=t
    for f in [f for f in netp if f.done()][:max(0,8-len(cpup))]:
     t=netp.pop(f);cpup[cpu.submit(decode,f.result())]=t
    for f in [f for f in cpup if f.done()]:
     t=cpup.pop(f)
     try:r=f.result()
     except Exception as e:r={'source':t['source'],'readable':False,'error':repr(e)}
     results.append(r)
    time.sleep(.01)
  return results
 results=run(todo,workers)
 for _ in range(2):
  bad=[{'source':r['source'],'old_sha256':next(t['old_sha256'] for t in todo if t['source']['key']==r['source']['key'])} for r in results if not r.get('readable')]
  if not bad:break
  good=[r for r in results if r.get('readable')];time.sleep(10);results=good+run(bad,8)
 bad=[r for r in results if not r.get('readable')]
 if bad:raise RuntimeError(f'{day} unresolved GLM extraction files: {len(bad)}')
 arrays={n:np.concatenate([r[n] for r in results]) if results else np.array([]) for n in ['time','lat','lon','qf','flash_id']}
 order=np.argsort(arrays['time'],kind='stable');arrays={k:v[order] for k,v in arrays.items()}
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(dest)
 m={'date':day,'source_files':len(todo),'source_files_verified':len(results),'full_flash_count':sum(r['full_flash_count'] for r in results),'support_flash_count':len(arrays['time']),
  'support_qf_counts':{str(k):int((arrays['qf']==k).sum()) for k in [0,1,3,5]},'duplicate_flash_id_time_count':int(len(arrays['time'])-len(np.unique(np.rec.fromarrays([arrays['flash_id'],arrays['time']],names='id,t')))),
  'source_manifest_sha256':__import__('hashlib').sha256('\n'.join(r['source']['key']+' '+r['raw_sha256'] for r in results).encode()).hexdigest(),'npz_sha256':sha(dest),'utc':now()}
 dump(meta,m);return m
def readmeta(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=128);args=ap.parse_args()
 days=[(d.date()).isoformat() for d in (__import__('datetime').datetime(2024,6,1)+__import__('datetime').timedelta(days=i) for i in range(61)) if d.date().isoformat() not in FIXED]
 for i,day in enumerate(days,1):
  m=extract_day(day,args.workers);dump(OUT/'PROGRESS_R5.json',{'utc':now(),'phase':'GLM_FLASH_EXTRACTION','date':day,'dates_complete':i,'dates_required':59,'support_flashes':m['support_flash_count'],'pid':os.getpid(),'no_labels_yet':True});print('R5_GLM_DAY',i,day,m['support_flash_count'],flush=True)
 dump(OUT/'GLM_EXTRACTION_COMPLETE.json',{'utc':now(),'status':'PASS','dates':59,'total_support_flashes':sum(readmeta(p)['support_flash_count'] for p in (OUT/'flash_daily').glob('*.json'))})
if __name__=='__main__':
 try:main()
 except Exception as e:dump(OUT/'FAILURE_R5.json',{'utc':now(),'phase':'GLM_FLASH_EXTRACTION','error':repr(e),'traceback':traceback.format_exc(),'status':'BLOCKED','no_training':True});raise
