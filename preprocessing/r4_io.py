"""R4 streaming acquisition and actual content QC; R1-R3 files are read-only."""
import datetime as dt,gzip,hashlib,http.client,json,os,sys,threading,time,urllib.parse
from pathlib import Path
from r4_prepare import ROOT,OUT,UTC,QC,sha,now,epoch
sys.path.insert(0,str(ROOT/'runtime_qc'))
LOCAL=threading.local()
SPECS={QC:(10,0,-99,-999),'MergedReflectivityComposite':(11,1,-99,-999),'Reflectivity_-10C':(3,54,-99,-999),'EchoTop_30':(3,45,-1,-3)}
def download(task):
 s=task['source'];u=urllib.parse.urlsplit(s['url']);attempts=[];path=OUT/'payload'/(hashlib.sha256(s['key'].encode()).hexdigest()+'.payload')
 for i in range(1,4):
  a={'utc':now(),'attempt':i};start=time.monotonic()
  try:
   if not getattr(LOCAL,'connections',None):LOCAL.connections={}
   c=LOCAL.connections.get(u.hostname)
   if c is None:c=http.client.HTTPSConnection(u.hostname,timeout=60);LOCAL.connections[u.hostname]=c
   c.request('GET',u.path,headers={'User-Agent':'J1-R3-FullContentR4/1.0'});r=c.getresponse();a.update(http_status=r.status,etag=r.getheader('ETag'));assert r.status==200
   h=hashlib.sha256();n=0
   with path.open('wb') as f:
    while True:
     b=r.read(65536)
     if not b:break
     f.write(b);h.update(b);n+=len(b)
     if time.monotonic()-start>1800:raise TimeoutError('Payload transfer exceeds 1800s')
   a.update(bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start)
   assert n==int(s['bytes'])
   if s.get('etag'):assert a['etag']==s['etag']
   if task.get('old_sha256'):assert a['sha256']==task['old_sha256'],'ARCHIVE_CONTENT_CHANGED'
   attempts.append(a);return task,str(path),attempts
  except Exception as e:
   a.update(error=repr(e),seconds=time.monotonic()-start);attempts.append(a)
   c=getattr(LOCAL,'connections',{}).pop(u.hostname,None)
   if c:c.close()
   if i<3:time.sleep(2**i)
 return task,None,attempts
def glm(s,body):
 from stream_content_qc import glm_qc
 r=glm_qc(s,body)
 r.pop('roi_flash_counts_by_epoch_bin',None)
 # R1's feature-QF-to-observation rule is explicitly removed in R4.
 r.pop('valid_for_observation',None)
 r['feature_issues']=list(r.get('issues',[]))
 structural=not r.get('readable') or not r.get('coverage_start') or not r.get('coverage_end')
 if not structural:structural=epoch(r['coverage_end'])<=epoch(r['coverage_start'])
 mismatch='CONTENT_FILENAME_TIME_MISMATCH' in r.get('issues',[])
 r['observation_usable']=not structural and not mismatch
 r['observation_status']='OBS_FILE_CORRUPT' if structural else 'OBS_PARTIAL_SOURCE_GAP' if mismatch or r.get('coverage_seconds')!=20 else 'OBS_VALID_COMPLETE'
 q=r.get('flash_quality_counts',{});r.update(nominal_flash_count=q.get('0',0),degraded_flash_count=sum(q.get(str(k),0) for k in [1,3,5]))
 return r
def mrms(s,body):
 import eccodes as ec,numpy as np
 r={'readable':False};h=None
 try:
  raw=gzip.decompress(body);assert raw[:4]==b'GRIB' and raw[-4:]==b'7777' and int.from_bytes(raw[8:16],'big')==len(raw)
  h=ec.codes_new_from_message(raw)
  keys=['edition','centre','discipline','parameterCategory','parameterNumber','gridType','Ni','Nj','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees','iScansNegatively','jScansPositively','jPointsAreConsecutive','alternativeRowScanning','year','month','day','hour','minute','second','missingValue','bitmapPresent']
  m={k:ec.codes_get(h,k) for k in keys};r['metadata']=m
  for k in ['tablesVersion','localTablesVersion','packingType','shapeOfTheEarth']:
   try:m[k]=ec.codes_get(h,k)
   except Exception:m[k]=None
  cat,num,missing,nocov=SPECS[s['dataset']]
  assert (m['discipline'],m['parameterCategory'],m['parameterNumber'])==(209,cat,num)
  stamp=dt.datetime(*(m[k] for k in ['year','month','day','hour','minute','second']),tzinfo=UTC).isoformat();assert stamp==s['start_utc']
  assert m['gridType']=='regular_ll' and not m['jPointsAreConsecutive'] and not m['alternativeRowScanning']
  ni,nj=m['Ni'],m['Nj'];lat=m['latitudeOfFirstGridPointInDegrees']+np.arange(nj)*m['jDirectionIncrementInDegrees']*(1 if m['jScansPositively'] else -1)
  lon=(m['longitudeOfFirstGridPointInDegrees']+np.arange(ni)*m['iDirectionIncrementInDegrees']*(-1 if m['iScansNegatively'] else 1)+180)%360-180
  yi=np.flatnonzero((lat>=31)&(lat<=39));xi=np.flatnonzero((lon>=-100)&(lon<=-92));assert len(yi)==len(xi)==800
  assert lat.min()<=31 and lat.max()>=39 and lon.min()<=-100 and lon.max()>=-92
  v=ec.codes_get_array(h,'values');assert v.size==ni*nj
  v=v.reshape(nj,ni)[np.ix_(yi,xi)].copy();b=np.zeros(v.shape,dtype='u1');b[v==missing]=1;b[v==nocov]=2;b[(~np.isfinite(v))|(v==m['missingValue'])]=3
  p=OUT/'raw_roi'/(hashlib.sha256(s['key'].encode()).hexdigest()+'.npz');np.savez_compressed(p,value_raw=v,base_class=b,lat=lat[yi],lon=lon[xi])
  physical=v[b==0];quant=np.quantile(physical,[0,.01,.5,.99,1]).tolist() if physical.size else [None]*5
  r.update(readable=True,roi_pixels=int(v.size),roi_path=str(p),roi_sha256=sha(p),base_counts={str(i):int((b==i).sum()) for i in range(4)},
   physical_quantiles=dict(zip(['min','p01','median','p99','max'],quant)),negative_physical_pixels=int((physical<0).sum()),
   content_timestamp=stamp,roi_covered=True,full_grib_decoded=True,gzip_crc_valid=True)
  if s['dataset']==QC:
   vals,counts=np.unique(physical,return_counts=True);r['physical_histogram']={str(v):int(n) for v,n in zip(vals,counts)}
 except Exception as e:r['error']=repr(e)
 finally:
  if h is not None:ec.codes_release(h)
 return r
def decode(item):
 task,path,attempts=item;s=task['source'];r={'source':s,'attempts':attempts,'qc_policy_version':'R4','qc_utc':now(),'reuse_version':'NEW_READ','readable':False}
 if path is None:r['error']='DOWNLOAD_FAILED_NOT_SENSOR_ZERO';return r
 try:
  body=Path(path).read_bytes();r.update(glm(s,body) if s['dataset']=='GLM-L2-LCFA' else mrms(s,body));r['raw_source_sha256']=attempts[-1]['sha256']
 except Exception as e:r.update(error=repr(e),readable=False)
 finally:
  # Only this task's known transient payload, never source archives or ROI caches.
  p=Path(path).resolve();assert p.parent==(OUT/'payload').resolve();p.unlink(missing_ok=True)
 return r
