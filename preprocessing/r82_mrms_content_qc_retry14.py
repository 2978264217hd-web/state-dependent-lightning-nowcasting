"""R82 frozen-method MRMS content-QC stage.

Reads only the official R82 inventory.  For each five-minute anchor it uses
the latest object at or before the anchor with age <=180 seconds, exactly as
R4, and writes atomic per-day QC records.  It deliberately constructs no
labels, samples, normalisation, predictions, or metrics.
"""
import bisect, csv, datetime as dt, gzip, hashlib, http.client, json, os, tempfile, time, traceback, urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
import eccodes as ec
import numpy as np

ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION'/'R82_CONFIRMATORY_DATA_BUILD'; TMP=OUT/'tmp_r82'; INV=OUT/'R82_MRMS_OBJECT_INVENTORY.csv'; QCOUT=OUT/'mrms_content_qc'
UTC=dt.timezone.utc; PRODUCTS=('MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30')
SPECS={'MergedReflectivityQCComposite':(10,0,-99,-999),'Reflectivity_-10C':(3,54,-99,-999),'EchoTop_30':(3,45,-1,-3)}
DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]

def now(): return dt.datetime.now(UTC).isoformat()
def epoch(s): return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def dump(path,obj):
    path=Path(path); TMP.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=TMP)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
        json.loads(Path(tmp).read_text(encoding='utf-8'));path.parent.mkdir(parents=True,exist_ok=True);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def download(row):
    u=urllib.parse.urlsplit(row['url']);last=None
    for i in range(1,6):
        c=None
        try:
            # Bound both connection and socket reads.  This is transport-only:
            # source selection, source bytes, and all R4 data semantics stay
            # unchanged when an official object needs a retry.
            c=http.client.HTTPSConnection(u.hostname,timeout=30);c.request('GET',u.path,headers={'User-Agent':'J1-R3-R82-MRMS-QC/1.0'});r=c.getresponse();parts=[]
            while True:
                block=r.read(65536)
                if not block: break
                parts.append(block)
            body=b''.join(parts);assert r.status==200 and len(body)==int(row['bytes']);return body
        except Exception as e:
            last=repr(e)
            if i<5: time.sleep(min(30,2**i))
        finally:
            if c:c.close()
    raise RuntimeError('MRMS_DOWNLOAD_RETRIES_EXHAUSTED:'+row['key']+':'+last)
def decode(row):
    raw=gzip.decompress(download(row)); assert raw[:4]==b'GRIB' and raw[-4:]==b'7777' and int.from_bytes(raw[8:16],'big')==len(raw)
    h=None
    try:
        h=ec.codes_new_from_message(raw); ds=row['dataset'];cat,num,missing,nocov=SPECS[ds]
        assert (ec.codes_get(h,'discipline'),ec.codes_get(h,'parameterCategory'),ec.codes_get(h,'parameterNumber'))==(209,cat,num)
        ni,nj=ec.codes_get(h,'Ni'),ec.codes_get(h,'Nj'); assert (ni,nj)==(7000,3500)
        lat0=ec.codes_get(h,'latitudeOfFirstGridPointInDegrees');lon0=ec.codes_get(h,'longitudeOfFirstGridPointInDegrees');di=ec.codes_get(h,'iDirectionIncrementInDegrees');dj=ec.codes_get(h,'jDirectionIncrementInDegrees')
        lat=lat0+np.arange(nj)*dj*(1 if ec.codes_get(h,'jScansPositively') else -1);lon=(lon0+np.arange(ni)*di*(-1 if ec.codes_get(h,'iScansNegatively') else 1)+180)%360-180
        yi=np.flatnonzero((lat>=31)&(lat<=39));xi=np.flatnonzero((lon>=-100)&(lon<=-92));assert len(yi)==len(xi)==800
        v=ec.codes_get_array(h,'values').reshape(nj,ni)[np.ix_(yi,xi)]; base=np.zeros(v.shape,dtype='u1');base[v==missing]=1;base[v==nocov]=2;base[(~np.isfinite(v))|(v==ec.codes_get(h,'missingValue'))]=3
        return {'base_counts':{str(i):int((base==i).sum()) for i in range(4)},'metadata':{'Ni':ni,'Nj':nj,'gridType':ec.codes_get(h,'gridType'),'parameterCategory':cat,'parameterNumber':num},'roi_sha256':hashlib.sha256(v.tobytes()).hexdigest()}
    finally:
        if h is not None:ec.codes_release(h)
def main():
    by=defaultdict(list)
    with INV.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):r['bytes']=int(r['bytes']);r['t']=epoch(r['start_utc']);by[r['dataset']].append(r)
    for p in PRODUCTS:by[p].sort(key=lambda r:r['t'])
    QCOUT.mkdir(parents=True,exist_ok=True)
    for n,day in enumerate(DAYS,1):
        dest=QCOUT/(day+'.json')
        if dest.exists():continue
        start=epoch(day+'T00:00:00+00:00'); anchors=list(range(int(start+300),int(start+86400),300));selected={};records=[];causal_invalid={}
        for p in PRODUCTS:
            ts=[r['t'] for r in by[p]];used={};invalid=[]
            for t in anchors:
                j=bisect.bisect_right(ts,t)-1
                age=int(t-ts[j]) if j>=0 else -1
                if j < 0 or not 0 <= age <= 180:
                    # Frozen R4 semantics: an unavailable causal source is
                    # represented explicitly, never relaxed or future-filled.
                    invalid.append({'anchor_utc':dt.datetime.fromtimestamp(t,UTC).isoformat(),'age_seconds':age})
                    continue
                used[by[p][j]['key']]=by[p][j]
            selected[p]=used
            causal_invalid[p]=invalid
            for row in used.values():
                q=decode(row);records.append({'dataset':p,'key':row['key'],'start_utc':row['start_utc'],'bytes':row['bytes'],**q})
        totals={p:Counter() for p in PRODUCTS}
        for r in records:totals[r['dataset']].update(r['base_counts'])
        dump(dest,{'utc':now(),'date':day,'status':'PASS','anchors_examined':len(anchors),'max_mrms_age_seconds':180,'no_future_source':True,'selected_objects':{p:len(selected[p]) for p in PRODUCTS},'causal_invalid_anchors':causal_invalid,'records':records,'base_class_totals':{p:dict(v) for p,v in totals.items()},'labels_constructed':False,'models_loaded':False})
        dump(OUT/'R82_PROGRESS.json',{'utc':now(),'phase':'MRMS_CONTENT_QC','date':day,'dates_completed':n,'dates_required':len(DAYS),'labels_constructed':False,'models_loaded':False})
    dump(OUT/'R82_MRMS_CONTENT_QC_COMPLETE.json',{'utc':now(),'status':'PASS','dates':len(DAYS),'method':'R4 latest source timestamp <= anchor, age <=180 seconds; no future source','labels_constructed':False,'models_loaded':False})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(OUT/'R82_MRMS_CONTENT_QC_RETRY14_FAILURE.json',{'utc':now(),'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'labels_constructed':False,'models_loaded':False});raise
