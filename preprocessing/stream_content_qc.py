"""Resumable real-payload QC. No interpolation, labels, model, or prediction metric.

Network payloads are transient memory buffers, not retained full archives.
SQLite stores every source, attempt, hash, parsed metadata, statistics and error.
Decoder processes are isolated because netCDF/HDF5 is not thread-safe.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request

ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
sys.path.insert(0,str(ROOT/'runtime_qc'))
UTC=dt.timezone.utc
EXCLUDED={'2024-07-08','2024-07-10'}
SPECS={'Reflectivity_-10C':(3,54,'dBZ',-99,-999,15),
       'EchoTop_30':(3,45,'km MSL',-1,-3,None),
       'MergedReflectivityComposite':(11,1,'dBZ',-99,-999,30)}

def isotime(value):
    return dt.datetime.fromisoformat(value.replace('Z','+00:00'))

def download(row):
    attempts=[]
    for attempt in range(1,4):
        rec={'attempt':attempt,'utc':dt.datetime.now(UTC).isoformat()}
        try:
            t=time.monotonic()
            req=urllib.request.Request(row['url'],headers={'User-Agent':'Gate0ContentQC/1.0'})
            with urllib.request.urlopen(req,timeout=60) as r:
                body=r.read()
                rec.update(http_status=r.status,etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'))
            rec.update(seconds=time.monotonic()-t,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            if len(body)!=int(row['bytes']): raise ValueError('CONTENT_LENGTH_DIFFERS_FROM_INVENTORY')
            attempts.append(rec)
            return row,body,attempts
        except Exception as e:
            rec['error']=repr(e); attempts.append(rec)
            if attempt<3: time.sleep(attempt)
    return row,None,attempts

def glm_qc(row,body):
    import netCDF4 as nc4
    import numpy as np
    out={'readable':False,'issues':[],'warnings':[]}
    with nc4.Dataset('memory.nc',memory=body) as nc:
        names=['flash_id','flash_lat','flash_lon','flash_time_offset_of_first_event',
               'flash_time_offset_of_last_event','flash_quality_flag','group_id','group_lat',
               'group_lon','group_time_offset','group_parent_flash_id','group_quality_flag',
               'event_id','event_lat','event_lon','event_time_offset','event_parent_group_id',
               'product_time','product_time_bounds']
        missing=[n for n in names if n not in nc.variables]
        if missing: out['issues'].append('MISSING_FIELDS:'+','.join(missing)); return out
        arrays={n:nc[n][:] for n in names}
        out['readable']=True
        out['coverage_start']=nc.getncattr('time_coverage_start')
        out['coverage_end']=nc.getncattr('time_coverage_end')
        start,end=map(isotime,[out['coverage_start'],out['coverage_end']])
        out['coverage_seconds']=(end-start).total_seconds()
        if end<=start: out['issues'].append('NONPOSITIVE_COVERAGE')
        if abs((start-isotime(row['start_utc'])).total_seconds())>0.001 or abs((end-isotime(row['end_utc'])).total_seconds())>0.001:
            out['issues'].append('CONTENT_FILENAME_TIME_MISMATCH')
        out['fields']={n:{'shape':list(v.shape),'units':getattr(nc[n],'units',None)} for n,v in arrays.items()}
        for n,a in arrays.items():
            if np.ma.count_masked(a) or (a.dtype.kind in 'fiu' and not np.isfinite(np.asarray(a)).all()): out['issues'].append('INVALID_VALUES:'+n)
        for entity in ['flash','group','event']:
            ids=np.asarray(arrays[entity+'_id'])
            lat=np.asarray(arrays[entity+'_lat']); lon=np.asarray(arrays[entity+'_lon'])
            out[entity+'_count']=int(ids.size)
            if len(np.unique(ids))!=len(ids): out['issues'].append('DUPLICATE_ID:'+entity)
            if not (ids.shape==lat.shape==lon.shape): out['issues'].append('SHAPE_MISMATCH:'+entity)
            if np.any((lat < -90)|(lat>90)|(lon < -180)|(lon>180)): out['issues'].append('ILLEGAL_COORDINATE:'+entity)
        for child,parent in [('event_parent_group_id','group_id'),('group_parent_flash_id','flash_id')]:
            orphan=int(np.count_nonzero(~np.isin(arrays[child],arrays[parent])))
            out[child+'_orphan_count']=orphan
            if orphan: out['issues'].append('ORPHAN:'+child)
        for n in ['event_time_offset','group_time_offset','flash_time_offset_of_first_event','flash_time_offset_of_last_event']:
            a=np.asarray(arrays[n]); units=nc[n].units
            dates=nc4.num2date([0],units,only_use_cftime_datetimes=False)
            epoch=dates[0].replace(tzinfo=UTC)
            offset=(epoch-start).total_seconds()
            rel=a.astype(float)+offset
            # LCFA packed offset includes [-5,20] seconds in the actual metadata.
            # Do not reject legitimate pre-file flash starts or require sorted records.
            scale=float(getattr(nc[n],'scale_factor',1)); add=float(getattr(nc[n],'add_offset',0))
            out[n]={'units':units,'min_relative_seconds':float(rel.min()) if rel.size else None,
                    'max_relative_seconds':float(rel.max()) if rel.size else None,'scale_factor':scale,'add_offset':add}
            if np.any(rel < add+offset-abs(scale)) or np.any(rel > (end-start).total_seconds()+abs(scale)):
                out['issues'].append('TIME_OUTSIDE_PACKED_COVERAGE:'+n)
        first=np.asarray(arrays['flash_time_offset_of_first_event']); last=np.asarray(arrays['flash_time_offset_of_last_event'])
        if np.any(first>last): out['issues'].append('FLASH_END_BEFORE_START')
        for entity in ['flash','group']:
            q=np.asarray(arrays[entity+'_quality_flag'])
            out[entity+'_quality_counts']={str(k):int(v) for k,v in zip(*np.unique(q,return_counts=True))}
            if np.any(~np.isin(q,[0,1,3,5])): out['issues'].append('UNKNOWN_QUALITY_FLAG:'+entity)
        lat=np.asarray(arrays['flash_lat']); lon=np.asarray(arrays['flash_lon'])
        roi=(lat>=31)&(lat<=39)&(lon>=-100)&(lon<=-92)
        # Larger rectangular support encloses the 20 km label buffer. No labels built here.
        support=(lat>=30.7)&(lat<=39.3)&(lon>=-100.4)&(lon<=-91.6)
        out['roi_flash_count']=int(roi.sum()); out['support_flash_count']=int(support.sum())
        degraded=support&(np.asarray(arrays['flash_quality_flag'])!=0)
        out['support_degraded_flash_count']=int(degraded.sum())
        if degraded.any(): out['issues'].append('DEGRADED_FLASH_IN_SUPPORT')
        if not out['flash_count'] and (out['group_count'] or out['event_count']): out['issues'].append('ABNORMAL_EMPTY_FLASH_HIERARCHY')
        if not out['flash_count']: out['warnings'].append('EMPTY_FULL_DISK_REQUIRES_NEIGHBOR_REVIEW')
        units=nc['flash_time_offset_of_first_event'].units
        epoch=nc4.num2date([0],units,only_use_cftime_datetimes=False)[0].replace(tzinfo=UTC).timestamp()
        bins=Counter(int((epoch+float(x))//300)*300 for x in first[roi])
        out['roi_flash_counts_by_epoch_bin']={str(k):v for k,v in bins.items()}
        out['valid_for_observation']=not out['issues'] and out['flash_count']>0
    return out

def mrms_qc(row,body):
    import eccodes as ec
    import numpy as np
    out={'readable':False,'issues':[],'warnings':[]}
    raw=gzip.decompress(body)  # Includes CRC/end-of-stream validation.
    if not raw.startswith(b'GRIB') or not raw.endswith(b'7777'): raise ValueError('MALFORMED_GRIB')
    if int.from_bytes(raw[8:16],'big')!=len(raw): raise ValueError('MULTIPLE_OR_TRUNCATED_GRIB_MESSAGES')
    h=ec.codes_new_from_message(raw)
    try:
        keys=['edition','centre','discipline','parameterCategory','parameterNumber','gridType','Ni','Nj',
              'latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees',
              'latitudeOfLastGridPointInDegrees','longitudeOfLastGridPointInDegrees',
              'iDirectionIncrementInDegrees','jDirectionIncrementInDegrees','iScansNegatively',
              'jScansPositively','jPointsAreConsecutive','alternativeRowScanning',
              'year','month','day','hour','minute','second','missingValue','bitmapPresent']
        meta={k:ec.codes_get(h,k) for k in keys}; out['metadata']=meta
        actual=dt.datetime(*(meta[k] for k in ['year','month','day','hour','minute','second']),tzinfo=UTC)
        out['content_timestamp']=actual.isoformat()
        if actual!=isotime(row['start_utc']): out['issues'].append('CONTENT_FILENAME_TIME_MISMATCH')
        cat,num,unit,missing,nocover,threshold=SPECS[row['dataset']]
        out['unit_from_NOAA_product_table']=unit
        if (meta['discipline'],meta['parameterCategory'],meta['parameterNumber'])!=(209,cat,num): out['issues'].append('PRODUCT_CODE_MISMATCH')
        if meta['gridType']!='regular_ll' or meta['jPointsAreConsecutive'] or meta['alternativeRowScanning']:
            raise ValueError('UNSUPPORTED_GRID_OR_SCAN')
        ni,nj=meta['Ni'],meta['Nj']
        lon=meta['longitudeOfFirstGridPointInDegrees']+np.arange(ni)*meta['iDirectionIncrementInDegrees']*(-1 if meta['iScansNegatively'] else 1)
        lon=(lon+180)%360-180
        lat=meta['latitudeOfFirstGridPointInDegrees']+np.arange(nj)*meta['jDirectionIncrementInDegrees']*(1 if meta['jScansPositively'] else -1)
        rows=np.flatnonzero((lat>=31)&(lat<=39)); cols=np.flatnonzero((lon>=-100)&(lon<=-92))
        covered=bool(lat.min()<=31 and lat.max()>=39 and lon.min()<=-100 and lon.max()>=-92 and len(rows) and len(cols))
        out['roi_covered']=covered
        if not covered: out['issues'].append('ROI_NOT_COVERED')
        values=ec.codes_get_array(h,'values')  # Actually decode ALL packed grid values.
        if values.size!=ni*nj: raise ValueError('GRID_VALUE_COUNT_MISMATCH')
        roi=values.reshape(nj,ni)[np.ix_(rows,cols)]
        out['readable']=True; out['roi_pixels']=int(roi.size)
        out['roi_shape']=list(roi.shape)
        out['roi_coordinate_bounds']=[float(lat[rows].min()),float(lat[rows].max()),float(lon[cols].min()),float(lon[cols].max())]
        invalid=(~np.isfinite(roi))|(roi==missing)|(roi==nocover)|(roi==meta['missingValue'])
        out['sentinel_counts']={str(v):int(np.count_nonzero(roi==v)) for v in [missing,nocover,meta['missingValue']]}
        out['missing_fraction']=float(invalid.mean()); out['valid_fraction']=float((~invalid).mean())
        valid=roi[~invalid]
        out['all_missing_roi']=bool(not valid.size)
        out['constant_valid_field']=bool(valid.size and np.ptp(valid)==0)
        out['numeric_summary']={k:float(v) for k,v in zip(['min','p01','median','p99','max'],np.quantile(valid,[0,.01,.5,.99,1]))} if valid.size else {k:None for k in ['min','p01','median','p99','max']}
        if not valid.size: out['issues'].append('ALL_MISSING_ROI')
        if out['constant_valid_field']: out['warnings'].append('CONSTANT_FIELD_REVIEW_NOT_AUTO_REJECT')
        out['negative_nonsentinel_pixels']=int(np.count_nonzero(valid<0))
        if row['dataset']=='EchoTop_30' and np.any(valid<0): out['issues'].append('NEGATIVE_ECHOTOP_NOT_SENTINEL')
        out['threshold_pixels']=int(np.count_nonzero(valid>=threshold)) if threshold is not None else None
        out['threshold_fraction_of_full_roi']=out['threshold_pixels']/roi.size if threshold is not None and roi.size else None
        out['threshold_fraction_of_valid_roi']=out['threshold_pixels']/valid.size if threshold is not None and valid.size else None
        out['zero_pixels']=int(np.count_nonzero(valid==0))
        out['valid_for_observation']=not out['issues']
        # Conservative complete-ROI time mask; pixel-level missing remains reported.
        out['complete_roi_valid']=out['valid_for_observation'] and out['missing_fraction']==0
    finally: ec.codes_release(h)
    return out

def decode(downloaded):
    row,body,attempts=downloaded
    result={'source':row,'attempts':attempts,'qc_version':'R1.0','qc_utc':dt.datetime.now(UTC).isoformat()}
    if body is None:
        result.update(readable=False,issues=['DOWNLOAD_FAILED_NOT_SENSOR_ZERO'],valid_for_observation=False)
        return result
    try: result.update(glm_qc(row,body) if row['dataset']=='GLM-L2-LCFA' else mrms_qc(row,body))
    except Exception as exc: result.update(readable=False,issues=['DECODE_ERROR'],error=repr(exc),valid_for_observation=False)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--download-workers',type=int,default=16)
    ap.add_argument('--decode-workers',type=int,default=2)
    ap.add_argument('--limit',type=int,default=0,help='Preflight only; never treated as a full audit')
    args=ap.parse_args()
    run=ROOT/'recovery_r1'/'full_content_qc'; run.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(run/'FILE_QC.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS files (key TEXT PRIMARY KEY,dataset TEXT,date TEXT,timestamp TEXT,readable INTEGER,eligible INTEGER,bytes INTEGER,result TEXT)')
    done={r[0] for r in db.execute('SELECT key FROM files')}
    with (ROOT/'audit/20260905T103339Z/OBJECT_INVENTORY.csv').open(newline='',encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f) if not (r['dataset']=='GLM-L2-LCFA' and r['date'] in EXCLUDED)]
    # Date order independent of weather. MRMS includes excluded dates for bias audit.
    rows.sort(key=lambda r:(r['start_utc'],r['dataset']))
    todo=[r for r in rows if r['key'] not in done]
    if args.limit: todo=todo[:args.limit]
    provenance={'scope_objects':len(rows),'scope_bytes':sum(int(r['bytes']) for r in rows),
                'excluded_glm_dates':sorted(EXCLUDED),'mrms_dates':61,'network_workers':args.download_workers,
                'decode_workers':args.decode_workers,'limit':args.limit,'authorization':'USER_APPROVED_FULL_STREAMING',
                'source_manifest_sha256':hashlib.sha256((ROOT/'audit/20260905T103339Z/OBJECT_INVENTORY.csv').read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp=dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    (run/f'RUN_{stamp}.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    iterator=iter(todo); pending_download={}; pending_decode={}; completed=0; last=time.monotonic(); began=last
    print(json.dumps(provenance),flush=True)
    with ThreadPoolExecutor(max_workers=args.download_workers) as net,ProcessPoolExecutor(max_workers=args.decode_workers) as cpu:
        exhausted=False
        while not exhausted or pending_download or pending_decode:
            while not exhausted and len(pending_download)<args.download_workers:
                try: row=next(iterator)
                except StopIteration: exhausted=True; break
                pending_download[net.submit(download,row)]=row
            # Bound queued decoded payloads to limit RAM; all writes by parent process.
            ready=[f for f in pending_download if f.done()]
            for f in ready[:max(0,args.decode_workers*2-len(pending_decode))]:
                row=pending_download.pop(f)
                pending_decode[cpu.submit(decode,f.result())]=row
            for f in [f for f in pending_decode if f.done()]:
                row=pending_decode.pop(f)
                try: result=f.result()
                except Exception as exc: result={'source':row,'readable':False,'issues':['WORKER_FAILURE'],'error':repr(exc)}
                db.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?,?)',(row['key'],row['dataset'],row['date'],row['start_utc'],int(result.get('readable',False)),int(result.get('valid_for_observation',False)),int(row['bytes']),json.dumps(result,separators=(',',':'))))
                completed+=1
            if time.monotonic()-last>=20 or (exhausted and not pending_download and not pending_decode):
                db.commit()
                progress={'utc':dt.datetime.now(UTC).isoformat(),'processed_total':len(done)+completed,'scope_total':len(rows),
                          'processed_this_run':completed,'elapsed_seconds':time.monotonic()-began,
                          'complete':len(done)+completed==len(rows),'pending_downloads':len(pending_download),'pending_decodes':len(pending_decode)}
                (run/'PROGRESS.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(json.dumps(progress),flush=True); last=time.monotonic()
            time.sleep(.01)
    db.commit(); db.close()

if __name__=='__main__': main()
