"""R82 frozen independent onset events and compact confirmatory sidecars."""
import argparse, collections, csv, datetime as dt, hashlib, json, math, os, traceback
from pathlib import Path
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from r82_frozen_label_readiness_build import ROOT,OUT,DAYS,CLASSES,F,REGISTRY,sha,read,dump,xyz,chord,now,epoch,stamp,truth
MAN=OUT/'R82_CONFIRMATORY_SAMPLE_MANIFEST'
def invalid_intervals():
 result=[]
 for file,start,end in [('R82_GLM_GAP_AUDIT.csv','gap_start_utc','gap_end_utc'),('R82_GLM_CONTENT_UNCERTAINTY.csv','start_utc','end_utc')]:
  with (OUT/file).open(encoding='utf-8-sig',newline='') as h:result.extend((epoch(r[start]),epoch(r[end])) for r in csv.DictReader(h))
 return result
def run_day(day,invalid):
 mp=OUT/'event_manifest_daily'/(day+'.json');path=MAN/(day+'.nc');lm=read(OUT/'label_readiness_daily'/(day+'.json'))
 if mp.exists() and path.exists():
  m=read(mp);assert m['manifest_sha256']==sha(path) and m['source_label_sha256']==lm['store_sha256'];return m
 f,ids,excluded=F['onset_flashes'](day,False,invalid);events,membership=F['cluster_events'](day,f,ids,'E')
 fq,iq,excludedq=F['onset_flashes'](day,True,invalid);qevents,_=F['cluster_events'](day,fq,iq,'Q')
 eventindex={e['event_id']:k+1 for k,e in enumerate(events)}
 assigned=collections.Counter();anchors=collections.defaultdict(set);pixels=collections.defaultdict(set)
 lp=OUT/'labels_r82'/(day+'.nc');assert sha(lp)==lm['store_sha256']
 with nc.Dataset(lp) as n:
  n.set_auto_mask(False);ti=n['time_index'][:];pi=n['pixel_index'][:];cl=n['class_primary'][:];ready=(n['readiness_flags'][:]&4)!=0;times=n['time'][:];lat=n['lat'][:];lon=n['lon'][:]
  rows=np.flatnonzero(ready).astype('u4');sti=ti[rows];spi=pi[rows];codes=np.zeros(len(rows),'u4')
  pts=xyz(f['lat'][ids],f['lon'][ids]);tree=cKDTree(pts) if len(ids) else None
  # Exact R5 assignment predicate and nearest-flash/time/event tie-break,
  # with the explicitly frozen R52 MASKABLE eligible set replacing FULL.
  if tree is not None:
   eligible=cl[rows]==0
   for tix in np.unique(sti[eligible]):
    lo=int(np.searchsorted(sti,tix,'left'));hi=int(np.searchsorted(sti,tix,'right'));sel=np.arange(lo,hi)[eligible[lo:hi]];p=spi[sel];t=times[tix]
    lists=tree.query_ball_point(xyz(lat[p//800],lon[p%800]),chord(10))
    for row,candidates in zip(sel,lists):
     valid=[k for k in candidates if f['time'][ids[k]]>t and f['time'][ids[k]]<=t+1800]
     if not valid:continue
     point=xyz(np.array([lat[spi[row]//800]]),np.array([lon[spi[row]%800]]))[0]
     best=min(valid,key=lambda k:(np.linalg.norm(pts[k]-point),f['time'][ids[k]],membership[ids[k]]));eid=membership[ids[best]]
     codes[row]=eventindex[eid];assigned[eid]+=1;anchors[eid].add(int(tix));pixels[eid].add(int(spi[row]))
  for e in events:
   eid=e['event_id'];e.update(n_positive_pixel_times=assigned[eid],n_unique_anchors=len(anchors[eid]),n_unique_target_pixels=len(pixels[eid]),duration_min=(max(anchors[eid])-min(anchors[eid]))*5 if anchors[eid] else 0,
      spatial_area_proxy_km2=len(pixels[eid])*.01*.01*111.195**2*math.cos(math.radians(e['centroid_lat'])),usable=assigned[eid]>0,
      exclusion_reason='' if assigned[eid]>0 else 'NO_MODEL_READY_MASKABLE_I_POS_IN_PRECEDING_30MIN',
      denominator_eligible=True,history_observation_complete=True,fixed_exclusion=False)
   e['spatial_region']=min(3,int((e['centroid_lat']-31)/2))*4+min(3,int((e['centroid_lon']+100)/2))
  sid=times[sti].astype('u8')*np.uint64(640000)+spi.astype('u8')
  assert not len(sid) or np.all(np.diff(sid.astype('i8'))>0)
  region=(np.minimum(3,((lat[spi//800]-31)/2).astype('i4'))*4+np.minimum(3,((lon[spi%800]+100)/2).astype('i4'))).astype('u1')
 path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp.nc')
 with nc.Dataset(tmp,'w',format='NETCDF4') as n:
  n.createDimension('sample',len(rows))
  for name,typ,value in [('sample_id','u8',sid),('source_label_row','u4',rows),('event_code','u4',codes),('spatial_region','u1',region)]:
   n.createVariable(name,typ,('sample',),zlib=True,complevel=4,fill_value=False)[:]=value
  n.source_label_path=str(lp);n.source_label_sha256=lm['store_sha256'];n.date=day
  n.sample_id_rule='uint64(anchor_epoch_seconds*640000+pixel_id), globally unique, ordered date then anchor then pixel'
  n.virtual_columns=json.dumps({'date':'partition date','anchor_time':'source time[source time_index[source_label_row]]','pixel_id':'source pixel_index[source_label_row]','C':1,
     'H':'source H_primary[source_label_row]','Y':'source Y_primary[source_label_row]','class':'source class_primary[source_label_row] mapped 0..3=I+,I-,O+,O-','temporal_valid':1,'maskable_ready':1,'event_id':'daily event code dictionary; 0 means no assigned onset event'})
 os.replace(tmp,path)
 m={'utc':now(),'date':day,'status':'PASS','samples':len(rows),'manifest_path':str(path),'manifest_sha256':sha(path),'source_label_sha256':lm['store_sha256'],
    'sample_id_partition_hash':hashlib.sha256(sid.astype('<u8',copy=False).tobytes()).hexdigest(),'event_code_dictionary':{str(k+1):e['event_id'] for k,e in enumerate(events)},'events':events,'qf0_raw_events':len(qevents),
    'raw_onset_events':len(events),'usable_onset_events':sum(e['usable'] for e in events),'excluded_incomplete_onset_flashes':excluded,'qf0_excluded_incomplete_onset_flashes':excludedq,
    'assignment_policy':'verbatim R5 nearest geodesic/time/event tie-break on MODEL_READY_MASKABLE I+','frozen_function_hashes':REGISTRY,'models_loaded':False}
 dump(mp,m);print('R82_EVENTS',day,len(events),m['usable_onset_events'],flush=True);return m
def writecsv(path,rows,fields):
 path=Path(path);q=path.with_name(path.name+'.tmp')
 with q.open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(q,path)
def finish(daily):
 events=[e for d in daily for e in d['events']];usable=[e for e in events if e['usable']]
 fields=list(events[0]) if events else ['event_id','date','usable']
 writecsv(OUT/'R82_ONSET_EVENT_AUDIT.csv',events,fields)
 by=collections.Counter(e['date'] for e in usable);top=sorted(by.values(),reverse=True);raw=len(events);n=len(usable)
 regions=collections.Counter(e['spatial_region'] for e in usable)
 spatial=[{'spatial_region':r,'lat_band':r//4,'lon_band':r%4,'event_count':regions[r],'share':regions[r]/n if n else 0,'lat_range':f'{31+2*(r//4)}..{33+2*(r//4)}','lon_range':f'{-100+2*(r%4)}..{-98+2*(r%4)}'} for r in range(16)]
 writecsv(OUT/'R82_SPATIAL_DISTRIBUTION.csv',spatial,list(spatial[0]))
 countsrows=[];matrix=np.zeros((4,4),'i8')
 total=collections.Counter()
 for d in daily:
  lm=read(OUT/'label_readiness_daily'/(d['date']+'.json'));cc=lm['class_counts']['MODEL_READY_MASKABLE'];total.update(cc);matrix+=np.array(lm['qf_transition_matrix'],'i8')
  countsrows.append({'date':d['date'],**cc,'I_positive_fraction':cc['I+']/(cc['I+']+cc['I-']) if cc['I+']+cc['I-'] else None,'O_positive_fraction':cc['O+']/(cc['O+']+cc['O-']) if cc['O+']+cc['O-'] else None})
 countsrows.append({'date':'TOTAL',**total,'I_positive_fraction':total['I+']/(total['I+']+total['I-']) if total['I+']+total['I-'] else None,'O_positive_fraction':total['O+']/(total['O+']+total['O-']) if total['O+']+total['O-'] else None})
 writecsv(OUT/'R82_CLASS_COUNTS.csv',countsrows,list(countsrows[0]))
 qn=sum(d['qf0_raw_events'] for d in daily);den=int(matrix.sum());agreement=float(np.trace(matrix)/den) if den else None;change=abs(qn-raw)/raw if raw else None
 dump(OUT/'R82_QF_SENSITIVITY.json',{'utc':now(),'primary_raw_onset_events':raw,'qf0_raw_onset_events':qn,'event_change_fraction':change,'label_agreement':agreement,'label_transition_matrix':matrix.tolist(),
   'QF_SENSITIVITY_ALERT_R82':'YES' if change is None or agreement is None or change>.15 or agreement<.95 else 'NO','primary_labels_changed':False})
 digest=hashlib.sha256()
 for d in daily:
  with nc.Dataset(d['manifest_path']) as f:
   f.set_auto_mask(False);sid=f['sample_id']
   for i in range(0,len(sid),500000):digest.update(np.asarray(sid[i:i+500000],dtype='<u8').tobytes())
 samplehash=digest.hexdigest()
 dump(MAN/'manifest.json',{'utc':now(),'status':'PASS','date_range':['2025-06-01','2025-07-31'],'dates':61,'set':'EXTERNAL_CONFIRMATION_SET','samples':sum(d['samples'] for d in daily),
  'sample_id_hash':samplehash,'sample_id_hash_encoding':'SHA256 concatenated little-endian uint64 ordered by date/anchor/pixel',
  'partitions':[{k:d[k] for k in ['date','samples','manifest_path','manifest_sha256','source_label_sha256','sample_id_partition_hash','event_code_dictionary']} for d in daily],
  'virtual_label_columns':True,'full_label_data_not_duplicated':True,'models_loaded':False})
 summary={'utc':now(),'status':'PASS','N_RAW_ONSET_EVENTS_R82':raw,'N_USABLE_ONSET_EVENTS_R82':n,'ONSET_EVENT_CAPTURE_RATE_R82':n/raw if raw else 0,
  'active_onset_dates':len(by),'top1_date_share':top[0]/n if n else 0,'top5_date_share':sum(top[:5])/n if n else 0,'max_16_region_share':max(regions.values(),default=0)/n if n else 0,
  'TEMPORAL_CONCENTRATION_ALERT':'YES' if n and (top[0]/n>.2 or sum(top[:5])/n>.5) else 'NO','SPATIAL_CONFIRMATION_ALERT':'YES' if n and max(regions.values(),default=0)/n>.3 else 'NO',
  'sample_id_hash':samplehash,'sample_count':sum(d['samples'] for d in daily),'sample_size_sufficient':n>=500 and len(by)>=20,'models_loaded':False,'R82_final_gate':'PENDING'}
 dump(OUT/'R82_ONSET_SUFFICIENCY_SUMMARY.json',summary)
 writecsv(OUT/'R82_SAMPLE_SUMMARY.csv',[{'date':d['date'],'samples':d['samples'],'raw_onset_events':d['raw_onset_events'],'usable_onset_events':d['usable_onset_events']} for d in daily],['date','samples','raw_onset_events','usable_onset_events'])
 return summary
def main(limit):
 invalid=invalid_intervals();daily=[];days=DAYS[:limit] if limit else DAYS
 for day in days:
  daily.append(run_day(day,invalid));dump(OUT/'R82_EVENT_MANIFEST_PROGRESS.json',{'utc':now(),'date':day,'dates_completed':len(daily),'dates_required':61,'pid':os.getpid(),'models_loaded':False})
 if not limit:
  summary=finish(daily);dump(OUT/'R82_EVENT_MANIFEST_COMPLETE.json',summary)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--limit-days',type=int,default=0);args=ap.parse_args()
 try:main(args.limit_days)
 except Exception as e:
  dump(OUT/('R82_EVENT_MANIFEST_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
