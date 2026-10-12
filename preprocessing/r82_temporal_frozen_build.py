"""R82 data-only temporal audit using unchanged R4 union and R45 endpoints."""
import ast, bisect, csv, datetime as dt, hashlib, json, os, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
REC=OUT/'glm_precision_recovery';UTC=dt.timezone.utc
DAYS=[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
FIELDS=('MergedReflectivityQCComposite','Reflectivity_-10C','EchoTop_30')
def now():return dt.datetime.now(UTC).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def stamp(t):return dt.datetime.fromtimestamp(float(t),UTC).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def write_csv(path,rows,fields=None):
 p=Path(path);q=p.with_suffix('.tmp.csv');fields=fields or list(rows[0])
 with q.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(q,p)
def install():
 env={}
 for rel,names in [('scripts/r4_analyze.py',['merge','gaps']),('scripts/r45_resolve.py',['intersections'])]:
  source=(ROOT/rel).read_text(encoding='utf-8-sig');nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
  for name in names:exec(compile(ast.get_source_segment(source,nodes[name]),str(ROOT/rel)+'::'+name,'exec'),env)
 return env
def main():
 assert read(OUT/'R82_CACHE_FINAL_VALIDATION.json')['status']=='PASS'
 assert read(OUT/'R82_GLM_PRECISION_RECOVERY_COMPLETE.json')['status']=='PASS'
 funcs=install();intersections=funcs['intersections'];gap_func=funcs['gaps']
 # Exact closed-gap touching cases, including future's open lower endpoint.
 assert intersections(0,0,0)==(True,False)
 assert intersections(0,-1800,-1800)==(True,False)
 assert intersections(0,1800,1800)==(False,True)
 assert intersections(0,1800.00001,1801)==(False,False)
 inventory={p:[] for p in FIELDS}
 with (OUT/'R82_MRMS_OBJECT_INVENTORY.csv').open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['dataset'] in inventory:inventory[row['dataset']].append((epoch(row['start_utc']),row['key']))
 for p in FIELDS:inventory[p].sort()
 itimes={p:[x[0] for x in inventory[p]] for p in FIELDS}
 rows=[];gaprows=[];uncertainrows=[];daily=[];sourcekeys=[];all_causal=True
 # R45 reviews every non-QF content issue across the full date span. A neighbor
 # granule's declared -5s packed lower bound can affect the preceding UTC day.
 all_uncertain=[]
 for d in DAYS:
  for r in read(REC/'timeline'/(d+'.json'))['records']:
   qc=r['R4_QC'];issues=qc.get('issues',[])
   review=any(x!='DEGRADED_FLASH_IN_SUPPORT' for x in issues)
   if not review:continue
   a=epoch(qc['coverage_start']);b=epoch(qc['coverage_end'])
   offsets=[qc.get(k,{}).get('add_offset') for k in ['event_time_offset','group_time_offset','flash_time_offset_of_first_event']]
   a+=min([0]+[x for x in offsets if x is not None])
   all_uncertain.append((a,b))
   uncertainrows.append({'date':d,'start_utc':stamp(a),'end_utc':stamp(b),'key':r['key'],'reason':'FROZEN_R45_REVISION2_NON_QF_CONTENT_UNCERTAINTY'})
 for day in DAYS:
  flash=REC/'flash_daily'/(day+'.npz');meta=read(flash.with_suffix('.json'));timeline=REC/'timeline'/(day+'.json')
  assert meta['npz_sha256']==sha(flash) and meta['timeline_sha256']==sha(timeline)
  records=read(timeline)['records'];assert len(records)==meta['source_files']
  intervals=[];uncertain=all_uncertain;day0=int(epoch(day+'T00:00:00+00:00'))
  for r in records:
   q=r['R4_QC'];sourcekeys.append(r['key'])
   if r['observation_usable']:intervals.append([epoch(q['coverage_start']),epoch(q['coverage_end'])])
  gs=gap_func(intervals,day0,day0+86400)
  for a,b in gs:gaprows.append({'date':day,'gap_start_utc':stamp(a),'gap_end_utc':stamp(b),'duration_seconds':b-a})
  dayrows=[];sources=[];qc=read(OUT/'mrms_content_qc'/(day+'.json'));available={r['key'] for r in qc['records']}
  for i in range(288):
   t=day0+300*i;hist=any(intersections(t,a,b)[0] for a,b in gs);future=any(intersections(t,a,b)[1] for a,b in gs)
   content=any(any(intersections(t,a,b)) for a,b in uncertain)
   inside=day0+1800<=t<=day0+84600
   mrms_current=True;selected=[]
   for field in FIELDS:
    ix=bisect.bisect_right(itimes[field],t)-1
    ts,key=inventory[field][ix] if ix>=0 else (-1,'')
    age=t-ts if ix>=0 else -1;valid=bool(key) and 0<=age<=180
    mrms_current &= valid
    if valid and i>0:assert key in available,'SELECTED_MRMS_SOURCE_NOT_IN_IMMUTABLE_QC'
    selected.append({'date':day,'time_index':i,'anchor_utc':stamp(t),'product':field,'source_key':key,'source_utc':stamp(ts) if ts>=0 else '', 'age_seconds':age,'causal_valid':valid,'cache_available':key in available})
    all_causal &= (not valid or (ts<=t and 0<=age<=180))
   sources.extend(selected)
   dayrows.append({'date':day,'time_index':i,'anchor_time':stamp(t),'fixed_excluded':False,'glm_gap_history_invalid':hist,'glm_gap_future_invalid':future,'glm_gap_invalid':hist or future,'glm_content_uncertain':content,'same_utc_day_support':inside,'glm_label_temporal_valid':inside and not(hist or future or content),'mrms_current_causal_valid':bool(mrms_current)})
  for i,row in enumerate(dayrows):
   row['mrms_six_frame_causal_valid']=bool(i>=5 and all(r['mrms_current_causal_valid'] for r in dayrows[i-5:i+1]))
   row['temporal_input_available']=row['glm_label_temporal_valid'] and row['mrms_six_frame_causal_valid']
  write_csv(OUT/'temporal_daily'/(day+'.mrms_sources.csv'),sources)
  daily.append({'date':day,'anchors':288,'same_day_support':sum(r['same_utc_day_support'] for r in dayrows),'gap_affected_anchors':sum(r['glm_gap_invalid'] for r in dayrows),'content_affected_anchors':sum(r['glm_content_uncertain'] for r in dayrows),'valid_glm_anchors':sum(r['glm_label_temporal_valid'] for r in dayrows),'valid_input_temporal_anchors':sum(r['temporal_input_available'] for r in dayrows),'gap_count':len(gs),'max_gap_seconds':max((b-a for a,b in gs),default=0)})
  rows.extend(dayrows)
 assert len(sourcekeys)==len(set(sourcekeys))==263425
 assert all_causal
 write_csv(OUT/'R82_GLM_TEMPORAL_MASK.csv',rows)
 write_csv(OUT/'R82_GLM_GAP_AUDIT.csv',gaprows,['date','gap_start_utc','gap_end_utc','duration_seconds'])
 write_csv(OUT/'R82_GLM_CONTENT_UNCERTAINTY.csv',uncertainrows,['date','start_utc','end_utc','key','reason'])
 write_csv(OUT/'R82_TEMPORAL_DAILY_SUMMARY.csv',daily)
 total=len(rows);valid=sum(r['glm_label_temporal_valid'] for r in rows)
 summary={'utc':now(),'status':'PASS','dates':61,'total_5min_anchors':total,'same_day_supported_anchors':sum(r['same_utc_day_support'] for r in rows),'total_gap_count':len(gaprows),'max_gap_seconds':max((r['duration_seconds'] for r in gaprows),default=0),'gap_affected_anchors':sum(r['glm_gap_invalid'] for r in rows),'gap_affected_supported_anchors':sum(r['glm_gap_invalid'] and r['same_utc_day_support'] for r in rows),'content_uncertain_source_intervals':len(uncertainrows),'content_affected_anchors':sum(r['glm_content_uncertain'] for r in rows),'valid_anchors':valid,'valid_anchor_fraction':valid/total,'valid_supported_anchor_fraction':valid/sum(r['same_utc_day_support'] for r in rows),'mrms_causal_history_available_anchors':sum(r['temporal_input_available'] for r in rows),'closed_gap_history':'[t-1800,t]','closed_gap_future':'(t,t+1800]','max_mrms_age_seconds':180,'future_fill':False,'gap_minimum_duration_threshold':None,'content_rule':'Frozen R45 revision2 uncertainty intervals, extended by declared packed time lower bound; QF alone nonblocking','R45_policy_sha256':sha(ROOT/'FINAL_DATA_MASK_POLICY_R45.md'),'mask_sha256':sha(OUT/'R82_GLM_TEMPORAL_MASK.csv'),'source_resolution_directory':str(OUT/'temporal_daily'),'models_loaded':False}
 dump(OUT/'R82_TEMPORAL_VALIDITY_SUMMARY.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 (OUT/'temporal_daily').mkdir(exist_ok=True)
 try:main()
 except Exception as e:
  dump(OUT/('R82_TEMPORAL_BUILD_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
