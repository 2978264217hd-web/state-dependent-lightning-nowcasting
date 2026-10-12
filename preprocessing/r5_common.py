"""Frozen helpers for GATE0_LABEL_BUILD_R5."""
import csv,datetime as dt,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve();OUT=ROOT/'recovery_r5';UTC=dt.timezone.utc
FIXED={'2024-07-08','2024-07-10'};R=6371008.8
LABELS={0:'I+',1:'I-',2:'O+',3:'O-',-1:'INVALID'}
def now():return dt.datetime.now(UTC).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def stamp(t):return dt.datetime.fromtimestamp(float(t),UTC).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def to_json_safe(obj):
 try:
  import numpy as np
  if isinstance(obj,np.bool_):return bool(obj)
  if isinstance(obj,np.integer):return int(obj)
  if isinstance(obj,np.floating):return float(obj)
  if isinstance(obj,np.datetime64):return str(obj)
  if isinstance(obj,np.ndarray):return to_json_safe(obj.tolist())
 except ImportError:pass
 try:
  import pandas as pd
  if isinstance(obj,pd.Timestamp):return obj.isoformat()
 except ImportError:pass
 if isinstance(obj,Path):return str(obj)
 if isinstance(obj,(dt.datetime,dt.date)):return obj.isoformat()
 if isinstance(obj,dict):return {str(k):to_json_safe(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [to_json_safe(v) for v in obj]
 return obj
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');payload=json.dumps(to_json_safe(x),ensure_ascii=False,indent=2);q.write_text(payload,encoding='utf-8');json.loads(q.read_text(encoding='utf-8'));q.replace(p)
def csvwrite(p,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def xyz(lat,lon):
 import numpy as np
 a=np.deg2rad(lat);o=np.deg2rad(lon);return np.column_stack((np.cos(a)*np.cos(o),np.cos(a)*np.sin(o),np.sin(a)))
def chord(km):
 import numpy as np
 return 2*np.sin((km*1000/R)/2)
def freeze():
 OUT.mkdir(exist_ok=True);snap=OUT/'r45_frozen';snap.mkdir(exist_ok=True)
 if (snap/'manifest.json').exists():return
 names=['RESEARCH_CONTROL.md','LABEL_DEFINITION.md','GATE0_DECISION.md','DATA_AVAILABILITY_AUDIT.md','DATA_QC_REPORT.md','QC_POLICY.md','ALERT_RESOLUTION_R45.md','FINAL_DATA_MASK_POLICY_R45.md','R45_VALIDATION.json','R45_ANCHOR_VALIDITY.csv','GLM_ANCHOR_INVALID_MASK_R45.csv','GLM_CONTENT_ALERT_EVIDENCE_R45.csv','R4_AUDIT_SUMMARY.json','R4_AUDIT_VALIDATION.json']
 paths=[ROOT/n for n in names]+list((ROOT/'config').glob('*.json'))+list((ROOT/'scripts').glob('r45_*.py'))
 manifest={}
 for p in paths:
  rel=p.relative_to(ROOT);d=snap/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);manifest[str(rel)]={'sha256':sha(p)}
 dump(snap/'manifest.json',manifest)
 policy={'phase':'GATE0_LABEL_BUILD_R5','utc':now(),'candidate':'R3 QCComposite(t)>=30 and QC observation valid; no future MRMS',
  'history':'PRIMARY flashes within spherical-geodesic 20 km and [t-1800,t]','future':'PRIMARY flashes within spherical-geodesic 10 km and (t,t+1800]',
  'flash_time':'flash_time_offset_of_first_event with decoded NetCDF units','distance':'3D unit-sphere chord equivalent to mean-Earth great-circle distance; R=6371008.8m',
  'target_roi':[31,39,-100,-92],'glm_support':[30.7,39.3,-100.4,-91.6],'edge_rule':'support boundary verified >20km from ROI boundary; no erosion',
  'classes':{'I+':'C=1,H=0,Y=1','I-':'C=1,H=0,Y=0','O+':'C=1,H=1,Y=1','O-':'C=1,H=1,Y=0'},
  'primary':'all observation-valid flashes with original QF retained','sensitivity':'QF=0 only',
  'label_valid':'candidate plus R45 GLM history/future observability plus spatial edge validity; fixed dates excluded',
  'model_ready':'label_valid plus all three R4.5 required-field pixel masks at t-25,t-20,t-15,t-10,t-5,t and causal source age<=180; no interpolation',
  'onset_flash':'target-ROI primary flash with no earlier primary flash within 20km in [f-1800,f); observation-complete history required',
  'event_graph':'same UTC date, onset candidates connected when distance<=20km and time difference<=900s; connected components',
  'usable_event':'eligible event assigned at least one MODEL_READY_FULL I+ sample in 30min before an onset flash and within 10km; deterministic nearest flash/event tie-break',
  'gates':{'usable_events_min':100,'active_dates_min':20,'top1_max':.2,'top5_max':.5,'capture_pass_min':.7,'capture_hold_min':.5,'Ipos_attrition_max':.1,'qf_event_change_alert':.15,'qf_agreement_alert_below':.95,'extreme_Ipos_fraction_below':.1,'spatial_quadrant_alert_above':.5},
  'fixed_exclusions':sorted(FIXED),'no_sampling_or_weighting':True,'no_split':True,'no_training':True,'scientific_gate':'NOT_RUN'}
 dump(OUT/'PREREGISTRATION_R5.json',policy);dump(OUT/'PREREGISTRATION_SEAL_R5.json',{'sha256':sha(OUT/'PREREGISTRATION_R5.json')})
