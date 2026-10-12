"""R4.5 bounded alert resolution. No labels, models, or new archive discovery."""
import bisect, collections, csv, datetime as dt, hashlib, json, os, shutil, sqlite3, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
OUT=ROOT/'recovery_r45'
sys.path.insert(0,str(ROOT/'runtime_qc'))
sys.path.insert(0,str(ROOT/'runtime_r45'))
import numpy as np
import netCDF4 as nc
from scipy.ndimage import label
QC='MergedReflectivityQCComposite'; Z='Reflectivity_-10C'; E='EchoTop_30'
FIXED={'2024-07-08','2024-07-10'}
UPDATED=['RESEARCH_CONTROL.md','DATA_AVAILABILITY_AUDIT.md','DATA_QC_REPORT.md','QC_POLICY.md','GATE0_DECISION.md']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def stamp(t):return dt.datetime.fromtimestamp(float(t),dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def csvwrite(p,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def intersections(t,a,b):
 # User R45: closed gap [a,b], history [t-1800,t], future (t,t+1800].
 return max(t-1800,a)<=min(t,b), b>t and a<=t+1800
def support(lat,lon):return (lat>=30.7)&(lat<=39.3)&(lon>=-100.4)&(lon<=-91.6)
def freeze():
 OUT.mkdir(exist_ok=True);snap=OUT/'r4_frozen';snap.mkdir(exist_ok=True)
 if (snap/'manifest.json').exists():return
 paths=list(ROOT.glob('*.md'))+list(ROOT.glob('*.csv'))+list(ROOT.glob('*.json'))+list((ROOT/'config').glob('*.json'))+list((ROOT/'scripts').glob('r4_*.py'))
 paths += [ROOT/'scripts/r3_policy.py',ROOT/'scripts/analyze_semantic_pilot_r2.py',ROOT/'recovery_r4/PREREGISTRATION_R4.json',ROOT/'recovery_r4/PREREGISTRATION_SEAL.json']
 manifest={}
 for p in paths:
  rel=p.relative_to(ROOT);dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]={'sha256':sha(p)}
 dump(snap/'manifest.json',manifest)
 policy={'phase':'GATE0_ALERT_RESOLUTION_R45','utc':now(),'source':'User R45 instruction, existing R4 evidence only',
  'closed_gap':True,'history':'[t-1800,t]','future':'(t,t+1800]','endpoint_change_from_R4':'Explicit user R45 endpoint convention; R3/R4 half-open files preserved',
  'fixed_dates':sorted(FIXED),'max_mrms_age':180,'full_ambiguity_threshold':.05,'spatial_connectivity':4,
  'primary_glm':'all observation-valid flashes; preserve QF','nominal_sensitivity':'QF=0 only; not built',
  'orphan_rule':'Not a QF synonym. If all orphan event/group coordinates lie outside frozen enlarged support and flash fields pass structural QC, nonblocking for ROI flash use. Otherwise mark recorded source support interval observation-uncertain and exclude intersecting anchors; never reconstruct flashes.',
  'native_duration':'observed consecutive timestamp span; gaps >180 seconds split runs; no continuous sensor state inferred',
  'anchor_duration':'consecutive invalid 5min samples times 300 seconds; report first-last span separately',
  'budget':'61 days total; fixed dates, closed GLM gap union, MRMS causal union, UTC same-day 30min support, optional orphan uncertainty; report overlap and mutually exclusive losses',
  'adequacy':'No pre-existing minimum sample/positive-event count supplied. Assess remaining temporal coverage and nonempty joint pixel support only; outcome counts and statistical power remain untested.',
  'no_labels':True,'no_training':True}
 dump(OUT/'POLICY_R45.json',policy);dump(OUT/'POLICY_SEAL.json',{'sha256':sha(OUT/'POLICY_R45.json')})

def fetch_alert(r):
 s=r['source'];p=OUT/'glm_alert_sources'/(hashlib.sha256(s['key'].encode()).hexdigest()+'.nc');p.parent.mkdir(exist_ok=True)
 expected=r.get('raw_source_sha256')
 assert expected,'R4 source SHA missing'
 if p.exists():assert sha(p)==expected;return r,p
 errors=[]
 for i in range(3):
  try:
   with urllib.request.urlopen(s['url'],timeout=90) as resp:body=resp.read()
   assert len(body)==int(s['bytes']) and hashlib.sha256(body).hexdigest()==expected
   p.write_bytes(body);return r,p
  except Exception as e:errors.append(repr(e));time.sleep(i+1)
 raise RuntimeError('Targeted recorded GLM alert read failed '+s['key']+str(errors))

def glm_review(c,days):
 cache=OUT/'glm_review.json'
 if cache.exists() and read(cache).get('orphan_policy_revision')==2:return read(cache)
 records=[];pure_qf=collections.Counter()
 for txt, in c.execute('SELECT result FROM files WHERE dataset=?',('GLM-L2-LCFA',)):
  r=json.loads(txt);issues=r.get('feature_issues',r.get('issues',[]))
  if any(x!='DEGRADED_FLASH_IN_SUPPORT' for x in issues):records.append(r)
  elif issues:pure_qf[r['source']['date']]+=1
 rows=[];uncertain=[]
 with ThreadPoolExecutor(4) as pool:
  for r,p in pool.map(fetch_alert,records):
   s=r['source'];issues=r.get('feature_issues',r.get('issues',[]));orphan={};inside=0;total=0
   with nc.Dataset(str(p)) as f:
    flash_ok=True
    for name in ['flash_id','flash_lat','flash_lon','flash_quality_flag','flash_time_offset_of_first_event','flash_time_offset_of_last_event']:
     a=f[name][:];flash_ok &= not np.ma.count_masked(a) and bool(np.isfinite(np.asarray(a)).all()) and a.shape==(r['flash_count'],)
    ids=np.asarray(f['flash_id'][:]);flash_ok &= len(np.unique(ids))==len(ids)
    for ent,parent in [('event','group'),('group','flash')]:
     a=np.asarray(f[ent+'_parent_'+parent+'_id'][:]);bad=~np.isin(a,np.asarray(f[parent+'_id'][:]))
     lat=np.asarray(f[ent+'_lat'][:])[bad];lon=np.asarray(f[ent+'_lon'][:])[bad];n=int(bad.sum());ni=int(support(lat,lon).sum());inside+=ni;total+=n
     orphan[ent]={'count':n,'inside_frozen_support':ni,'lat_min':float(lat.min()) if n else None,'lat_max':float(lat.max()) if n else None,'lon_min':float(lon.min()) if n else None,'lon_max':float(lon.max()) if n else None}
     assert n==r[ent+'_parent_'+parent+'_id_orphan_count']
   only_link=all(x.startswith('ORPHAN:') or x=='DEGRADED_FLASH_IN_SUPPORT' for x in issues)
   a=epoch(r['coverage_start']);b=epoch(r['coverage_end']);hasgap=any(a<epoch(g['gap_end']) and b>epoch(g['gap_start']) for g in days[s['date']]['gaps'])
   complete=r['readable'] and b-a==20 and not hasgap
   # A missing parent's location/completeness cannot be proven from child locations.
   # ORPHAN is separate from QF: conservatively isolate every affected source interval.
   nonblocking=flash_ok and complete and all(x=='DEGRADED_FLASH_IN_SUPPORT' for x in issues)
   status='RESOLVED_NONBLOCKING' if nonblocking else 'RESOLVED_MASKABLE'
   if not nonblocking:
    offsets=[r.get(k,{}).get('add_offset') for k in ['event_time_offset','group_time_offset','flash_time_offset_of_first_event']]
    a+=min([0]+[x for x in offsets if x is not None]);uncertain.append({'start':a,'end':b,'key':s['key'],'reason':'ORPHAN_HIERARCHY_OBSERVATION_UNCERTAINTY_NOT_QF_OUTAGE'})
   rows.append({'date':s['date'],'source_key':s['key'],'coverage_start':r['coverage_start'],'coverage_end':r['coverage_end'],'coverage_seconds':b-epoch(r['coverage_start']),
    'readable':r['readable'],'overlaps_observation_gap':hasgap,'flash_fields_structurally_valid':bool(flash_ok),'orphan_count':total,'orphan_inside_support':inside,'orphan_details':json.dumps(orphan),
    'feature_alert_type':'ORPHAN_HIERARCHY_NOT_PURE_QF','resolution_status':status,'evidence_path':str(p),'source_sha256':sha(p)})
 result={'orphan_policy_revision':2,'files':rows,'pure_qf_file_counts':dict(pure_qf),'uncertain_intervals':uncertain}
 csvwrite(ROOT/'GLM_CONTENT_ALERT_EVIDENCE_R45.csv',rows);dump(cache,result);return result

def native_nocov(c,days):
 cache=OUT/'native_nocov_summary.json'
 if cache.exists() and read(cache).get('native_run_revision')==2:return read(cache)
 rows=[];series=collections.defaultdict(list);fileinfo=[]
 for day,d in days.items():
  if not d['native'][Z]['no_coverage_fraction']:continue
  for source_index,(key,txt) in enumerate(c.execute('SELECT f.key,f.result FROM files f JOIN inventory i ON i.key=f.key WHERE f.date=? AND f.dataset=? ORDER BY i.timestamp,f.key',(day,Z))):
   r=json.loads(txt)
   if not r['base_counts'].get('2'):continue
   assert r.get('roi_path') and sha(r['roi_path'])==r['roi_sha256']
   with np.load(r['roi_path']) as f:b=f['base_class'];la=f['lat'];lo=f['lon']
   bad=b==2;lab,n=label(bad);sizes=np.bincount(lab.ravel());tt=epoch(r['source']['start_utc']);ix=np.argwhere(bad)
   assert len(ix)==r['base_counts']['2']
   fileinfo.append({'date':day,'time':stamp(tt),'pixel_count':len(ix),'cluster_count':n,'largest_cluster_pixels':int(sizes[1:].max()),'key':key})
   for y,x in ix:
    row={'date':day,'time':stamp(tt),'lat':float(la[y]),'lon':float(lo[x]),'row':int(y),'col':int(x),'cluster_id':int(lab[y,x]),'spatial_cluster_size':int(sizes[lab[y,x]]),'source_key':key,'roi_evidence':r['roi_path'],'mask_rule':'base_class==2 => Z10 pixel invalid; never fill'}
    row['source_index_in_date']=source_index
    rows.append(row);series[(int(y),int(x))].append((tt,len(rows)-1,source_index,day))
 maxspan=0;maxframes=0
 for series_rows in series.values():
  runs=[]
  for pair in sorted(series_rows):
   if not runs or pair[0]-runs[-1][-1][0]>180 or pair[2]!=runs[-1][-1][2]+1 or pair[3]!=runs[-1][-1][3]:runs.append([])
   runs[-1].append(pair)
  for run in runs:
   span=run[-1][0]-run[0][0];maxspan=max(maxspan,span);maxframes=max(maxframes,len(run))
   for _,j,_,_ in run:rows[j].update(duration_seconds=span,duration_definition='observed_native_first_to_last_span_consecutive_source_indices',consecutive_native_frames=len(run),run_start=stamp(run[0][0]),run_end=stamp(run[-1][0]))
 expected=sum(d['native'][Z]['counts'].get('2',0) for d in days.values());assert len(rows)==expected
 csvwrite(ROOT/'Z10_NO_COVERAGE_R45.csv',rows);csvwrite(OUT/'Z10_NO_COVERAGE_SOURCE_CLUSTERS.csv',fileinfo)
 result={'native_run_revision':2,'native_pixel_times':len(rows),'native_files':len(fileinfo),'dates':sorted(set(r['date'] for r in rows)),'unique_pixels':len(series),'max_observed_native_span_seconds':maxspan,'max_consecutive_native_frames':maxframes,
  'max_spatial_cluster_pixels':max(r['largest_cluster_pixels'] for r in fileinfo),'lat_bounds':[min(r['lat'] for r in rows),max(r['lat'] for r in rows)],'lon_bounds':[min(r['lon'] for r in rows),max(r['lon'] for r in rows)],
  'radar_site_boundary_attribution':'NOT_IDENTIFIABLE_FROM_COMPOSITE_METADATA; geographic and time footprint retained; no claimed radar-site cause'}
 dump(cache,result);return result

def analyze_pixels(day,d):
 cache=OUT/'daily'/(day+'.json');cache.parent.mkdir(exist_ok=True)
 if cache.exists():return read(cache)
 assert sha(d['mask_path'])==d['mask_sha256']
 result={'date':day,'mask_sha256':d['mask_sha256'],'products':{},'nocov_anchor_rows':[]}
 runs={p:np.zeros((800,800),dtype='u2') for p in [Z,E]};maxrun={Z:0,E:0};anyrun={Z:0,E:0};maxany={Z:0,E:0}
 stats={p:{'candidate_denominator':0,'ambiguous_candidate_pixel_times':0,'max_spatial_ambiguous_fraction':0.,'max_spatial_cluster_pixels':0,'first_ambiguous_time':None,'last_ambiguous_time':None,'anchors_with_ambiguity':0,'bbox':None,'max_nocov_run':0,'nocov_candidate_pixel_times':0,'nocov_pixel_times':0} for p in [Z,E]}
 nr=np.zeros((800,800),dtype='u2')
 with nc.Dataset(d['mask_path']) as f:
  f.set_auto_mask(False);tt=f['time'][:];la=f['lat'][:];lo=f['lon'][:]
  for i,t in enumerate(tt):
   cand=f['qc_candidate_truth'][i]==1;den=int(cand.sum())
   for p,var,validvar in [(Z,'z10_semantic_class','z10_valid'),(E,'et30_semantic_class','et30_valid')]:
    s=f[var][i];valid=f[validvar][i].astype(bool);bad=cand&np.isin(s,[1,7]);n=int(bad.sum());st=stats[p]
    assert not np.any(valid & np.isin(s,[1,2,3,7,8]))
    st['candidate_denominator']+=den;st['ambiguous_candidate_pixel_times']+=n;st['nocov_candidate_pixel_times']+=int((cand&(s==2)).sum());st['nocov_pixel_times']+=int((s==2).sum())
    runs[p]=np.where(bad,runs[p]+1,0);maxrun[p]=max(maxrun[p],int(runs[p].max()));anyrun[p]=anyrun[p]+1 if n else 0;maxany[p]=max(maxany[p],anyrun[p])
    if n:
     st['anchors_with_ambiguity']+=1;st['first_ambiguous_time']=st['first_ambiguous_time'] or stamp(t);st['last_ambiguous_time']=stamp(t);st['max_spatial_ambiguous_fraction']=max(st['max_spatial_ambiguous_fraction'],n/den)
     lab,ncpt=label(bad);st['max_spatial_cluster_pixels']=max(st['max_spatial_cluster_pixels'],int(np.bincount(lab.ravel())[1:].max()));yy,xx=np.where(bad);bbox=[float(la[yy].min()),float(la[yy].max()),float(lo[xx].min()),float(lo[xx].max())]
     old=st['bbox'];st['bbox']=bbox if old is None else [min(old[0],bbox[0]),max(old[1],bbox[1]),min(old[2],bbox[2]),max(old[3],bbox[3])]
    if p==Z:
     no=s==2;nr=np.where(no,nr+1,0);st['max_nocov_run']=max(st['max_nocov_run'],int(nr.max()))
     if no.any():result['nocov_anchor_rows'].append({'date':day,'anchor_time':stamp(t),'nocov_pixel_count':int(no.sum()),'overlap_qc_ge30':int((no&cand).sum()),'source_key':str(f.groups[Z]['source_key'][i]),'mask_evidence':d['mask_path'],'time_index':i})
 for p in [Z,E]:
  st=stats[p];den=st['candidate_denominator'];st.update(product=p,date=day,daily_ambiguous_fraction=st['ambiguous_candidate_pixel_times']/den if den else None,
    maximum_consecutive_ambiguous_duration_seconds=maxrun[p]*300,maximum_consecutive_ambiguous_observed_span_seconds=max(0,maxrun[p]-1)*300,
    max_anywhere_consecutive_duration_seconds=maxany[p]*300,spatial_connectivity=4,mask_available=True,mask_rule='class in {1,7} invalid for product/sample; existing R4 pixel validity enforces this',mask_path=d['mask_path'])
  assert den==d['conditional'][p]['candidate_pixel_times'] and st['ambiguous_candidate_pixel_times']==d['conditional'][p]['ambiguous_count']
  assert st['nocov_candidate_pixel_times']==d['conditional'][p]['no_coverage_count']
  result['products'][p]=st
 dump(cache,result);print('R45_PIXELS_VERIFIED',day,flush=True);return result

def temporal(days,gaps,glm):
 rows=[];anchors=[];contentrows=[]
 for day,d in sorted(days.items()):
  for ar in d['anchors']:
   t=epoch(ar['time']);hit=[]
   for i,g in enumerate(gaps,1):
    h,f=intersections(t,epoch(g['gap_start']),epoch(g['gap_end']))
    if h or f:
     hit.append(i);rows.append({'anchor_time':ar['time'],'affected_gap_id':f'G{i:03d}','history_intersection':h,'future_intersection':f,'invalid_reason':'GLM_OBSERVATION_GAP_CLOSED_ENDPOINT_R45','fixed_exclusion':day in FIXED,'within_utc_day_label_support':ar['within_utc_day_label_support']})
   uncertain=[]
   for v in glm['uncertain_intervals']:
    h,f=intersections(t,v['start'],v['end'])
    if h or f:uncertain.append(v['key']);contentrows.append({'anchor_time':ar['time'],'source_key':v['key'],'history_intersection':h,'future_intersection':f,'invalid_reason':v['reason']})
   fixed=day in FIXED;gap=bool(hit);causal=bool(ar['mrms_causal_joint']);insame=bool(ar['within_utc_day_label_support']);content=bool(uncertain)
   anchors.append({'date':day,'anchor_time':ar['time'],'fixed_excluded':fixed,'glm_gap_invalid':gap,'glm_content_uncertain':content,'mrms_causal_valid':causal,'same_utc_day_support':insame,
    'temporal_valid_before_day_boundary':not fixed and not gap and not content and causal,'final_temporal_valid':not fixed and not gap and not content and causal and insame})
 # Independent expanded-interval test: history+future union = closed [t-1800,t+1800].
 expected={a['anchor_time'] for a in anchors if any(epoch(g['gap_start'])-1800<=epoch(a['anchor_time'])<=epoch(g['gap_end'])+1800 for g in gaps)}
 assert expected=={r['anchor_time'] for r in rows}
 csvwrite(ROOT/'GLM_ANCHOR_INVALID_MASK_R45.csv',rows);csvwrite(ROOT/'R45_ANCHOR_VALIDITY.csv',anchors)
 csvwrite(OUT/'GLM_CONTENT_UNCERTAIN_ANCHORS.csv',contentrows,['anchor_time','source_key','history_intersection','future_intersection','invalid_reason'])
 total=len(anchors);candidate=[a for a in anchors if not a['fixed_excluded']];n=len(candidate);left=anchors;budget=[]
 steps=[('FIXED_EXCLUDED_DATE',lambda a:a['fixed_excluded']),('GLM_OBSERVATION_GAP',lambda a:a['glm_gap_invalid']),('MRMS_CAUSAL_UNAVAILABLE',lambda a:not a['mrms_causal_valid']),('GLM_CONTENT_UNCERTAIN',lambda a:a['glm_content_uncertain']),('UTC_DAY_30MIN_SUPPORT_BOUNDARY',lambda a:not a['same_utc_day_support'])]
 for reason,pred in steps:
  lost=[a for a in left if pred(a)];left=[a for a in left if not pred(a)];budget.append({'reason':reason,'exclusive_removed_count':len(lost),'fraction_of_61day_anchors':len(lost)/total,'fraction_of_59day_anchors':len(lost)/n,'standalone_candidate_count':sum(pred(a) for a in candidate),'remaining_count':len(left)})
 budget.append({'reason':'FINAL_TEMPORALLY_VALID','exclusive_removed_count':0,'fraction_of_61day_anchors':len(left)/total,'fraction_of_59day_anchors':len(left)/n,'standalone_candidate_count':len(left),'remaining_count':len(left)})
 assert len(left)==sum(a['final_temporal_valid'] for a in anchors)
 csvwrite(ROOT/'R45_EFFECTIVE_ANCHOR_BUDGET.csv',budget)
 return {'total_61day_anchors':total,'candidate_59day_anchors':n,'fixed_removed':total-n,'glm_gap_invalid_candidate_anchors':sum(a['glm_gap_invalid'] for a in candidate),
  'glm_gap_valid_candidate_anchors':sum(not a['glm_gap_invalid'] for a in candidate),'glm_gap_loss_fraction':sum(a['glm_gap_invalid'] for a in candidate)/n,
  'glm_content_uncertain_candidate_anchors':sum(a['glm_content_uncertain'] for a in candidate),'remaining_before_day_boundary':sum(a['temporal_valid_before_day_boundary'] for a in candidate),
  'remaining_final':len(left),'remaining_fraction_of_candidate':len(left)/n,'remaining_dates':len(set(a['date'] for a in left)),
  'per_gap_anchor_links':len(rows),'max_gap_seconds':max(float(g['duration_sec']) for g in gaps),'closed_endpoint_independent_check':'PASS','budget':budget}

def extreme(days):
 rows=[];summary={};ref=[d for d in days.values() if not d['fixed_exclusion']];ext=days['2024-07-08']['bias']
 for metric in ['qc_ge30_fraction_valid_roi','z10_ge15_fraction_valid_roi','qc_physical_p99','convective_area_time_km2_hours']:
  ranked=sorted(ref,key=lambda d:(-d['bias'][metric],d['date']))
  for k in [1,3,5,10]:
   top=ranked[:k];vals=[d['bias'][metric] for d in top]
   rows.append({'excluded_date':'2024-07-08','metric':metric,'top_k':k,'excluded_value':ext[metric],'top_dates':';'.join(d['date'] for d in top),'top_values':json.dumps(vals),'top_mean':sum(vals)/k,
    'top_min':min(vals),'top_max':max(vals),'top_mean_to_excluded_ratio':sum(vals)/k/ext[metric] if 'p99' not in metric else 'NA_LOGARITHMIC_DBZ','strongest_to_excluded_ratio':max(vals)/ext[metric] if 'p99' not in metric else 'NA_LOGARITHMIC_DBZ','top_mean_minus_excluded':sum(vals)/k-ext[metric],
    'metric_units':'dBZ' if 'p99' in metric else 'km2 hours' if 'integral' in metric or 'area_time' in metric else 'fraction','excluded_share_of_59_plus_excluded_metric_sum':ext[metric]/(sum(d['bias'][metric] for d in ref)+ext[metric]) if 'p99' not in metric else 'NA_NONADDITIVE_INTENSITY',
    'EXTREME_EVENT_SAMPLE_LOSS':'YES','fixed_exclusion_retained':True})
  summary[metric]={'strongest_date':ranked[0]['date'],'strongest_value':ranked[0]['bias'][metric],'excluded_value':ext[metric],'ratio':ranked[0]['bias'][metric]/ext[metric] if 'p99' not in metric else None,'difference':ranked[0]['bias'][metric]-ext[metric]}
 csvwrite(ROOT/'EXTREME_DAY_EXCLUSION_IMPACT_R45.csv',rows);return summary

def main():
 freeze();assert sha(OUT/'POLICY_R45.json')==read(OUT/'POLICY_SEAL.json')['sha256']
 revision={'revision':2,'supersedes':'POLICY_R45.json orphan_rule only; original sealed file preserved','rule':'All recorded ORPHAN source intervals are conservatively observation-uncertain regardless of child coordinates; extend to packed time lower bound (add_offset), then exact anchor-window intersection. QF degradation alone remains nonblocking.',
  'reason':'Child coordinates outside ROI support do not prove a missing parent flash position or completeness. No parent reconstruction, weather selection, or changed observation-gap duration.',
  'scientific_rules_unchanged':True}
 dump(OUT/'ORPHAN_MASK_POLICY_REVISION_R45.json',revision)
 days={p.stem:read(p) for p in (ROOT/'recovery_r4/daily').glob('*.json')};assert len(days)==61
 r4=read(ROOT/'R4_AUDIT_SUMMARY.json');assert r4['full_results_available'] and read(ROOT/'R4_AUDIT_VALIDATION.json')['status']=='PASS_ARTIFACT_CHECKS'
 c=sqlite3.connect('file:'+(ROOT/'recovery_r4/FILE_QC_R4.sqlite').as_posix()+'?mode=ro',uri=True)
 assert c.execute('SELECT count(*) FROM files WHERE readable=1').fetchone()[0]==430553
 glm=glm_review(c,days);print('R45_GLM_REVIEW',len(glm['files']),len(glm['uncertain_intervals']),flush=True)
 no=native_nocov(c,days);c.close();print('R45_NATIVE_NOCOV',json.dumps(no),flush=True)
 alertdays=[d for d in days.values() if d['alerts'] and not d['fixed_exclusion']];assert len(alertdays)==13
 selected=sorted(set(d['date'] for d in alertdays)|set(no['dates']))
 pix={day:analyze_pixels(day,days[day]) for day in selected}
 amb=[r for day in sorted(pix) if day in {d['date'] for d in alertdays} for r in pix[day]['products'].values()]
 for r in amb:
  r['largest_spatial_cluster_fraction_of_roi']=r['max_spatial_cluster_pixels']/640000
  r['A_spatial_concentration_evidence']=f'Largest connected cluster {r["max_spatial_cluster_pixels"]}/640000 ROI pixels; may still cover {r["max_spatial_ambiguous_fraction"]:.6f} of QC>=30 at one anchor.'
  r['B_temporal_concentration_evidence']=f'Any candidate ambiguity at {r["anchors_with_ambiguity"]}/288 anchors; longest same-pixel run {r["maximum_consecutive_ambiguous_duration_seconds"]}s. Not assumed to be one brief episode.'
  r['C_systemic_failure_evidence']='Native source decode/grid consistency verified by R4; ambiguity exactly encoded in existing semantic classes and invalid masks. Physical cause unresolved; no unisolatable systemic failure demonstrated.'
 csvwrite(ROOT/'DAILY_AMBIGUITY_RESOLUTION_R45.csv',amb);csvwrite(OUT/'Z10_NO_COVERAGE_ANCHOR_OVERLAP.csv',[r for p in pix.values() for r in p['nocov_anchor_rows']])
 gaps=list(csv.DictReader((ROOT/'GLM_TEMPORAL_INVALID_MASK_R4.csv').open(encoding='utf-8-sig')));assert len(gaps)==43
 temp=temporal(days,gaps,glm);ext=extreme(days)
 ledger=[]
 for d in sorted(alertdays,key=lambda d:d['date']):
  day=d['date']
  for code in d['alerts']:
   if code.startswith('DAILY_AMBIGUITY_SCREEN:'):
    p=code.split(':',1)[1];s=pix[day]['products'][p]
    row={'date':day,'alert_code':code,'alert_source':'DAILY_DATA_QC_R4.csv','product':p,'start_time':s['first_ambiguous_time'],'end_time':s['last_ambiguous_time'],
     'spatial_extent':json.dumps(s['bbox']),'temporal_extent':s['maximum_consecutive_ambiguous_duration_seconds'],'convective_region_overlap':s['ambiguous_candidate_pixel_times'],
     'deterministic_mask_available':True,'mask_rule':s['mask_rule'],'requires_date_exclusion':False,'evidence':s['mask_path'],'resolution_status':'RESOLVED_MASKABLE',
     'notes':f'daily={s["daily_ambiguous_fraction"]}; maximum within-candidate spatial fraction={s["max_spatial_ambiguous_fraction"]}; temporal concentration quantified, no inferred product failure or changed 5% full threshold'}
   elif code=='GLM_FEATURE_CONTENT_REVIEW':
    rr=[r for r in glm['files'] if r['date']==day];assert rr
    status='RESOLVED_MASKABLE' if any(r['resolution_status']=='RESOLVED_MASKABLE' for r in rr) else 'RESOLVED_NONBLOCKING'
    row={'date':day,'alert_code':code,'alert_source':'R4 FILE_QC + SHA-matched targeted source reread','product':'GLM-L2-LCFA','start_time':min(r['coverage_start'] for r in rr),'end_time':max(r['coverage_end'] for r in rr),
     'spatial_extent':'See per-file orphan coordinate bounds in GLM_CONTENT_ALERT_EVIDENCE_R45.csv','temporal_extent':sum(r['coverage_seconds'] for r in rr),'convective_region_overlap':'No outcome-based selection; frozen support intersection='+str(sum(r['orphan_inside_support'] for r in rr)),
     'deterministic_mask_available':True,'mask_rule':'Pure QF does not invalidate observation. Every ORPHAN source interval, including its declared packed time lower bound, masks intersecting anchors as observation-uncertain.',
     'requires_date_exclusion':False,'evidence':str(ROOT/'GLM_CONTENT_ALERT_EVIDENCE_R45.csv'),'resolution_status':status,'notes':'ORPHAN is not degraded QF. Missing parent validity/location is not inferred from child coordinates; all 39 affected source intervals conservatively masked.'}
   else:row={'date':day,'alert_code':code,'resolution_status':'UNRESOLVED','notes':'Unhandled content alert; gate cannot pass'}
   ledger.append(row)
 for g in gaps:
  ledger.append({'date':g['affected_date'],'alert_code':'GLM_OBSERVATION_GAP','alert_source':'GLM_TEMPORAL_INVALID_MASK_R4.csv','product':'GLM-L2-LCFA','start_time':g['gap_start'],'end_time':g['gap_end'],'spatial_extent':'observation temporal coverage; whole ROI','temporal_extent':g['duration_sec'],'convective_region_overlap':'Not conditioned on lightning outcome','deterministic_mask_available':True,'mask_rule':'closed gap intersects history [t-1800,t] or future (t,t+1800] => sample invalid','requires_date_exclusion':False,'evidence':g['source'],'resolution_status':'RESOLVED_MASKABLE','notes':'No zero fill or interpolation'})
 for day in no['dates']:
  rows=pix[day]['nocov_anchor_rows'];s=pix[day]['products'][Z]
  ledger.append({'date':day,'alert_code':'Z10_LOCAL_NO_COVERAGE','alert_source':'R4 native base_class + R4 anchor semantic_class','product':Z,'start_time':min(r['anchor_time'] for r in rows) if rows else 'No selected anchor','end_time':max(r['anchor_time'] for r in rows) if rows else 'No selected anchor','spatial_extent':str(no['lat_bounds'])+str(no['lon_bounds']),'temporal_extent':s['max_nocov_run']*300,'convective_region_overlap':s['nocov_candidate_pixel_times'],'deterministic_mask_available':True,'mask_rule':'class==2 => corresponding product pixel/sample invalid','requires_date_exclusion':False,'evidence':str(ROOT/'Z10_NO_COVERAGE_R45.csv'),'resolution_status':'RESOLVED_MASKABLE','notes':'Coordinates/native runs/connected clusters retained; no radar-site causation claimed'})
 csvwrite(ROOT/'R45_ALERT_LEDGER.csv',ledger)
 rank={'RESOLVED_NONBLOCKING':0,'RESOLVED_MASKABLE':1,'UNRESOLVED':2,'SYSTEMIC_FAILURE':3};date_status={d['date']:max((r['resolution_status'] for r in ledger if r['date']==d['date'] and r['alert_code'] in d['alerts']),key=rank.get) for d in alertdays}
 count=dict(collections.Counter(date_status.values()));count={k:count.get(k,0) for k in rank}
 den=sum(d['conditional'][Z]['candidate_pixel_times'] for d in days.values() if not d['fixed_exclusion']);numer=sum(d['conditional'][Z]['no_coverage_count'] for d in days.values() if not d['fixed_exclusion'])
 assert numer==sum(p['products'][Z]['nocov_candidate_pixel_times'] for p in pix.values())
 sufficient=temp['remaining_final']>0 and temp['remaining_dates']==59 and all(d['radar_joint_valid_fraction']>0 for d in days.values() if not d['fixed_exclusion'])
 passed=not count['UNRESOLVED'] and not count['SYSTEMIC_FAILURE'] and sufficient and all(v<.05 for v in r4['conditional_ambiguity_full'].values()) and r4['no_coverage_full'][QC]==0
 gate='PASS' if passed else 'HOLD'
 summary={'utc':now(),'phase':'GATE0_ALERT_RESOLUTION_R45','alert_date_status_counts':count,'alert_date_status':date_status,'ledger_row_count':len(ledger),'glm_review_files':len(glm['files']),
  'glm_pure_qf_nonblocking_file_count':sum(glm['pure_qf_file_counts'].values()),'glm_orphan_inside_support_total':sum(r['orphan_inside_support'] for r in glm['files']),'temporal':temp,'z10_nocov':no,
  'z10_nocov_given_qc_ge30':{'numerator':numer,'denominator':den,'fraction':numer/den},'max_daily_ambiguity':{},
  'unmaskable_mrms_issue':False,'extreme_comparison':ext,'EXTREME_EVENT_SAMPLE_LOSS':'YES','ALERT_RESOLUTION_GATE_R45':gate,'FULL_CONTENT_DATA_GATE':'PASS_WITH_EXCLUSIONS' if passed else 'HOLD','DATA_GATE':'PASS_WITH_EXCLUSIONS' if passed else 'HOLD','READY_FOR_LABEL_BUILD':'YES' if passed else 'NO',
  'FULL_CONTENT_DATA_GATE_R4':'HOLD','SCIENTIFIC_GATE0':'NOT_RUN','GI':'NA','GO':'NA','Gamma':'NA','95% CI':'NA','labels_built':False,'training_started':False,'adequacy_interpretation':'Temporal/data-quality eligibility only; no outcome frequencies or inferential power established'}
 for p in [Z,E]:
  rr=[{'date':d['date'],'fraction':d['conditional'][p]['ambiguous_count']/d['conditional'][p]['candidate_pixel_times']} for d in days.values() if not d['fixed_exclusion'] and d['conditional'][p]['candidate_pixel_times']]
  summary['max_daily_ambiguity'][p]=max(rr,key=lambda r:r['fraction'])
 dump(OUT/'SUMMARY_R45.json',summary)
 write_reports(summary,glm)
 validate(summary,days,pix)
 print('R45_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)

def write_reports(s,glm):
 policy=f'''# FINAL_DATA_MASK_POLICY_R45

Phase: GATE0_ALERT_RESOLUTION_R45. Prior scientific hypotheses, required QCComposite/Z10/ET30, 5% full-period ambiguity threshold and two fixed exclusions remain frozen.

A. 2024-07-08 and 2024-07-10 remain excluded for SENSOR_DATA_INCOMPLETE.
B. GLM gap [a,b] is CLOSED under the explicit R45 user instruction. History [t-1800,t] intersects iff max(t-1800,a)<=min(t,b); future (t,t+1800] intersects iff b>t and a<=t+1800. Any intersection invalidates the sample. This explicitly supersedes the R4 half-open endpoint convention only in new R45 masks; old files are preserved. Gap length remains b-a; endpoint inclusion changes anchors, not measured seconds. All 61-day anchors are evaluated before fixed-date exclusion; duplicate gap-to-anchor links are deduplicated for budget counts.
C/D/E. QC/Z10/ET30 no coverage invalidates the corresponding pixel/sample, never fill with zero.
F. Z10/ET30 ambiguous semantic class 1 or 7 is invalid for that product/sample. R4 z10_valid accepts only 0/5; et30_valid only 0/6; qccomp_valid only 0/1. radar_joint_valid is their pixelwise intersection. A sample requiring multiple pixels must require all used input pixels valid; spatial aggregation/support details must be fixed before Label Build, not learned from outcomes.
G. Feature QF 1/3/5 alone never removes observation coverage or a 5-min window. PRIMARY_GLMLABEL = all observation-valid flashes; SENSITIVITY_GLMLABEL_NOMINAL = QF=0 only. Neither label is built here. ORPHAN hierarchy anomalies are assessed separately: targeted SHA-matched records verify flash-array structural integrity and orphan coordinates against the frozen support rectangle lat 30.7..39.3, lon -100.4..-91.6. Child coordinates do not prove missing parent location/completeness. Therefore ALL recorded ORPHAN source intervals are conservatively observation-uncertain, including the declared packed-time lower bound (add_offset, normally -5 seconds); exact touching anchors are masked. This is not a measured observation outage and is separately budgeted from the 43 actual gaps. ORPHAN_MASK_POLICY_REVISION_R45.json supersedes the initial outside-support nonblocking rule, whose sealed record is retained for traceability. No parent reconstruction or flash synthesis.
H. Latest MRMS source timestamp <= anchor, age 0..180 seconds; causal validity and per-pixel masks both required. No future replacement or interpolation. R4 source-key/time/age variables are reused.

Final deterministic validity is: not fixed excluded AND no closed GLM-gap intersection AND no GLM content-uncertainty intersection AND all three MRMS causal flags AND same UTC day 30min support AND R4 per-pixel radar_joint_valid. Temporal counts do not count spatial samples, lightning outcomes, I/O groups or model labels. Future-window coverage is used only to determine future target observability, never as an input feature or a source replacement at t.

R45_ANCHOR_VALIDITY.csv is the complete temporal mask; GLM_ANCHOR_INVALID_MASK_R45.csv retains gap links. Z10_NO_COVERAGE_R45.csv lists every native no-coverage pixel-time. Source-native durations are observed first-last spans, not interpolated continuous outages. Consecutive 5-min ambiguity durations are run length times 300 seconds; spatial connectivity is four-neighbor. Causation by a particular radar boundary cannot be established from composite metadata.

There is no pre-registered numerical minimum sample/positive-event count. Eligibility here requires all 59 candidate dates retain temporal anchors and nonempty joint radar support; this does not establish event counts, subgroup adequacy or statistical power. EXTREME_EVENT_SAMPLE_LOSS = YES; exclusion of July 8 remains a limitation.

ALERT_RESOLUTION_GATE_R45 = {s['ALERT_RESOLUTION_GATE_R45']}
DATA_GATE = {s['DATA_GATE']}
READY_FOR_LABEL_BUILD = {s['READY_FOR_LABEL_BUILD']}
SCIENTIFIC_GATE0 = NOT_RUN; GI/GO/Gamma/95% CI = NA. STOP BEFORE LABEL BUILD.
'''
 (ROOT/'FINAL_DATA_MASK_POLICY_R45.md').write_text(policy,encoding='utf-8')
 report='# ALERT_RESOLUTION_R45\n\nR4.5 resolves existing alerts only. Full R4 historical gate remains HOLD; current data readiness is governed by the R45 resolution below.\n\n'+policy+'\n## Verified results\n\n```json\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n```\n'
 (ROOT/'ALERT_RESOLUTION_R45.md').write_text(report,encoding='utf-8')
 header=f'''# R45 current data validity status ({s['utc']})

ALERT_RESOLUTION_GATE_R45 = {s['ALERT_RESOLUTION_GATE_R45']}
FULL_CONTENT_DATA_GATE = {s['FULL_CONTENT_DATA_GATE']}
DATA_GATE = {s['DATA_GATE']}
READY_FOR_LABEL_BUILD = {s['READY_FOR_LABEL_BUILD']}
SCIENTIFIC_GATE0 = NOT_RUN
GI = NA
GO = NA
Gamma = NA
95% CI = NA
EXTREME_EVENT_SAMPLE_LOSS = YES

R4 FULL_CONTENT_DATA_GATE_R4=HOLD is preserved as historical evidence. R45 alert resolutions and exact endpoint-aware masks are documented in ALERT_RESOLUTION_R45.md and FINAL_DATA_MASK_POLICY_R45.md. All original required fields and fixed excluded dates remain unchanged. No label build, model training or Gate-1. Historical content follows verbatim.

---

'''
 for name in UPDATED:(ROOT/name).write_text(header+(OUT/'r4_frozen'/name).read_text(encoding='utf-8'),encoding='utf-8')
 new=['R45_ALERT_LEDGER.csv','GLM_ANCHOR_INVALID_MASK_R45.csv','Z10_NO_COVERAGE_R45.csv','DAILY_AMBIGUITY_RESOLUTION_R45.csv','EXTREME_DAY_EXCLUSION_IMPACT_R45.csv','FINAL_DATA_MASK_POLICY_R45.md','R45_EFFECTIVE_ANCHOR_BUDGET.csv','ALERT_RESOLUTION_R45.md','R45_VALIDATION.json','GLM_CONTENT_ALERT_EVIDENCE_R45.csv','R45_ANCHOR_VALIDITY.csv']
 dump(OUT/'DELIVERABLE_PATHS.json',{'new_files':[str(ROOT/n) for n in new],'updated_files':[str(ROOT/n) for n in UPDATED],'evidence_directory':str(OUT),'reused_pixel_masks':str(ROOT/'SPATIOTEMPORAL_VALIDITY_MASK_R4'),'script':str(Path(__file__))})

def validate(s,days,pix):
 preserved={}
 for name,r in read(OUT/'r4_frozen/manifest.json').items():
  if name in UPDATED:assert (ROOT/name).read_text(encoding='utf-8').endswith((OUT/'r4_frozen'/name).read_text(encoding='utf-8'))
  else:assert sha(ROOT/name)==r['sha256'],name;preserved[name]=r['sha256']
 for day,r in pix.items():assert sha(days[day]['mask_path'])==r['mask_sha256']
 assert sum(s['alert_date_status_counts'].values())==13
 assert s['z10_nocov']['native_pixel_times']==50674
 assert all(s[k]=='NA' for k in ['GI','GO','Gamma','95% CI']) and not s['labels_built'] and not s['training_started']
 assert s['READY_FOR_LABEL_BUILD']==('YES' if s['ALERT_RESOLUTION_GATE_R45']=='PASS' else 'NO')
 assert sha(OUT/'POLICY_R45.json')==read(OUT/'POLICY_SEAL.json')['sha256']
 # Endpoint regressions independent of real gaps.
 assert intersections(0,1800,1801)==(False,True)
 assert intersections(0,-1801,-1800)==(True,False)
 assert intersections(0,0,1)==(True,True)
 assert intersections(0,-1,0)==(True,False)
 assert intersections(0,1801,1802)==(False,False)
 checks={'utc':now(),'status':'PASS_ARTIFACT_AND_MASK_CHECKS','gate':s['ALERT_RESOLUTION_GATE_R45'],'r4_historical_files_preserved':preserved,
  'alert_dates_classified':13,'selected_days_pixel_masks_recounted':sorted(pix),'all_native_z10_nocov_pixels_located':s['z10_nocov']['native_pixel_times'],
  'closed_gap_anchor_union_independent_check':'PASS','endpoint_boundary_regression_checks':'PASS','r4_conditional_counts_reconciled':'PASS','glm_alert_raw_sha_verified_files':s['glm_review_files'],
  'frozen_policy_sha256':sha(OUT/'POLICY_R45.json'),'orphan_policy_revision_sha256':sha(OUT/'ORPHAN_MASK_POLICY_REVISION_R45.json'),'script_sha256':sha(__file__),'no_labels':True,'no_models':True,'scientific_gate':'NOT_RUN'}
 dump(ROOT/'R45_VALIDATION.json',checks)

if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  OUT.mkdir(exist_ok=True);dump(OUT/'FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'engineering_state':'BLOCKED','labels_built':False});raise
