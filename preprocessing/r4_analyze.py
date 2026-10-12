"""Date-level R4 masks and descriptive audits, no H/Y or onset labels."""
import bisect,collections,csv,datetime as dt,json,sqlite3,sys
from pathlib import Path
from r4_prepare import ROOT,OUT,DAYS,FIXED,QC,epoch,now,sha,dump
sys.path.insert(0,str(ROOT/'runtime_qc'))
import numpy as np,netCDF4 as nc
from analyze_semantic_pilot_r2 import classify
from r3_policy import primary_candidate,glm_anchor_invalid
PRODUCTS=[QC,'Reflectivity_-10C','EchoTop_30','MergedReflectivityComposite']
def stamp(t):return dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat()
def merge(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
def coverage(intervals,a,b):return sum(max(0,min(y,b)-max(x,a)) for x,y in intervals)
def gaps(intervals,a,b):
 out=[];cur=a
 for x,y in merge(intervals):
  x=max(x,a);y=min(y,b)
  if y<=a or x>=b:continue
  if x>cur:out.append([cur,x])
  cur=max(cur,y)
 if cur<b:out.append([cur,b])
 return out
def quant(hist,q):
 if not hist:return None
 a=np.array(sorted(hist));c=np.cumsum([hist[x] for x in a]);h=(int(c[-1])-1)*q;lo=int(np.floor(h));hi=int(np.ceil(h))
 return float(a[np.searchsorted(c,lo,side='right')]*(1-(h-lo))+a[np.searchsorted(c,hi,side='right')]*(h-lo))
def load(r):
 with np.load(r['roi_path']) as f:return f['value_raw'],f['base_class'],f['lat'],f['lon']
def analyze_day(day):
 dest=OUT/'daily'/(day+'.json')
 if dest.exists():return json.loads(dest.read_text())
 c=sqlite3.connect('file:'+(OUT/'FILE_QC_R4.sqlite').as_posix()+'?mode=ro',uri=True)
 expected=c.execute('SELECT count(*) FROM inventory WHERE date=?',(day,)).fetchone()[0]
 current=[json.loads(t) for t, in c.execute('SELECT result FROM files WHERE date=?',(day,))]
 # A failed access is engineering-pending, not a measured sensor outage.
 if len(current)!=expected or any(not r.get('readable') and not any(a.get('sha256') for a in r.get('attempts',[])) for r in current):c.close();return None
 base=int(epoch(day+'T00:00:00+00:00'));timeline=list(range(base,base+86400,300));fixed=day in FIXED
 previous=(dt.date.fromisoformat(day)-dt.timedelta(days=1)).isoformat()
 prior=[json.loads(t) for t, in c.execute("SELECT result FROM files WHERE date=? AND dataset!='GLM-L2-LCFA'",(previous,))];c.close()
 by={p:sorted([r for r in current+prior if r['source']['dataset']==p and r.get('readable')],key=lambda r:epoch(r['source']['start_utc'])) for p in PRODUCTS}
 times={p:[epoch(r['source']['start_utc']) for r in by[p]] for p in PRODUCTS}
 reference=next((r for r in by[QC] if r.get('roi_path')),None)
 if reference is None:return None
 _,_,lat,lon=load(reference);assert len(lat)==len(lon)==800
 glm=sorted([r for r in current if r['source']['dataset']=='GLM-L2-LCFA'],key=lambda r:epoch(r['source']['start_utc']))
 intervals=[];feature_issues=collections.Counter();overlap=reversal=0;laststart=lastend=None;untrustworthy=[]
 for r in glm:
  feature_issues.update(x for x in r.get('feature_issues',r.get('issues',[])) if x!='DEGRADED_FLASH_IN_SUPPORT')
  good=r.get('readable') and r.get('coverage_start') and r.get('coverage_end') and 'CONTENT_FILENAME_TIME_MISMATCH' not in r.get('issues',[])
  if good:
   a,b=epoch(r['coverage_start']),epoch(r['coverage_end']);good=b>a
  if good:
   if laststart is not None:reversal+=a<laststart;overlap+=a<lastend
   laststart,lastend=a,b;intervals.append([max(base,a),min(base+86400,b)])
  else:untrustworthy.append(r)
 union=merge(intervals);invalid=gaps(union,base,base+86400) if not fixed else []
 gaprows=[]
 for a,b in invalid:
  touching=[r for r in untrustworthy if epoch(r['source']['start_utc'])<b and epoch(r['source']['end_utc'])>a]
  kind='OBS_FILE_CORRUPT' if touching else 'METADATA_OR_ARCHIVE_COVERAGE_GAP'
  sources=[r['source']['key'] for r in glm if abs(epoch(r.get('coverage_end',r['source']['end_utc']))-a)<.001 or abs(epoch(r.get('coverage_start',r['source']['start_utc']))-b)<.001]
  gaprows.append({'gap_start':stamp(a),'gap_end':stamp(b),'duration_sec':b-a,'source':json.dumps(sources or [r['source']['key'] for r in touching]),'evidence_type':kind,
   'affected_date':day,'status':'INVALID_INTERVAL_MASK_NO_ZERO_FILL'})
 glmbins=[]
 for t in timeline:
  rr=[r for r in glm if t<=epoch(r['source']['start_utc'])<t+300];q=collections.Counter()
  for r in rr:q.update(r.get('flash_quality_counts',{}))
  cov=coverage(union,t,t+300) if not fixed else None
  status='EXCLUDED_FIXED' if fixed else 'OBS_VALID_COMPLETE' if cov>=300-1e-6 else 'OBS_FILE_MISSING' if not rr else 'OBS_FILE_CORRUPT' if any(not r.get('readable') for r in rr) else 'OBS_PARTIAL_SOURCE_GAP'
  glmbins.append({'time':stamp(t),'observation_status':status,'coverage_seconds':cov,'nominal_flash_count':q['0'],'degraded_flash_count':sum(q[str(k)] for k in [1,3,5]),
    **{f'qf{k}_count':q[str(k)] for k in [0,1,3,5]},'count_scope':'FULL_DISK_FILE_MEMBERSHIP_NOT_LABELS'})
 native={};hist=collections.Counter();fingerprints={}
 for p in PRODUCTS:
  rr=[r for r in current if r['source']['dataset']==p];read=[r for r in rr if r.get('readable')];counts=collections.Counter()
  for r in read:counts.update(r['base_counts'])
  native[p]={'listed':len(rr),'readable':len(read),'decode_fraction':len(read)/len(rr) if rr else None,'roi_pixel_times':sum(r['roi_pixels'] for r in read),
    'counts':dict(counts),'no_coverage_fraction':counts['2']/sum(counts.values()) if counts else None,
    'physical_min':min((r['physical_quantiles']['min'] for r in read if r.get('physical_quantiles',{}).get('min') is not None),default=None),
    'physical_max':max((r['physical_quantiles']['max'] for r in read if r.get('physical_quantiles',{}).get('max') is not None),default=None)}
  fp=collections.Counter()
  native[p]['native_cadence_seconds_histogram']=dict(collections.Counter(str(round(float(v),3)) for v in np.diff(sorted(epoch(r['source']['start_utc']) for r in rr))))
  native[p]['product_version_metadata']=dict(collections.Counter(json.dumps({k:r['metadata'].get(k) for k in ['edition','centre','tablesVersion','localTablesVersion']},sort_keys=True) for r in read if r['metadata'].get('tablesVersion') is not None))
  for r in read:
   m=r['metadata'];keys=['gridType','Ni','Nj','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees','iScansNegatively','jScansPositively','jPointsAreConsecutive','alternativeRowScanning','discipline','parameterCategory','parameterNumber']
   fp[json.dumps({k:m.get(k) for k in keys},sort_keys=True)]+=1
   if p==QC:
    if r.get('physical_histogram'):hist.update({float(k):v for k,v in r['physical_histogram'].items()})
    else:
     v,b,_,_=load(r);vals,nums=np.unique(v[b==0],return_counts=True);hist.update(dict(zip(vals.tolist(),nums.tolist())))
  fingerprints[p]=dict(fp)
 maskdir=ROOT/'SPATIOTEMPORAL_VALIDITY_MASK_R4';target=maskdir/(day+'.nc');temp=maskdir/(day+'.tmp.nc')
 out=nc.Dataset(temp,'w',format='NETCDF4');out.createDimension('time',288);out.createDimension('lat',800);out.createDimension('lon',800)
 out.createVariable('time','i8',('time',))[:]=timeline;out['time'].units='seconds since 1970-01-01 00:00:00 UTC'
 out.createVariable('lat','f8',('lat',))[:]=lat;out['lat'].units='degrees_north';out.createVariable('lon','f8',('lon',))[:]=lon;out['lon'].units='degrees_east'
 out.qc_policy_version='R4';out.max_mrms_age_seconds=180;out.fixed_exclusion=int(fixed)
 out.semantic_class_legend='0 physical;1 ambiguous missing;2 no coverage;3 decode invalid;4 R2 conditional no echo;5 R2 no isotherm echo;6 R2 no 30dBZ echo;7 ambiguous derived missing;8 no causal source'
 out.glm_observation_valid_definition='5-min bin [t,t+300) complete; for label support also apply both 30-min window gap masks and same-UTC-day constraint; feature QF does not invalidate bins.'
 out.radar_joint_valid_definition='qccomp_valid AND z10_valid AND et30_valid at each pixel; no full-ROI admission requirement'
 vs={name:out.createVariable(name,'u1',('time','lat','lon'),zlib=True,complevel=3,chunksizes=(1,100,100)) for name in ['qccomp_valid','z10_valid','et30_valid','radar_joint_valid','qccomp_class','z10_semantic_class','et30_semantic_class']}
 truthv=out.createVariable('qc_candidate_truth','i1',('time','lat','lon'),zlib=True,complevel=3,chunksizes=(1,100,100));truthv.legend='-1 unknown;0 FALSE;1 TRUE; QC -99 with valid source is FALSE'
 vs['qccomp_class'].flag_values=np.array([0,1,2,3,4],dtype='u1');vs['qccomp_class'].flag_meanings='QC_PHYSICAL_VALUE QC_MISSING_WITH_COVERAGE QC_NO_COVERAGE QC_DECODE_INVALID SOURCE_UNAVAILABLE_AT_ANCHOR'
 gv=out.createVariable('glm_observation_valid','u1',('time',));gw=out.createVariable('glm_history_future_window_valid','u1',('time',))
 srcvars={}
 for p in PRODUCTS:
  group=out.createGroup(p);srcvars[p]={n:group.createVariable(n,typ,('time',)) for n,typ in [('source_timestamp','i8'),('mrms_age_seconds','i4'),('causal_valid','u1'),('source_key',str)]}
 conditional={p:collections.Counter() for p in ['Reflectivity_-10C','EchoTop_30']};cross=collections.Counter();anchors=[];area_time=0.;z15=zcov=q30=qcov=joint_pixels=0
 cellarea=(6371.0088**2)*np.deg2rad(.01)*(np.sin(np.deg2rad(lat+.005))-np.sin(np.deg2rad(lat-.005)))
 for j,t in enumerate(timeline):
  values={};bases={};causal={};ages={};selected={}
  for p in PRODUCTS:
   k=bisect.bisect_right(times[p],t)-1;r=by[p][k] if k>=0 else None;age=int(t-times[p][k]) if k>=0 else -1;good=r is not None and 0<=age<=180
   selected[p]=r;causal[p]=good;ages[p]=age
   srcvars[p]['source_timestamp'][j]=int(times[p][k]) if k>=0 else -1;srcvars[p]['mrms_age_seconds'][j]=age;srcvars[p]['causal_valid'][j]=good;srcvars[p]['source_key'][j]=r['source']['key'] if r else ''
   if good:
    if not r.get('roi_path'):out.close();temp.unlink(missing_ok=True);return None
    v,b,la,lo=load(r);assert np.array_equal(la,lat) and np.array_equal(lo,lon);values[p]=v;bases[p]=b
   else:values[p]=np.full((800,800),np.nan);bases[p]=np.full((800,800),3,dtype='u1')
  q,z,e,c=[values[p] for p in PRODUCTS];qb,zb,eb,cb=[bases[p] for p in PRODUCTS]
  zs,es,cs=classify(z,e,c,zb,eb,cb)
  if not causal['Reflectivity_-10C']:zs[:]=8
  if not causal['EchoTop_30']:es[:]=8
  if not causal[QC]:qb[:]=4
  truth=primary_candidate(q,qb,causal[QC]);cand=truth==1;den=int(cand.sum())
  qv=(qb==0)|(qb==1);zv=(zs==0)|(zs==5);ev=(es==0)|(es==6);joint=qv&zv&ev
  for name,v in [('qccomp_valid',qv),('z10_valid',zv),('et30_valid',ev),('radar_joint_valid',joint),('qccomp_class',qb),('z10_semantic_class',zs),('et30_semantic_class',es)]:vs[name][j]=v
  truthv[j]=truth;gvalid=glmbins[j]['observation_status']=='OBS_VALID_COMPLETE';in_day=base+1800<=t<=base+84600;windowvalid=not fixed and in_day and not glm_anchor_invalid(t,invalid)
  gv[j]=gvalid;gw[j]=windowvalid
  causaljoint=all(causal[p] for p in PRODUCTS[:3]);joint_pixels+=int(joint.sum());q30+=den;qcov+=int(qv.sum());z15+=int(((zs==0)&(z>=15)).sum());zcov+=int(zv.sum())
  area_time+=float((cand.sum(axis=1)*cellarea).sum())/12
  for p,sem,code in [('Reflectivity_-10C',zs,5),('EchoTop_30',es,6)]:
   cc={'candidate_pixel_times':den,'physical_count':int((cand&(sem==0)).sum()),'conditional_no_echo_count':int((cand&(sem==code)).sum()),'ambiguous_count':int((cand&np.isin(sem,[1,7])).sum()),'no_coverage_count':int((cand&(sem==2)).sum()),'unavailable_count':int((cand&np.isin(sem,[3,8])).sum())};assert sum(v for k,v in cc.items() if k!='candidate_pixel_times')==den;conditional[p].update(cc)
  if causal[QC] and causal['MergedReflectivityComposite']:
   cross.update({'paired_pixel_times':640000,'raw_physical_qc_physical':int(((cb==0)&(qb==0)).sum()),'raw_physical_qc_missing':int(((cb==0)&(qb==1)).sum()),'raw_missing_qc_missing':int(((cb==1)&(qb==1)).sum()),'either_no_coverage':int(((cb==2)|(qb==2)).sum()),'raw_missing_qc_physical':int(((cb==1)&(qb==0)).sum())})
  anchors.append({'time':stamp(t),'date':day,'fixed_exclusion':fixed,'glm_bin_complete':gvalid,'glm_history_future_complete':windowvalid,
   'within_utc_day_label_support':in_day,'mrms_causal_joint':causaljoint,'joint_temporal_valid':windowvalid and causaljoint,
   'excluded_due_to_local_glm_gap':not fixed and in_day and glm_anchor_invalid(t,invalid),
   **{p+'_causal':causal[p] for p in PRODUCTS[:3]},**{p+'_age_seconds':ages[p] for p in PRODUCTS[:3]}})
 out.close();temp.replace(target)
 alerts=[]
 for p,v in conditional.items():
  if v['candidate_pixel_times'] and v['ambiguous_count']/v['candidate_pixel_times']>.05:alerts.append('DAILY_AMBIGUITY_SCREEN:'+p)
 for p,v in fingerprints.items():
  if len(v)>1:alerts.append('GRID_OR_PRODUCT_METADATA_TRANSITION:'+p)
 if feature_issues:alerts.append('GLM_FEATURE_CONTENT_REVIEW')
 if overlap or reversal:alerts.append('GLM_METADATA_OVERLAP_OR_REVERSAL_REVIEW')
 for p in PRODUCTS:
  if len(native[p]['product_version_metadata'])>1:alerts.append('PRODUCT_VERSION_METADATA_TRANSITION:'+p)
 daily={'date':day,'complete':True,'fixed_exclusion':fixed,'source_objects':expected,'native':native,'conditional':{k:dict(v) for k,v in conditional.items()},'raw_qc':dict(cross),
  'glm':{'listed_files':len(glm),'gap_seconds':sum(b-a for a,b in invalid) if not fixed else None,'longest_gap_seconds':max((b-a for a,b in invalid),default=0) if not fixed else None,
   'observation_complete_fraction':sum(r['observation_status']=='OBS_VALID_COMPLETE' for r in glmbins)/288 if not fixed else None,
   'coverage_seconds':coverage(union,base,base+86400) if not fixed else None,'timestamp_overlap_count':overlap,'timestamp_reversal_count':reversal,'feature_issues':dict(feature_issues),
   'source_cadence_seconds_histogram':dict(collections.Counter(str(round(float(v),3)) for v in np.diff([epoch(r['source']['start_utc']) for r in glm])))},
  'gaps':gaprows,'glm_bins':glmbins,'anchors':anchors,'causal_anchor_valid_fraction':sum(r['mrms_causal_joint'] for r in anchors)/288,
  'causal_by_field':{p:sum(r[p+'_causal'] for r in anchors)/288 for p in PRODUCTS[:3]},'radar_joint_valid_fraction':joint_pixels/(288*640000),
  'bias':{'qc_ge30_fraction_valid_roi':q30/qcov if qcov else None,'z10_ge15_fraction_valid_roi':z15/zcov if zcov else None,'qc_physical_max':quant(hist,1),'qc_physical_p99':quant(hist,.99),'convective_area_time_km2_hours':area_time},
  'qc_ge30_count':q30,'qc_valid_pixel_times':qcov,'alerts':alerts,'metadata_fingerprints':fingerprints,
  'mask_path':str(target),'mask_sha256':sha(target),'utc':now(),'qc_policy_version':'R4','analysis_script_sha256':sha(__file__)}
 dump(dest,daily);print('R4_DAY_ANALYZED',day,'gaps',len(gaprows),'alerts',alerts,flush=True);return daily
if __name__=='__main__':
 for d in sys.argv[1:] or DAYS:analyze_day(d)
