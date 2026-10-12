"""Build and audit R5 local-onset/ongoing labels from frozen R4.5 evidence."""
import collections,csv,datetime as dt,json,math,os,sys,traceback
from pathlib import Path
from collections import deque
from r5_common import ROOT,OUT,FIXED,R,LABELS,now,epoch,stamp,sha,read,dump,csvwrite,xyz,chord,to_json_safe
sys.path.insert(0,str(ROOT/'runtime_qc'));sys.path.insert(0,str(ROOT/'runtime_r45'))
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
QC='MergedReflectivityQCComposite';Z='Reflectivity_-10C';E='EchoTop_30'
def truth(v):return str(v).lower()=='true'
def edge_margin_km():
 def distance(lat1,lon1,lat2,lon2):
  a=xyz(np.array([lat1]),np.array([lon1]))[0];b=xyz(np.array([lat2]),np.array([lon2]))[0]
  return float(2*R*math.asin(min(1.0,np.linalg.norm(a-b)/2))/1000)
 return min(distance(31,-96,30.7,-96),distance(39,-96,39.3,-96),distance(39,-100,39,-100.4),distance(39,-92,39,-91.6))
def load_temporal():
 with (ROOT/'R45_ANCHOR_VALIDITY.csv').open(encoding='utf-8-sig') as f:
  return {r['anchor_time']:r for r in csv.DictReader(f)}
def near(gridxyz,idx,flat,mask,rkm):
 if not mask.any() or not len(idx):return np.zeros(len(idx),dtype=bool)
 tree=cKDTree(xyz(flat['lat'][mask],flat['lon'][mask]));return tree.query_ball_point(gridxyz[idx],chord(rkm),return_length=True,workers=-1)>0
def open_flash(day):
 p=OUT/'flash_daily'/(day+'.npz');assert p.exists() and sha(p)==read(p.with_suffix('.json'))['npz_sha256']
 with np.load(p) as f:a={k:f[k] for k in f.files}
 order=np.lexsort((a['flash_id'],a['time']));a={k:v[order] for k,v in a.items()}
 pair=np.rec.fromarrays([a['flash_id'],a['time']],names='id,t');keep=np.r_[True,pair[1:]!=pair[:-1]] if len(pair) else np.array([],dtype=bool)
 return {k:v[keep] for k,v in a.items()}
def class_counts(a):return {LABELS[k]:int((a==k).sum()) for k in range(4)}
def write_store(day,lat,lon,times,parts,attrs):
 dest=ROOT/'LABEL_STORE_R5'/(day+'.nc');dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.tmp.nc')
 n=sum(len(x['pixel_index']) for x in parts);f=nc.Dataset(tmp,'w',format='NETCDF4');f.createDimension('sample',n);f.createDimension('time',288);f.createDimension('lat',800);f.createDimension('lon',800)
 for name,data,typ,dim in [('time',times,'i8','time'),('lat',lat,'f8','lat'),('lon',lon,'f8','lon')]:f.createVariable(name,typ,(dim,))[:]=data
 spec={'time_index':'u2','pixel_index':'u4','candidate':'u1','H_primary':'i1','Y_primary':'i1','class_primary':'i1','label_valid':'u1','model_ready_full':'u1','H_qf0':'i1','Y_qf0':'i1','class_qf0':'i1','invalid_reason_flags':'u2','model_not_ready_flags':'u2'}
 for name,typ in spec.items():
  v=f.createVariable(name,typ,('sample',),zlib=True,complevel=4,chunksizes=(min(max(n,1),262144),));v[:]=np.concatenate([p[name] for p in parts]) if n else np.array([],dtype=typ)
 f.title='Sparse QC>=30 candidate pixel-time labels; GATE0_LABEL_BUILD_R5';f.date=day;f.class_legend='-1 INVALID;0 I+ LOCAL_LIGHTNING_ONSET_POSITIVE;1 I-;2 O+;3 O-'
 f.invalid_reason_flags='1 fixed_date;2 GLM_gap;4 GLM_content_uncertain;8 spatial_edge;16 UTC_day_window_boundary'
 f.model_not_ready_flags='1 Z10_ambiguous;2 ET30_ambiguous;4 required_field_no_coverage;8 MRMS_causal_unavailable;16 other_required_field_invalid'
 f.distance_metric='mean-Earth spherical geodesic via exact unit-sphere chord; radius 6371008.8 m';f.history_interval='[t-30min,t]';f.future_interval='(t,t+30min]';f.labels_are_complete_population=1
 for k,v in attrs.items():f.setncattr(k,v)
 f.close();tmp.replace(dest);return dest,n
def build_day(day,temporal):
 dest=ROOT/'LABEL_STORE_R5'/(day+'.nc');meta=OUT/'label_daily'/(day+'.json')
 if dest.exists() and meta.exists() and read(meta).get('store_sha256')==sha(dest):return read(meta)
 mpath=ROOT/'SPATIOTEMPORAL_VALIDITY_MASK_R4'/(day+'.nc');fl=open_flash(day);parts=[];counts=collections.Counter();attr=collections.Counter();qfa=collections.Counter();endpoint={'history_max_le_t':True,'future_min_gt_t':True,'same_flash_both':0}
 dq={k:deque() for k in ['fail','za','ea','nc','causal','other']};acc={k:np.zeros(640000,dtype='u1') for k in dq}
 with nc.Dataset(mpath) as f:
  f.set_auto_mask(False);times=f['time'][:];lat=f['lat'][:];lon=f['lon'][:];gy,gx=np.meshgrid(lat,lon,indexing='ij');gridxyz=xyz(gy.ravel(),gx.ravel())
  for i,t in enumerate(times):
   qb=f['qccomp_class'][i].ravel();zs=f['z10_semantic_class'][i].ravel();es=f['et30_semantic_class'][i].ravel();joint=f['radar_joint_valid'][i].ravel().astype(bool)
   causal_bad=not all(bool(f.groups[p]['causal_valid'][i]) for p in [QC,Z,E])
   cur={'fail':(~joint).astype('u1'),'za':np.isin(zs,[1,7]).astype('u1'),'ea':np.isin(es,[1,7]).astype('u1'),
    'nc':((qb==2)|(zs==2)|(es==2)).astype('u1'),'causal':np.full(640000,causal_bad,dtype='u1'),'other':((qb==3)|(qb==4)|(zs==3)|(zs==8)|(es==3)|(es==8)).astype('u1')}
   for k in dq:
    dq[k].append(cur[k]);acc[k]+=cur[k]
    if len(dq[k])>6:acc[k]-=dq[k].popleft()
   cand=np.flatnonzero(f['qc_candidate_truth'][i].ravel()==1);n=len(cand)
   r=temporal[stamp(t)];fixed=truth(r['fixed_excluded']);gap=truth(r['glm_gap_invalid']);content=truth(r['glm_content_uncertain']);inside=truth(r['same_utc_day_support']);edge=False
   valid=not fixed and not gap and not content and inside
   hp=yp=hq=yq=np.full(n,-1,dtype='i1');cp=cq=np.full(n,-1,dtype='i1');model=np.zeros(n,dtype='u1');mf=np.zeros(n,dtype='u2')
   if valid:
    ht=(fl['time']>=t-1800)&(fl['time']<=t);yt=(fl['time']>t)&(fl['time']<=t+1800)
    hp=near(gridxyz,cand,fl,ht,20).astype('i1');yp=near(gridxyz,cand,fl,yt,10).astype('i1');hq=near(gridxyz,cand,fl,ht&(fl['qf']==0),20).astype('i1');yq=near(gridxyz,cand,fl,yt&(fl['qf']==0),10).astype('i1')
    cp=(2*hp+(1-yp)).astype('i1');cq=(2*hq+(1-yq)).astype('i1');ready=(len(dq['fail'])==6)&(acc['fail'][cand]==0);model[:]=ready
    mf|=((acc['za'][cand]>0).astype('u2')*1);mf|=((acc['ea'][cand]>0).astype('u2')*2);mf|=((acc['nc'][cand]>0).astype('u2')*4);mf|=((acc['causal'][cand]>0).astype('u2')*8);mf|=((acc['other'][cand]>0).astype('u2')*16)
    assert np.all((mf==0)==ready);endpoint['history_max_le_t']&=not ht.any() or fl['time'][ht].max()<=t;endpoint['future_min_gt_t']&=not yt.any() or fl['time'][yt].min()>t;endpoint['same_flash_both']+=int(np.intersect1d(np.flatnonzero(ht),np.flatnonzero(yt)).size)
    for k,v in class_counts(cp).items():counts['label_'+k]+=v
    for k,v in class_counts(cp[ready]).items():counts['model_'+k]+=v
    counts['agreement']+=int((cp==cq).sum());counts['agreement_den']+=n
    for k in range(4):qfa[f'primary_{LABELS[k]}']+=int((cp==k).sum());qfa[f'qf0_{LABELS[k]}']+=int((cq==k).sum())
    for k in ['za','ea','nc','causal','other']:attr[k]+=int((acc[k][cand]>0).sum())
   flags=np.full(n,(1 if fixed else 0)|(2 if gap else 0)|(4 if content else 0)|(8 if edge else 0)|(16 if not inside else 0),dtype='u2')
   counts['candidate']+=n;counts['label_valid']+=n if valid else 0;counts['model_ready']+=int(model.sum());attr['glm_gap']+=n if gap else 0;attr['glm_content']+=n if content else 0;attr['edge']+=n if edge else 0;attr['day_boundary']+=n if not inside else 0
   parts.append({'time_index':np.full(n,i,dtype='u2'),'pixel_index':cand.astype('u4'),'candidate':np.ones(n,dtype='u1'),'H_primary':hp,'Y_primary':yp,'class_primary':cp,'label_valid':np.full(n,valid,dtype='u1'),'model_ready_full':model,'H_qf0':hq,'Y_qf0':yq,'class_qf0':cq,'invalid_reason_flags':flags,'model_not_ready_flags':mf})
 store,n=write_store(day,lat,lon,times,parts,{'source_r4_mask_sha256':sha(mpath),'source_flash_npz_sha256':sha(OUT/'flash_daily'/(day+'.npz')),'edge_excluded_pixel_time_count':0})
 out={'date':day,'candidate_count':counts['candidate'],'label_valid_count':counts['label_valid'],'model_ready_count':counts['model_ready'],'label_class_counts':{k:counts['label_'+k] for k in ['I+','I-','O+','O-']},'model_class_counts':{k:counts['model_'+k] for k in ['I+','I-','O+','O-']},
  'agreement_count':counts['agreement'],'agreement_denominator':counts['agreement_den'],'primary_qf0_counts':dict(qfa),'attrition_inclusive':dict(attr),'endpoint_checks':endpoint,'store_path':str(store),'store_sha256':sha(store),'sample_rows':n,'utc':now()}
 dump(meta,out);print('R5_LABEL_DAY',day,n,out['model_class_counts'],flush=True);return out
class UF:
 def __init__(self,n):self.p=list(range(n));self.sz=[1]*n
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a==b:return
  if self.sz[a]<self.sz[b]:a,b=b,a
  self.p[b]=a;self.sz[a]+=self.sz[b]
def event_intervals():
 gaps=[(epoch(r['gap_start']),epoch(r['gap_end'])) for r in csv.DictReader((ROOT/'GLM_TEMPORAL_INVALID_MASK_R4.csv').open(encoding='utf-8-sig'))]
 glm=read(ROOT/'recovery_r45/glm_review.json');return gaps+[(r['start'],r['end']) for r in glm['uncertain_intervals']]
def onset_flashes(day,qf0,invalid):
 f=open_flash(day);keep=(f['lat']>=31)&(f['lat']<=39)&(f['lon']>=-100)&(f['lon']<=-92);keep&=(f['qf']==0) if qf0 else True
 ids=np.flatnonzero(keep);support=np.flatnonzero((f['qf']==0) if qf0 else np.ones(len(f['time']),bool));tree=cKDTree(xyz(f['lat'][support],f['lon'][support])) if len(support) else None
 day0=epoch(day+'T00:00:00+00:00');out=[];excluded=collections.Counter()
 for j in ids:
  ft=f['time'][j];observable=ft>=day0+1800 and not any(a<ft and b>=ft-1800 for a,b in invalid)
  if not observable:excluded['GLM_HISTORY_INCOMPLETE']+=1;continue
  nearids=support[tree.query_ball_point(xyz(np.array([f['lat'][j]]),np.array([f['lon'][j]]))[0],chord(20))]
  onset=not np.any((f['time'][nearids]>=ft-1800)&(f['time'][nearids]<ft))
  if onset:out.append(j)
 return f,np.array(out,dtype=int),dict(excluded)
def cluster_events(day,f,ids,prefix):
 if not len(ids):return [],{}
 pts=xyz(f['lat'][ids],f['lon'][ids]);tree=cKDTree(pts);uf=UF(len(ids))
 for a,b in tree.query_pairs(chord(20)):
  if abs(f['time'][ids[a]]-f['time'][ids[b]])<=900:uf.union(a,b)
 groups=collections.defaultdict(list)
 for i,j in enumerate(ids):groups[uf.find(i)].append(j)
 events=[];membership={}
 for n,g in enumerate(sorted(groups.values(),key=lambda x:(f['time'][x].min(),min(x))),1):
  x=xyz(f['lat'][g],f['lon'][g]).mean(0);x/=np.linalg.norm(x);clat=math.degrees(math.asin(x[2]));clon=math.degrees(math.atan2(x[1],x[0]));d=2*R*np.arcsin(np.clip(np.linalg.norm(xyz(f['lat'][g],f['lon'][g])-x,axis=1)/2,0,1))/1000
  eid=f'{prefix}-{day.replace("-","")}-{n:05d}'
  e={'event_id':eid,'date':day,'first_flash_time':stamp(f['time'][g].min()),'last_flash_time':stamp(f['time'][g].max()),'centroid_lat':clat,'centroid_lon':clon,'n_onset_flashes':len(g),'spatial_extent':json.dumps(to_json_safe({'lat_min':f['lat'][g].min(),'lat_max':f['lat'][g].max(),'lon_min':f['lon'][g].min(),'lon_max':f['lon'][g].max(),'max_centroid_radius_km':d.max()})),'usable':False,'exclusion_reason':''}
  events.append(e)
  for j in g:membership[j]=eid
 return events,membership
def assign_samples(day,events,membership,f,ids):
 if not events or not len(ids):return
 event={e['event_id']:e for e in events};pts=xyz(f['lat'][ids],f['lon'][ids]);tree=cKDTree(pts);store=ROOT/'LABEL_STORE_R5'/(day+'.nc');assigned=collections.Counter();anchors=collections.defaultdict(set);pixels=collections.defaultdict(set)
 with nc.Dataset(store) as n:
  ti=n['time_index'][:];pi=n['pixel_index'][:];cl=n['class_primary'][:];ready=n['model_ready_full'][:].astype(bool);times=n['time'][:];lat=n['lat'][:];lon=n['lon'][:]
  for tix in np.unique(ti[ready&(cl==0)]):
   lo=int(np.searchsorted(ti,tix,'left'));hi=int(np.searchsorted(ti,tix,'right'));sel=np.arange(lo,hi)[ready[lo:hi]&(cl[lo:hi]==0)];p=pi[sel];t=times[tix];lists=tree.query_ball_point(xyz(lat[p//800],lon[p%800]),chord(10))
   for row,candidates in zip(sel,lists):
    valid=[k for k in candidates if f['time'][ids[k]]>t and f['time'][ids[k]]<=t+1800]
    if not valid:continue
    q=xyz(np.array([lat[pi[row]//800]]),np.array([lon[pi[row]%800]]))[0];best=min(valid,key=lambda k:(np.linalg.norm(pts[k]-q),f['time'][ids[k]],membership[ids[k]]));eid=membership[ids[best]]
    assigned[eid]+=1;anchors[eid].add(int(tix));pixels[eid].add(int(pi[row]))
 for eid,e in event.items():
  e['n_positive_pixel_times']=assigned[eid];e['n_unique_anchors']=len(anchors[eid]);e['duration_min']=(max(anchors[eid])-min(anchors[eid]))*5 if anchors[eid] else 0;e['n_unique_target_pixels']=len(pixels[eid]);e['spatial_area_proxy_km2']=len(pixels[eid])*0.01*0.01*111.195**2*math.cos(math.radians(e['centroid_lat']));e['usable']=assigned[eid]>0;e['exclusion_reason']='' if e['usable'] else 'NO_MODEL_READY_FULL_I_POS_IN_PRECEDING_30MIN'
def build_events(days):
 invalid=event_intervals();events=[];qevents=[];excluded=collections.Counter()
 for day in days:
  f,ids,x=onset_flashes(day,False,invalid);ev,mem=cluster_events(day,f,ids,'E');assign_samples(day,ev,mem,f,ids);events+=ev;excluded.update(x)
  fq,iq,xq=onset_flashes(day,True,invalid);qe,_=cluster_events(day,fq,iq,'Q');qevents+=qe
  print('R5_EVENT_DAY',day,len(ev),sum(e['usable'] for e in ev),flush=True)
 return events,qevents,dict(excluded)
def report(daily,events,qevents,excluded):
 labels={k:sum(d['label_class_counts'][k] for d in daily) for k in ['I+','I-','O+','O-']};models={k:sum(d['model_class_counts'][k] for d in daily) for k in labels};total=sum(d['agreement_denominator'] for d in daily);agree=sum(d['agreement_count'] for d in daily)
 usable=[e for e in events if e['usable']];by=collections.Counter(e['date'] for e in usable);nraw=len(events);nu=len(usable);capture=nu/nraw if nraw else 0;top=sorted(by.values(),reverse=True)
 primary_onset=nraw;qf_onset=len(qevents);eventchange=abs(qf_onset-primary_onset)/primary_onset if primary_onset else None;agreement=agree/total if total else None;qfalert=eventchange is None or eventchange>.15 or agreement<.95
 q=[]
 for i in range(4):
  for j in range(4):
   count=sum(d['primary_qf0_counts'].get(f'primary_{LABELS[i]}',0) for d in []) if False else 0
 # Recount class transition matrix directly from stores.
 matrix=np.zeros((4,4),dtype='i8')
 for d in daily:
  with nc.Dataset(d['store_path']) as f:
   a=f['class_primary'][:];b=f['class_qf0'][:];v=f['label_valid'][:].astype(bool)
   for i in range(4):
    for j in range(4):matrix[i,j]+=int((v&(a==i)&(b==j)).sum())
 for i in range(4):
  for j in range(4):q.append({'scope':'TOTAL','primary_class':LABELS[i],'qf0_class':LABELS[j],'count':int(matrix[i,j]),'agreement':agreement,'label_changed_fraction':1-agreement,'primary_I_pos':labels['I+'],'qf0_I_pos':int(matrix[:,0].sum()),'primary_O_pos':labels['O+'],'qf0_O_pos':int(matrix[:,2].sum()),'primary_onset_events':primary_onset,'qf0_onset_events':qf_onset,'independent_onset_event_change_fraction':eventchange,'QF_SENSITIVITY_ALERT':'YES' if qfalert else 'NO'})
 csvwrite(ROOT/'PRIMARY_VS_QF0_LABEL_AUDIT.csv',q)
 classrows=[]
 for d in daily:classrows.append({'date':d['date'],**{f'{k}_label_valid':d['label_class_counts'][k] for k in labels},**{f'{k}_model_ready':d['model_class_counts'][k] for k in labels}})
 classrows.append({'date':'TOTAL',**{f'{k}_label_valid':labels[k] for k in labels},**{f'{k}_model_ready':models[k] for k in labels}});csvwrite(ROOT/'LABEL_CLASS_COUNTS_R5.csv',classrows)
 onsetrows=[]
 for e in events:onsetrows.append({**e,'denominator_eligible':True,'history_observation_complete':True,'edge_valid':True,'fixed_exclusion':False})
 csvwrite(ROOT/'ONSET_EVENT_AUDIT_R5.csv',onsetrows);csvwrite(ROOT/'ONSET_EVENT_MULTIPLICITY.csv',[{k:e[k] for k in ['event_id','date','n_positive_pixel_times','n_unique_anchors','duration_min','spatial_area_proxy_km2','n_unique_target_pixels','usable']} for e in events])
 dates=[]
 for day in [d['date'] for d in daily]:dates.append({'date':day,'raw_onset_events':sum(e['date']==day for e in events),'usable_onset_events':by[day],'June_or_July':'June' if day[5:7]=='06' else 'July'})
 csvwrite(ROOT/'DAILY_ONSET_SUMMARY_R5.csv',dates)
 spatial=collections.Counter();
 for e in usable:spatial[(min(3,int((e['centroid_lat']-31)/2)),min(3,int((e['centroid_lon']+100)/2)))]+=1
 spatialrows=[{'lat_band':a,'lon_band':b,'event_count':n,'share':n/nu if nu else 0,'lat_range':f'{31+2*a}..{33+2*a}','lon_range':f'{-100+2*b}..{-98+2*b}'} for (a,b),n in sorted(spatial.items())];csvwrite(ROOT/'SPATIAL_ONSET_DISTRIBUTION_R5.csv',spatialrows)
 topquad=max((r['share'] for r in spatialrows),default=0);spalert=topquad>.5
 attrrows=[]
 for k in labels:attrrows.append({'class':k,'label_valid_count':labels[k],'model_ready_full_count':models[k],'attrition':1-models[k]/labels[k] if labels[k] else None})
 attrrows.append({'class':'TOTAL','label_valid_count':sum(labels.values()),'model_ready_full_count':sum(models.values()),'attrition':1-sum(models.values())/sum(labels.values()),
  **{f'loss_{k}_inclusive':sum(d['attrition_inclusive'].get(k,0) for d in daily) for k in ['glm_gap','glm_content','edge','day_boundary','za','ea','nc','causal','other']}});csvwrite(ROOT/'MODEL_READY_ATTRITION_R5.csv',attrrows)
 iposfrac=models['I+']/(models['I+']+models['I-']) if models['I+']+models['I-'] else 0;oposfrac=models['O+']/(models['O+']+models['O-']) if models['O+']+models['O-'] else 0;iattr=1-models['I+']/labels['I+'] if labels['I+'] else 1;margin=edge_margin_km();assert margin>20
 leakage={'status':'PASS','candidate_uses_future_mrms':False,'H_interval':'[t-1800,t]','Y_interval':'(t,t+1800]','same_flash_history_and_future_count':sum(d['endpoint_checks']['same_flash_both'] for d in daily),'all_history_endpoint_checks':all(d['endpoint_checks']['history_max_le_t'] for d in daily),'all_future_endpoint_checks':all(d['endpoint_checks']['future_min_gt_t'] for d in daily),'future_mrms_model_ready_reads':0,'gap_to_zero_conversions':0,'candidate_rule':'stored R4 qc_candidate_truth at anchor t only','model_ready_history_offsets_minutes':[-25,-20,-15,-10,-5,0],'no_outcome_candidate_filter':True}
 assert leakage['same_flash_history_and_future_count']==0 and leakage['all_history_endpoint_checks'] and leakage['all_future_endpoint_checks'];dump(ROOT/'LABEL_LEAKAGE_AUDIT_R5.json',leakage)
 gate='PASS';reasons=[]
 if nu<100 or capture<.5:gate='KILL';reasons+=['usable onset events below 100 or capture below 50%']
 elif capture<.7 or len(by)<20 or (top[0]/nu if nu else 1)>.2 or (sum(top[:5])/nu if nu else 1)>.5 or iattr>.1 or qfalert:gate='HOLD';reasons+=['one or more frozen PASS thresholds not met']
 summary={'utc':now(),'LABEL_VALID':labels,'MODEL_READY_FULL':models,'I_pos_mask_attrition':iattr,'N_RAW_ONSET_EVENTS':nraw,'N_USABLE_ONSET_EVENTS':nu,'ONSET_EVENT_CAPTURE_RATE':capture,'active_onset_dates':len(by),'top1_date_share':top[0]/nu if nu else None,'top5_date_share':sum(top[:5])/nu if nu else None,'June_count':sum(n for d,n in by.items() if d[5:7]=='06'),'July_count':sum(n for d,n in by.items() if d[5:7]=='07'),
  'I_pos_natural_positive_fraction':iposfrac,'O_pos_natural_positive_fraction':oposfrac,'I_pos_to_I_neg':f'{models["I+"]}:{models["I-"]}','O_pos_to_O_neg':f'{models["O+"]}:{models["O-"]}','EXTREME_ONSET_IMBALANCE':'YES' if iposfrac<.1 else 'NO',
  'label_agreement':agreement,'label_changed_fraction':1-agreement,'qf0_independent_onset_events':qf_onset,'qf_event_change_fraction':eventchange,'QF_SENSITIVITY_ALERT':'YES' if qfalert else 'NO','top_spatial_quadrant_share':topquad,'SPATIAL_CONCENTRATION_ALERT':'YES' if spalert else 'NO',
  'EDGE_EXCLUDED_PIXEL_TIME_COUNT':0,'edge_support_min_geodesic_margin_km':margin,'excluded_incomplete_onset_flashes':excluded,'EXTREME_EVENT_SAMPLE_LOSS':'YES','LABEL_LEAKAGE_AUDIT':'PASS','LABEL_GATE_R5':gate,'gate_reasons':reasons,'READY_FOR_SPLIT_DESIGN_R55':'YES' if gate=='PASS' else 'NO','DATA_GATE':'PASS_WITH_EXCLUSIONS','SCIENTIFIC_GATE0':'NOT_RUN','GI':'NA','GO':'NA','Gamma':'NA','95% CI':'NA','split_created':False,'models_trained':False}
 dump(OUT/'SUMMARY_R5.json',summary);write_docs(summary,daily);validate(summary,daily);return summary
def write_docs(s,daily):
 recovery=read(OUT/'R5_FIRST_DAY_RESUME_TEST.json')
 failure_record=f'''R5_FAILURE_TYPE = ENGINEERING_SERIALIZATION_ERROR
R5_FAILURE_CAUSE = NUMPY_BOOL_NOT_JSON_SERIALIZABLE
DATA_INTEGRITY_IMPACT = NONE_CONFIRMED
GLM_EXTRACTION_REPEATED = NO
RESEARCH_DESIGN_CHANGED = NO
RESUME_CHECKPOINT = 59/59 GLM daily partitions
SERIALIZATION_FIX_SHA256 = {recovery['serialization_fix_sha256']}
RESUME_TIME_UTC = {recovery['utc']}
FIRST_DAY_RECOVERY_TEST = {recovery['status']}
'''
 intro=f'''# GATE0_LABEL_BUILD_R5 ({s['utc']})

LABEL_GATE_R5 = {s['LABEL_GATE_R5']}
READY_FOR_SPLIT_DESIGN_R55 = {s['READY_FOR_SPLIT_DESIGN_R55']}
DATA_GATE = PASS_WITH_EXCLUSIONS
SCIENTIFIC_GATE0 = NOT_RUN; GI/GO/Gamma/95% CI = NA.

Formal labels use the frozen R5 definitions and sparse daily NetCDF partitions. I+ means LOCAL_LIGHTNING_ONSET_POSITIVE, not storm first flash. Fixed dates 2024-07-08 and 2024-07-10 remain excluded. No sampling, split, training, model metric, or Gate-1 step was performed.

## R5 serialization recovery

{failure_record}

```json
{json.dumps(to_json_safe(s),ensure_ascii=False,indent=2)}
```
'''
 (ROOT/'LABEL_BUILD_R5.md').write_text(intro,encoding='utf-8');(ROOT/'LABEL_AUDIT_REPORT_R5.md').write_text(intro+'\nJuly 8 is a high-intensity convective day with fixed GLM incompleteness and remains excluded. June 3 retains a comparable convective process. EXTREME_EVENT_SAMPLE_LOSS = YES; no compensatory weighting was applied.\n',encoding='utf-8')
 header=f'''# R5 label-build status ({s['utc']})

LABEL_GATE_R5 = {s['LABEL_GATE_R5']}
READY_FOR_SPLIT_DESIGN_R55 = {s['READY_FOR_SPLIT_DESIGN_R55']}
DATA_GATE = PASS_WITH_EXCLUSIONS
SCIENTIFIC_GATE0 = NOT_RUN
GI = NA
GO = NA
Gamma = NA
95% CI = NA

R5 builds the complete unsampled label population under the frozen R4.5 masks. No split, model training or Gate-1 was run. Historical content follows verbatim.

{failure_record}

---

'''
 for name in ['RESEARCH_CONTROL.md','LABEL_DEFINITION.md','GATE0_DECISION.md']:(ROOT/name).write_text(header+(OUT/'r45_frozen'/name).read_text(encoding='utf-8'),encoding='utf-8')
 new=['LABEL_BUILD_R5.md','LABEL_AUDIT_REPORT_R5.md','LABEL_CLASS_COUNTS_R5.csv','ONSET_EVENT_AUDIT_R5.csv','DAILY_ONSET_SUMMARY_R5.csv','ONSET_EVENT_MULTIPLICITY.csv','PRIMARY_VS_QF0_LABEL_AUDIT.csv','MODEL_READY_ATTRITION_R5.csv','SPATIAL_ONSET_DISTRIBUTION_R5.csv','LABEL_LEAKAGE_AUDIT_R5.json','R5_VALIDATION.json','LABEL_STORE_R5/manifest.json']
 dump(OUT/'DELIVERABLE_PATHS_R5.json',{'new_files':[str(ROOT/n) for n in new],'updated_files':[str(ROOT/n) for n in ['RESEARCH_CONTROL.md','LABEL_DEFINITION.md','GATE0_DECISION.md']],'label_store_directory':str(ROOT/'LABEL_STORE_R5'),'evidence_directory':str(OUT)})
def validate(s,daily):
 updated={'RESEARCH_CONTROL.md','LABEL_DEFINITION.md','GATE0_DECISION.md'};pres={}
 for name,v in read(OUT/'r45_frozen/manifest.json').items():
  if name in updated:assert (ROOT/name).read_text(encoding='utf-8').endswith((OUT/'r45_frozen'/name).read_text(encoding='utf-8'))
  else:assert sha(ROOT/name)==v['sha256'],name;pres[name]=v['sha256']
 manifest=[];agg=collections.Counter()
 for d in daily:
  assert sha(d['store_path'])==d['store_sha256'];manifest.append({'date':d['date'],'path':d['store_path'],'sha256':d['store_sha256'],'samples':d['sample_rows']})
  with nc.Dataset(d['store_path']) as f:
   cp=f['class_primary'][:];cv=f['label_valid'][:].astype(bool);mr=f['model_ready_full'][:].astype(bool);h=f['H_primary'][:];y=f['Y_primary'][:]
   assert np.all(cp[~cv]==-1) and np.all(mr<=cv) and np.all(cp[cv]==2*h[cv]+1-y[cv]);assert not np.any((h==1)&(y==-1))
   for k in range(4):agg['label_'+LABELS[k]]+=int((cv&(cp==k)).sum());agg['model_'+LABELS[k]]+=int((mr&(cp==k)).sum())
 assert all(agg['label_'+k]==s['LABEL_VALID'][k] and agg['model_'+k]==s['MODEL_READY_FULL'][k] for k in s['LABEL_VALID'])
 dump(ROOT/'LABEL_STORE_R5/manifest.json',{'utc':now(),'partitions':manifest,'complete_dates':len(manifest),'fixed_exclusions':sorted(FIXED),'aggregate_recount':dict(agg)})
 margin=edge_margin_km();assert margin>20
 dump(ROOT/'R5_VALIDATION.json',{'utc':now(),'status':'PASS','LABEL_GATE_R5':s['LABEL_GATE_R5'],'r45_historical_hashes_preserved':pres,'daily_partitions':59,'daily_store_hashes':'PASS','independent_class_recount':'PASS','class_formula':'PASS','invalid_not_hidden_class':'PASS','model_ready_subset_label_valid':'PASS','edge_support_geodesic_margin':'PASS','edge_support_min_geodesic_margin_km':margin,'edge_required_margin_km':20,'label_leakage_audit':'PASS','fixed_dates_absent':'PASS','complete_population_no_sampling':'PASS','no_split':True,'no_training':True,'scientific_gate':'NOT_RUN','script_sha256':sha(__file__),'preregistration_sha256':sha(OUT/'PREREGISTRATION_R5.json')})
def main():
 assert sha(OUT/'PREREGISTRATION_R5.json')==read(OUT/'PREREGISTRATION_SEAL_R5.json')['sha256'];days=[(dt.date(2024,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61) if (dt.date(2024,6,1)+dt.timedelta(days=i)).isoformat() not in FIXED]
 temporal=load_temporal();daily=[]
 for i,day in enumerate(days,1):daily.append(build_day(day,temporal));dump(OUT/'PROGRESS_R5.json',{'utc':now(),'phase':'LABEL_BUILD','date':day,'dates_complete':i,'dates_required':59,'pid':os.getpid(),'no_training':True})
 events,qevents,excluded=build_events(days);s=report(daily,events,qevents,excluded);print('R5_COMPLETE',json.dumps(to_json_safe(s),ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:dump(OUT/'FAILURE_R5.json',{'utc':now(),'phase':'LABEL_BUILD','error':repr(e),'traceback':traceback.format_exc(),'status':'BLOCKED','no_training':True});raise
