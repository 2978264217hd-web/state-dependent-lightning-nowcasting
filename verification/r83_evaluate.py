"""R83 frozen confirmatory evaluation: runs ONLY after61 prediction partitions seal."""
import gc,math,time,traceback,types
from r83_common import *
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
MODELS=('R6_GLOBAL_FUSION','R7_STATE_CALIBRATED_FUSION');SCOPES=('ALL','I','O')
ENV={'np':np,'ctypes':ctypes};REG=pure('scripts/r6_baseline.py',['pr_auc','metrics','peak_rss'],ENV);ENV['b']=types.SimpleNamespace(pr_auc=ENV['pr_auc'],metrics=ENV['metrics']);REG.update(pure('scripts/r7_minimal_bias_resolution.py',['sigmoid','logit','apply_cal','state_metrics','ece'],ENV))
NORM,STATE,GT=config();TI=STATE['thresholds']['I'];TO=STATE['thresholds']['O']

def calibrated(p,h):
 q=np.empty(len(p),'f4')
 for state,v in [('I',0),('O',1)]:m=h==v;q[m]=ENV['apply_cal'](p[m],STATE['calibration'][state])
 return q

def statrow(day,model,scope,y,routing_h,p,eval_h):
 select=np.ones(len(y),bool) if scope=='ALL' else eval_h==(0 if scope=='I' else 1);yy=y[select].astype('u1');pp=p[select];hh=routing_h[select];n=len(yy)
 row=ENV['metrics'](yy,pp,GT) if model==MODELS[0] else ENV['state_metrics'](yy,hh,pp,TI,TO)
 row.update(date=day,model=model,scope=scope,ECE=ENV['ece'](yy,pp),mean_prediction_probability=float(pp.mean(dtype='f8')),TN=n-row['TP']-row['FP']-row['FN'],positive_count=int(yy.sum()),squared_error_sum=float(((pp-yy)**2).sum(dtype='f8')))
 bins=[]
 for i in range(15):
  m=(pp>=i/15)&((pp<(i+1)/15) if i<14 else (pp<=1));bins.append([int(m.sum()),float(pp[m].sum(dtype='f8')),int(yy[m].sum())])
 row['bins']=bins;return row

def sorted_ap(path,totals,qf=False):
 counts={s:0 for s in SCOPES};tp={s:0 for s in SCOPES};area={s:0. for s in SCOPES};last_score=math.inf;last_sid=2**64-1
 for batch in pq.ParquetFile(path).iter_batches(batch_size=500000):
  a={k:batch.column(k).to_numpy() for k in batch.schema.names};p=a['p'];sid=a['sample_id'];y=a['Yq'] if qf else a['Y'];h=a['Hq'] if qf else a['H']
  assert len(p)==0 or p[0]<last_score or (p[0]==last_score and sid[0]<last_sid)
  if len(p):last_score=float(p[-1]);last_sid=int(sid[-1])
  assert np.all((p[1:]<p[:-1])|((p[1:]==p[:-1])&(sid[1:]<sid[:-1])))
  for scope in SCOPES:
   keep=np.ones(len(y),bool) if scope=='ALL' else h==(0 if scope=='I' else 1);yy=y[keep].astype(bool);n=len(yy);pos=totals[scope]['positive_count']
   if not n:continue
   cumulative=tp[scope]+np.cumsum(yy,dtype='i8');ranks=counts[scope]+np.arange(1,n+1,dtype='i8')
   if pos:
    rec=cumulative/pos;area[scope]+=float(np.sum((rec-np.r_[tp[scope]/pos,rec[:-1]])*(cumulative/ranks)))
   tp[scope]+=int(yy.sum());counts[scope]+=n
 for s in SCOPES:assert counts[s]==totals[s]['n'] and tp[s]==totals[s]['positive_count']
 return {s:area[s] if totals[s]['positive_count'] else float('nan') for s in SCOPES}

def regression():
 # Synthetic fixtures only, before inspecting external metrics. Stable tied rank exactness.
 p=np.array([.2,.8,.8,.4,.2,.9,.4,.8],'f4');y=np.array([0,1,0,1,1,0,0,1],'u1');h=np.array([0,0,1,1,0,1,1,0],'u1');sid=np.arange(8,dtype='u8');order=np.lexsort((-sid.astype('i8'),-p.astype('f8')))
 fixture=OUT/'regression_tied_scores.parquet';pq.write_table(pa.table({'sample_id':sid[order],'p':p[order],'H':h[order],'Y':y[order],'Hq':h[order],'Yq':y[order]}),fixture)
 totals={s:{'n':int((np.ones(len(y),bool) if s=='ALL' else h==(0 if s=='I' else 1)).sum()),'positive_count':int(y[np.ones(len(y),bool) if s=='ALL' else h==(0 if s=='I' else 1)].sum())} for s in SCOPES};ap=sorted_ap(fixture,totals)
 for s in SCOPES:
  m=np.ones(len(y),bool) if s=='ALL' else h==(0 if s=='I' else 1);assert abs(ap[s]-ENV['pr_auc'](y[m],p[m]))<1e-14
 assert np.array_equal(calibrated(p,h)[h==0],ENV['apply_cal'](p[h==0],STATE['calibration']['I']))
 dump(OUT/'R83_METRIC_IMPLEMENTATION_REGRESSION.json',{'utc':now(),'status':'PASS','synthetic_only':True,'frozen_function_hashes':REG,'exact_stable_tie_order_verified':True,'no_threshold_search':True,'no_calibration_fit':True,'duckdb_version':duckdb.__version__,'pyarrow_version':pa.__version__,'external_sort_docs':['https://duckdb.org/docs/current/clients/python/reference/','https://duckdb.org/2024/07/09/memory-management']})

def prepare(freeze):
 directory=OUT/'evaluation_prepared';directory.mkdir(exist_ok=True);(OUT/'evaluation_daily').mkdir(exist_ok=True)
 for m in freeze['prediction_partitions']:
  day=m['date'];dest=directory/(day+'.parquet');mp=OUT/'evaluation_daily'/(day+'.prepared.json');assert sha(m['prediction_path'])==m['prediction_sha256']
  if dest.exists() and mp.exists():assert sha(dest)==read(mp)['sha256'];continue
  diskcheck(m['samples']*24);tmp=dest.with_name(dest.name+'.tmp');writer=None
  for b in pq.ParquetFile(m['prediction_path']).iter_batches(batch_size=250000):
   a={k:b.column(k).to_numpy() for k in ['sample_id','H','Y','QF0_H','QF0_Y','ensemble_probability']};p=a['ensemble_probability'];pc=calibrated(p,a['H']);table=pa.table({'sample_id':a['sample_id'],'p_global':p,'p_state':pc,'H':a['H'],'Y':a['Y'],'Hq':a['QF0_H'],'Yq':a['QF0_Y']})
   if writer is None:writer=pq.ParquetWriter(tmp,table.schema,compression='zstd')
   writer.write_table(table)
  writer.close();assert pq.ParquetFile(tmp).metadata.num_rows==m['samples'];os.replace(tmp,dest);dump(mp,{'utc':now(),'status':'PASS','sha256':sha(dest),'source_prediction_sha256':m['prediction_sha256'],'calibration_hash':sha(ROOT/'R7_METHOD_FREEZE.json'),'labels_unchanged':True})
 (OUT/'rankings').mkdir(exist_ok=True);(OUT/'tmp_sort').mkdir(exist_ok=True);con=duckdb.connect();con.execute("SET memory_limit='2GB'");con.execute('SET threads=4');con.execute("SET temp_directory='"+str(OUT/'tmp_sort').replace('\\','/')+"'")
 for model,key in zip(MODELS,('p_global','p_state')):
  path=OUT/'rankings'/(model+'.parquet');mp=OUT/'rankings'/(model+'.json')
  if path.exists() and mp.exists():assert sha(path)==read(mp)['sha256'];continue
  diskcheck(15*2**30);tmp=path.with_name(path.name+'.tmp');src=str(directory/'*.parquet').replace('\\','/');target=str(tmp).replace('\\','/')
  sql=f"COPY (SELECT sample_id,{key} AS p,H,Y,Hq,Yq FROM read_parquet('{src}') ORDER BY {key} DESC,sample_id DESC) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)"
  con.execute(sql);assert pq.ParquetFile(tmp).metadata.num_rows==freeze['samples'];os.replace(tmp,path);dump(mp,{'utc':now(),'status':'PASS','sha256':sha(path),'rows':freeze['samples'],'exact_R6_order':'p DESC, sample_id DESC for reversed stable mergesort ties','memory_limit':'2GB','no_approximation':True});print('R83_EXACT_RANK_COMPLETE',model,flush=True)
 con.close()

def evaluate(qf,freeze):
 prefix='qf' if qf else 'primary';allrows=[]
 for day in DAYS:
  mp=OUT/'evaluation_daily'/(day+'.'+prefix+'.json');source=OUT/'evaluation_prepared'/(day+'.parquet')
  if mp.exists():allrows.extend(read(mp)['rows']);continue
  table=pq.read_table(source);a={k:table.column(k).to_numpy() for k in table.schema.names};y=a['Yq'] if qf else a['Y'];h=a['Hq'] if qf else a['H'];rr=[]
  for model,key in zip(MODELS,('p_global','p_state')):
   for scope in SCOPES:rr.append(statrow(day,model,scope,y,a['H'],a[key],h))
  dump(mp,{'utc':now(),'status':'PASS','source_sha256':sha(source),'rows':rr,'prediction_freeze_hash':sha(OUT/'R83_PREDICTION_FREEZE.json')});allrows.extend(rr);del table,a;gc.collect()
 aggregate={};totals={}
 for model in MODELS:
  totals[model]={}
  for scope in SCOPES:
   rr=[r for r in allrows if r['model']==model and r['scope']==scope];n=sum(r['n'] for r in rr);tp=sum(r['TP'] for r in rr);fp=sum(r['FP'] for r in rr);fn=sum(r['FN'] for r in rr);tn=sum(r['TN'] for r in rr);positive=sum(r['positive_count'] for r in rr);bins=np.sum([r['bins'] for r in rr],axis=0);ece=sum(abs(x[1]-x[2])/n for x in bins if x[0]);row={'model':model,'scope':scope,'n':n,'positive_rate':positive/n,'positive_count':positive,'TP':tp,'FP':fp,'FN':fn,'TN':tn,'CSI':tp/max(tp+fp+fn,1),'POD':tp/max(tp+fn,1),'FAR':fp/max(tp+fp,1),'F1':2*tp/max(2*tp+fp+fn,1),'Brier':sum(r['squared_error_sum'] for r in rr)/n,'ECE':ece,'mean_prediction_probability':sum(r['mean_prediction_probability']*r['n'] for r in rr)/n,'threshold_policy':'R6_GLOBAL' if model==MODELS[0] else 'FROZEN_R7_PRIMARY_H_ROUTING','probability_source':'same frozen R6 ensemble; R7 fixed Platt only for STATE'};totals[model][scope]=row
  ap=sorted_ap(OUT/'rankings'/(model+'.parquet'),totals[model],qf)
  for scope in SCOPES:totals[model][scope]['PR_AUC']=ap[scope];aggregate[(model,scope)]=totals[model][scope]
 writecsv(OUT/('R83_QF_SENSITIVITY_RESULTS.csv' if qf else 'R83_PRIMARY_RESULTS.csv'),aggregate.values())
 writecsv(OUT/('R83_QF_DATE_LEVEL_RESULTS.csv' if qf else 'R83_DATE_LEVEL_RESULTS.csv'),[{k:v for k,v in r.items() if k not in ('bins','squared_error_sum')} for r in allrows])
 return allrows,aggregate

def bootstrap(allrows,qf):
 name='R83_QF_BOOTSTRAP.csv' if qf else 'R83_BOOTSTRAP.csv';rng=np.random.default_rng(20260910);counts=np.zeros((61,2,2,3),'i8')
 for d,day in enumerate(DAYS):
  for mi,model in enumerate(MODELS):
   for si,scope in enumerate(('I','O')):
    r=next(r for r in allrows if r['date']==day and r['model']==model and r['scope']==scope);counts[d,mi,si]=[r['TP'],r['FP'],r['FN']]
 out=[]
 for k in range(1000):
  ix=rng.integers(0,61,size=61);x=counts[ix].sum(axis=0);c=x[:,:,0]/np.maximum(x.sum(axis=2),1);out.append({'replicate':k,'DELTA_I_R83':float(c[1,0]-c[0,0]),'DELTA_O_R83':float(c[1,1]-c[0,1]),'sampled_UTC_date_codes':'|'.join(map(str,ix)),'unit':'complete UTC date','dates_drawn':61})
 writecsv(OUT/name,out);ci={key:np.percentile([r[key] for r in out],[2.5,97.5]).tolist() for key in ('DELTA_I_R83','DELTA_O_R83')}
 dump(OUT/('R83_QF_BOOTSTRAP_SUMMARY.json' if qf else 'R83_BOOTSTRAP_SUMMARY.json'),{'utc':now(),'status':'PASS','replicates':1000,'unit':'61 complete UTC dates sampled with replacement; confusion counts are sufficient statistics for full frozen date replication','seed':20260910,'CI95':ci,'bootstrap_sha256':sha(OUT/name),'no_pixel_bootstrap':True});return ci

def domain(results):
 old=csvrows(ROOT/'R7_TEST_RESULTS.csv');rows=[]
 with np.load(ROOT/'predictions_r6/test_metadata_r6.npz',allow_pickle=False) as z:history_h=z['h'].copy()
 source={MODELS[0]:('R6_M_FUSION',ROOT/'predictions_r6/M_FUSION_ensemble_test_probability.npy'),MODELS[1]:('STATE_CALIBRATED_FUSION',ROOT/'predictions_r7/test_STATE_CALIBRATED_FUSION.npy')}
 for model,(oldname,path) in source.items():
  p=np.load(path,mmap_mode='r');assert len(p)==len(history_h)
  for scope in SCOPES:
   r=next(r for r in old if r['model']==oldname and r['scope']==scope);new=results[(model,scope)];total=0.;count=0
   for lo in range(0,len(p),500000):
    h=history_h[lo:lo+500000];m=np.ones(len(h),bool) if scope=='ALL' else h==(0 if scope=='I' else 1);v=p[lo:lo+500000][m];total+=float(v.sum(dtype='f8'));count+=len(v)
   for metric,oldvalue,newvalue in [('positive_fraction',float(r['positive_rate']),new['positive_rate']),('mean_prediction_probability',total/count,new['mean_prediction_probability']),('Brier',float(r['Brier']),new['Brier']),('ECE',float(r['ECE']),new['ECE']),('sample_count',int(r['n']),new['n'])]:rows.append({'model':model,'scope':scope,'metric':metric,'year2024':oldvalue,'year2025':newvalue,'shift2025_minus2024':newvalue-oldvalue,'reference':'frozen 2024 R55 TEST/R7 results; full61-date2025 confirmation','interpretation':'combined operational platform/year/weather/sampling shift; cannot isolate spacecraft effect; no recalibration'})
  del p
 split={r['date']:r['split'] for r in csvrows(ROOT/'SPLIT_ASSIGNMENT_R55/date_assignment.csv')};testdays={d for d,s in split.items() if s=='TEST'};raw=csvrows(ROOT/'ONSET_EVENT_AUDIT_R5.csv');ready=csvrows(ROOT/'R52_ONSET_EVENT_AUDIT.csv');rawcount=sum(r['date'] in testdays for r in raw);usablecount=sum(r['date'] in testdays and str(r['usable_r52']).lower()=='true' for r in ready)
 rows.extend([{'model':'SET','scope':'ALL','metric':'UTC_dates','year2024':len(testdays),'year2025':61,'shift2025_minus2024':61-len(testdays),'reference':'frozen2024TEST versus2025fullperiod','interpretation':'different period lengths and natural distributions'}, {'model':'SET','scope':'ALL','metric':'raw_onset_events','year2024':rawcount,'year2025':7230,'shift2025_minus2024':7230-rawcount,'reference':'R5 frozen2024TEST','interpretation':'unadjusted event totals; exposure differs'}, {'model':'SET','scope':'ALL','metric':'usable_onset_events','year2024':usablecount,'year2025':6124,'shift2025_minus2024':6124-usablecount,'reference':'R52 frozen2024TEST','interpretation':'R52 maskable; unadjusted exposure differs'}]);writecsv(OUT/'R83_DOMAIN_SHIFT_AUDIT.csv',rows);return rows

def finish(primary,pci,qf,qci,domainrows,started):
 di=primary[(MODELS[1],'I')]['CSI']-primary[(MODELS[0],'I')]['CSI'];do=primary[(MODELS[1],'O')]['CSI']-primary[(MODELS[0],'O')]['CSI'];dip=qf[(MODELS[1],'I')]['CSI']-qf[(MODELS[0],'I')]['CSI'];dop=qf[(MODELS[1],'O')]['CSI']-qf[(MODELS[0],'O')]['CSI'];iloss=primary[(MODELS[0],'I')]['PR_AUC']-primary[(MODELS[1],'I')]['PR_AUC'];oloss=primary[(MODELS[0],'O')]['PR_AUC']-primary[(MODELS[1],'O')]['PR_AUC']
 strong=di>=.04 and pci['DELTA_I_R83'][0]>0 and do>=-.01 and oloss<=.01 and iloss<=.01;confirm=di>=.03 and pci['DELTA_I_R83'][0]>0 and do>=-.02
 gate='FAIL' if di<=0 or do<-.02 else 'STRONG_CONFIRM' if strong else 'CONFIRM' if confirm else 'PARTIAL'
 robustness='FAIL' if dip<=0 else 'ROBUST' if di>0 and pci['DELTA_I_R83'][0]>0 and qci['DELTA_I_R83'][0]>0 else 'WARN'
 mechanism='NOT_SUPPORTED'
 if gate in ('STRONG_CONFIRM','CONFIRM'):mechanism='SUPPORTED' if robustness=='ROBUST' else 'SUPPORTED_WITH_QF_SENSITIVITY_LIMITATION' if robustness=='WARN' else 'PRIMARY_RESULT_CONFIRMED_BUT_QF_ROBUSTNESS_FAILED'
 after,_=frozen_hashes();before=read(OUT/'R83_FROZEN_HASHES_BEFORE.json');assert after==before['hashes']
 freeze=read(OUT/'R83_PREDICTION_FREEZE.json');size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());disk={'utc':now(),'status':'PASS','new_R83_retained_logical_bytes_snapshot':size,'GiB':size/2**30,'disk_free_after_bytes':shutil.disk_usage(OUT).free,'no_R1_R82_deletion_or_cleanup':True,'physical_cumulative_write_counter':'not reconstructed; logical retained storage reported'};dump(OUT/'R83_DISK_IO_AUDIT.json',disk)
 result={'utc':now(),'R83_CONFIRMATION_GATE':gate,'R83_QF_ROBUSTNESS':robustness,'EXTERNAL_MECHANISM_CONFIRMATION':mechanism,'formal_SCI_external_mechanism_claim_allowed':mechanism in ('SUPPORTED','SUPPORTED_WITH_QF_SENSITIVITY_LIMITATION'),'DELTA_I_R83':di,'DELTA_O_R83':do,'CI95':pci,'DELTA_I_R83_QF0':dip,'DELTA_O_R83_QF0':dop,'QF_CI95':qci,'I_PR_AUC_loss':iloss,'O_PR_AUC_loss':oloss,'strong_conditions':strong,'confirm_conditions':confirm,'I_PR_AUC_tolerance_source':'frozen R7 no-material-decline0.01','primary_results':list(primary.values()),'qf_results':list(qf.values()),'prediction_freeze_sha256':sha(OUT/'R83_PREDICTION_FREEZE.json'),'inference_dates':61,'normalization_source':'2024_R6_TRAIN_ONLY','no_threshold_search_or_fit':True,'no_training_or_calibration_fit':True,'R1_R82_hashes_unchanged':True,'mask_intervention':'NOT_RUN_OPTIONAL_RESOURCE_PRIORITY','secondary_baselines':'NOT_RUN_OPTIONAL_PRIMARY_ONLY','inference_runtime_seconds':freeze['inference_elapsed_seconds'],'evaluation_runtime_seconds':time.monotonic()-started,'total_runtime_seconds':(dt.datetime.now(UTC)-dt.datetime.fromisoformat(read(OUT/'R83_RUNTIME_VALIDATION.json')['utc'])).total_seconds(),'peak_GPU_allocated_bytes':freeze['peak_GPU_allocated_bytes'],'peak_RAM_inference_bytes':freeze['peak_RAM_bytes'],'peak_RAM_evaluation_bytes':ENV['peak_rss'](),'disk_bytes_snapshot':size,'stopped_after_confirmation':True}
 statement='External confirmation includes an operational GOES-East platform transition from GOES-16 (2024) to GOES-19 (2025).'
 report='# R83 frozen external confirmatory inference\n\n'+statement+' The differences describe combined platform/year/weather and sample-distribution shift; the spacecraft effect cannot be separated here.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nPrediction freeze preceded all metrics. Complete179861590 samples across61 dates were used without balancing or selection. Both primary rules use exactly the same frozen three-seed ensemble; STATE applies only the frozen R7 Platt coefficients and primary-H thresholds. QF0 uses identical probabilities and unchanged primary-H routing/decisions; only evaluation H/Y labels change. All required source hashes were verified before/after. PR-AUC follows the exact frozen R6 reversed-stable tie order via bounded external sorting, with no grouped-tie approximation. Brier/ECE aggregate float64 sufficient sums across complete dates; no scaler or calibration was fit on2025. The existing R7 I PR-AUC no-material-loss reference0.01 was fixed in runtime protocol before evaluation. Positive DELTA_I with CI crossing0 is PARTIAL under explicit R83 section15; nonpositive DELTA_I or DELTA_O<-0.02 is FAIL. Optional baselines/interventions were not selected. No follow-up tuning is allowed.\n'
 (OUT/'R83_CONFIRMATORY_REPORT.md').write_text(report,encoding='utf-8');dump(OUT/'R83_FROZEN_HASHES_AFTER.json',{'utc':now(),'status':'PASS','hashes':after,'unchanged':True})
 mark='R83_FROZEN_EXTERNAL_CONFIRMATION_20261005';controls=[]
 for name in ('RESEARCH_CONTROL.md','GATE0_DECISION.md'):
  p=ROOT/name;old=p.read_bytes();assert sha(p)==before['controls_before'][name]['sha256'];assert mark.encode() not in old
  text=f'\n\n## {mark}\n\nR83_CONFIRMATION_GATE={gate}; R83_QF_ROBUSTNESS={robustness}; EXTERNAL_MECHANISM_CONFIRMATION={mechanism}.\nDELTA_I={di}; DELTA_O={do}; primary95%CI={pci}.\nEvidence: R8_EXTERNAL_CONFIRMATION/R83_FROZEN_INFERENCE/R83_VALIDATION.json and R83_PREDICTION_FREEZE.json. {statement}\nAll prior R8 BLOCKED/R81 PASS/R82 PASS/R6/R7 records remain preserved. No thresholds, calibration, labels, models or R82 samples changed; no post-result tuning.\n'
  with p.open('ab') as h:h.write(text.encode('utf-8'))
  assert p.read_bytes().startswith(old);controls.append({'path':str(p),'before_hash':hashlib.sha256(old).hexdigest(),'after_hash':sha(p),'prefix_preserved':True})
 dump(OUT/'R83_CONTROL_APPEND_AUDIT.json',{'utc':now(),'status':'PASS','append_only':True,'controls':controls});dump(OUT/'R83_VALIDATION.json',result);dump(OUT/'R83_PROGRESS.json',{'utc':now(),'phase':'R83_COMPLETE','dates_completed':61,'R83_CONFIRMATION_GATE':gate,'R83_QF_ROBUSTNESS':robustness,'EXTERNAL_MECHANISM_CONFIRMATION':mechanism});print('R83_COMPLETE',gate,robustness,mechanism,flush=True)

def main():
 started=time.monotonic();freeze=read(OUT/'R83_PREDICTION_FREEZE.json');assert freeze['status']=='PASS' and freeze['dates']==61 and freeze['samples']==179861590
 regression();prepare(freeze);pr,primary=evaluate(False,freeze);pci=bootstrap(pr,False);qr,qf=evaluate(True,freeze);qci=bootstrap(qr,True);domains=domain(primary);finish(primary,pci,qf,qci,domains,started)
if __name__=='__main__':
 try:main()
 except Exception as e:dump(OUT/('R83_EVALUATION_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'no_method_change':True});raise
