"""R83 complete frozen ensemble inference before any confirmatory metrics."""
import gc,hashlib,os,sys,time,traceback,datetime as dt
from pathlib import Path
from r83_common import *
import torch
import pyarrow as pa
import pyarrow.parquet as pq

def load_models(reg):
 env={'torch':torch,'np':np,'ctypes':ctypes};reg.update(pure('scripts/r6_baseline.py',['MLP','nrm','peak_rss'],env));norm,state,gt=config();models=[];cp_hashes={}
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 for seed in SEEDS:
  cp=ROOT/'checkpoints_r6'/f'M_FUSION_seed{seed}.pt';cp_hashes[str(cp)]=sha(cp);a=torch.load(cp,map_location='cpu',weights_only=False)
  assert a['kind']=='M_FUSION' and a['seed']==seed and a['complete'] and a['normalization_sha256']==sha(ROOT/'TRAIN_NORMALIZATION_R6.json')
  m=env['MLP'](48);m.load_state_dict(a['model'],strict=True);m.to(device).eval();models.append(m);del a
 return env,norm,models,device,cp_hashes

def dayrun(part,env,norm,models,device,cp_hashes):
 day=part['date'];path=OUT/'predictions'/(day+'.parquet');meta=OUT/'prediction_daily'/(day+'.json')
 if meta.exists() and path.exists():
  m=read(meta);assert m['prediction_sha256']==sha(path) and m['checkpoint_hashes']==cp_hashes and m['source_feature_sha256']==part['feature_sha256'] and m['sample_id_partition_hash']==part['sample_id_partition_hash'];return m
 diskcheck(part['samples']*40);start=time.monotonic();feature=DATA/'cache_r82/features'/(day+'.npz');assert sha(feature)==part['feature_sha256'];z=mmap_npz(feature);n=part['samples']
 assert set(z)=={'met','ltg','y','h','clazz','row','time_index','pixel_index'} and z['met'].shape==(n,36) and z['ltg'].shape==(n,12) and z['met'].dtype==z['ltg'].dtype==np.dtype('f4')
 with nc.Dataset(DATA/'R82_CONFIRMATORY_SAMPLE_MANIFEST'/(day+'.nc')) as manifest,nc.Dataset(DATA/'labels_r82'/(day+'.nc')) as label:
  manifest.set_auto_mask(False);label.set_auto_mask(False);assert len(manifest['sample_id'])==n
  label_arrays={k:label[k][:] for k in ('H_primary','Y_primary','H_qf0','Y_qf0','class_qf0')}
  path.parent.mkdir(exist_ok=True);temp=path.with_name(path.name+'.tmp');schema=pa.schema([('sample_id',pa.uint64()),('date',pa.string()),('source_label_row',pa.uint32()),('H',pa.uint8()),('Y',pa.uint8()),('primary_label',pa.int8()),('QF0_H',pa.uint8()),('QF0_Y',pa.uint8()),('QF0_sensitivity_label',pa.int8()),('ensemble_probability',pa.float32())]);schema=schema.with_metadata({b'set':b'EXTERNAL_CONFIRMATION_SET',b'ensemble':b'float32 mean of frozen R6 FUSION seeds2026091001,2026091002,2026091003',b'no_partial_metrics':b'true',b'state_routing':b'frozen PRIMARY H for both label evaluations'})
  hashes=[hashlib.sha256() for _ in models];ensemblehash=hashlib.sha256();sidhash=hashlib.sha256();batch=65536 if device.type=='cuda' else 8192
  with pq.ParquetWriter(temp,schema,compression='zstd',use_dictionary=['date'],write_statistics=True) as writer,torch.inference_mode():
   for lo in range(0,n,batch):
    hi=min(n,lo+batch);r=np.asarray(manifest['source_label_row'][lo:hi],dtype='u4');assert np.array_equal(r,z['row'][lo:hi]);sid=np.asarray(manifest['sample_id'][lo:hi],dtype='<u8')
    assert np.array_equal(z['h'][lo:hi],label_arrays['H_primary'][r.astype('i8')].astype('u1')) and np.array_equal(z['y'][lo:hi],label_arrays['Y_primary'][r.astype('i8')].astype('u1'))
    raw={'met':z['met'][lo:hi],'ltg':z['ltg'][lo:hi]};x=env['nrm'](raw,norm,'M_FUSION');assert x.shape==(hi-lo,48) and x.dtype==np.dtype('f4') and np.all(np.isfinite(x));xx=torch.from_numpy(x).to(device);ps=[]
    for k,m in enumerate(models):
     p=torch.sigmoid(m(xx)).cpu().numpy().astype('f4',copy=False);assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1));hashes[k].update(p.astype('<f4',copy=False).tobytes());ps.append(p)
    ensemble=np.mean(np.stack(ps),axis=0,dtype='f4');assert np.array_equal(ensemble,((ps[0]+ps[1])+ps[2])/np.float32(3));ensemblehash.update(ensemble.astype('<f4',copy=False).tobytes());sidhash.update(sid.tobytes())
    a={'sample_id':sid,'date':pa.array([day]*(hi-lo)),'source_label_row':r,'H':z['h'][lo:hi],'Y':z['y'][lo:hi],'primary_label':z['clazz'][lo:hi],'QF0_H':label_arrays['H_qf0'][r.astype('i8')].astype('u1'),'QF0_Y':label_arrays['Y_qf0'][r.astype('i8')].astype('u1'),'QF0_sensitivity_label':label_arrays['class_qf0'][r.astype('i8')],'ensemble_probability':ensemble};writer.write_table(pa.Table.from_pydict(a,schema=schema),row_group_size=batch)
    if lo%(batch*32)==0:dump(OUT/'R83_PROGRESS.json',{'utc':now(),'phase':'BLIND_INFERENCE','pid':os.getpid(),'date':day,'dates_completed':DAYS.index(day),'samples_this_date':hi,'samples_required_date':n,'partial_metrics_generated':False,'elapsed_seconds':time.monotonic()-RUN_START})
    del x,xx,ps
  assert sidhash.hexdigest()==part['sample_id_partition_hash'];assert pq.ParquetFile(temp).metadata.num_rows==n;os.replace(temp,path)
 m={'utc':now(),'date':day,'status':'PASS','samples':n,'prediction_path':str(path),'prediction_sha256':sha(path),'prediction_bytes':path.stat().st_size,'sample_id_partition_hash':sidhash.hexdigest(),'seed_probability_hashes':dict(zip(map(str,SEEDS),(h.hexdigest() for h in hashes))),'ensemble_probability_hash':ensemblehash.hexdigest(),'ensemble_verified':'exact float32 arithmetic mean of all three seed outputs batchwise','source_feature_sha256':part['feature_sha256'],'checkpoint_hashes':cp_hashes,'elapsed_seconds':time.monotonic()-start,'device':str(device),'peak_ram_bytes':env['peak_rss'](),'peak_GPU_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else 0,'partial_metrics_generated':False}
 dump(meta,m);del z;gc.collect();print('R83_DATE_COMPLETE',day,n,flush=True);return m

RUN_START=time.monotonic()
def main():
 OUT.mkdir(exist_ok=True);(OUT/'prediction_daily').mkdir(exist_ok=True);start=now();hashes,freeze=frozen_hashes();schemahash,schema=runtime_schema();diskcheck();reg={};env,norm,models,device,cph=load_models(reg)
 beforefile=OUT/'R83_FROZEN_HASHES_BEFORE.json'
 if beforefile.exists():assert read(beforefile)['hashes']==hashes
 else:dump(beforefile,{'utc':start,'hashes':hashes,'controls_before':{name:{'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in ['RESEARCH_CONTROL.md','GATE0_DECISION.md']}})
 runtime={'utc':start,'status':'PASS','frozen_hashes_PASS':True,'runtime_feature_schema_hash':schemahash,'R82_feature_schema_hash':freeze['feature_schema_hash'],'normalization_source':'2024_R6_TRAIN_ONLY','normalization_hash':freeze['normalization_hash'],'R7_method_hash':freeze['R7_method_hash'],'global_threshold_rule_hash':sha(ROOT/'VALIDATION_THRESHOLDS_R6.csv'),'checkpoint_hashes':cph,'sample_id_hash':freeze['sample_id_hash'],'samples':freeze['sample_count'],'seeds':SEEDS,'exact_frozen_functions':reg,'device':str(device),'GPU':torch.cuda.get_device_name() if device.type=='cuda' else None,'torch_version':torch.__version__,'float32_no_autocast':True,'TF32':False,'batch_size':65536 if device.type=='cuda' else 8192,'evaluation_protocol':{'bootstrap_unit':'61 complete UTC dates','bootstrap_replicates':1000,'bootstrap_rng_seed':20260910,'PR_AUC':'verbatim R6 stable ascending mergesort reversed: score DESC, global sample_id DESC ties; external sort streams identical ranks','ECE_bins':15,'QF0_evaluation_subgroups':'QF0 H/Y; calibrated decision routing remains PRIMARY H; same predictions, no reroute/refit','I_PR_AUC_no_material_loss_tolerance':0.01,'I_PR_AUC_tolerance_source':'existing frozen R7 code preregistered0.01, not estimated from2025','gate_CI_cross_zero_positive_delta':'PARTIAL per explicit R83 section15; nonpositive delta or O<-0.02 FAIL','secondary_baselines':'not required; omitted to prioritize complete primary test','optional_intervention':'not selected; resource priority is complete primary and mandatory sensitivity'},'no_training':True,'no_calibration_fit':True,'no_partial_metrics':True,'all_new_outputs_under':str(OUT)}
 if (OUT/'R83_RUNTIME_VALIDATION.json').exists():assert read(OUT/'R83_RUNTIME_VALIDATION.json')['checkpoint_hashes']==cph
 else:dump(OUT/'R83_RUNTIME_VALIDATION.json',runtime)
 daily=[]
 for part in freeze['partitions']:daily.append(dayrun(part,env,norm,models,device,cph))
 assert len(daily)==61 and sum(x['samples'] for x in daily)==freeze['sample_count'];digest=hashlib.sha256()
 for m in daily:
  assert sha(m['prediction_path'])==m['prediction_sha256']
  for b in pq.ParquetFile(m['prediction_path']).iter_batches(batch_size=500000,columns=['sample_id']):digest.update(b.column(0).to_numpy().astype('<u8',copy=False).tobytes())
 assert digest.hexdigest()==freeze['sample_id_hash'];after,_=frozen_hashes();assert after==hashes
 writecsv(OUT/'R83_PREDICTION_MANIFEST.csv',[{k:m[k] for k in ('date','samples','prediction_path','prediction_sha256','prediction_bytes','sample_id_partition_hash','ensemble_probability_hash','elapsed_seconds')} for m in daily])
 sealed={'utc':now(),'status':'PASS','dates':61,'samples':freeze['sample_count'],'checkpoint_hashes':cph,'sample_id_hash':digest.hexdigest(),'R82_freeze_sha256':sha(DATA/'R82_CONFIRMATORY_SET_FREEZE.json'),'prediction_partitions':daily,'prediction_manifest_hash':sha(OUT/'R83_PREDICTION_MANIFEST.csv'),'ensemble_definition':'float32 mean(p_seed2026091001,p_seed2026091002,p_seed2026091003); seeds unchanged; same ensemble for primary models','R6_global_rule_hash':runtime['global_threshold_rule_hash'],'R7_method_hash':freeze['R7_method_hash'],'normalization_hash':freeze['normalization_hash'],'schema_hash':schemahash,'partial_metrics_generated':False,'inference_elapsed_seconds':time.monotonic()-RUN_START,'peak_RAM_bytes':env['peak_rss'](),'peak_GPU_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,'R1_R82_hashes_after':after}
 if (OUT/'R83_PREDICTION_FREEZE.json').exists():raise RuntimeError('EXISTING_PREDICTION_FREEZE_DO_NOT_OVERWRITE')
 dump(OUT/'R83_PREDICTION_FREEZE.json',sealed);dump(OUT/'R83_PROGRESS.json',{'utc':now(),'phase':'PREDICTIONS_FROZEN_EVALUATION_PENDING','dates_completed':61,'prediction_freeze':'PASS','pid':os.getpid(),'partial_metrics_generated':False});print('R83_PREDICTION_FREEZE_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:dump(OUT/('R83_INFERENCE_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'partial_metrics_generated':False});raise
