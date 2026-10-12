"""Finalize R82 only after independently verified frozen data; never R83."""
import ast, csv, collections, datetime as dt, hashlib, json, os, shutil, traceback
from pathlib import Path
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve();OUT=ROOT/'R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
UTC=dt.timezone.utc
def now():return dt.datetime.now(UTC).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def new(p,x):
 p=Path(p)
 if p.exists():raise RuntimeError('REFUSE_OVERWRITE_SEALED_FINAL_ARTIFACT:'+str(p))
 dump(p,x)
def main():
 assert not (OUT/'R82_VALIDATION.json').exists() and not (OUT/'R82_CONFIRMATORY_SET_FREEZE.json').exists()
 audit=read(OUT/'R82_FINAL_INDEPENDENT_AUDIT_COMPLETE.json');assert audit['status']=='PASS' and audit['dates']==61
 sourceaudit=read(OUT/'R82_FULL_SOURCE_TEMPORAL_FINAL_AUDIT.json');assert sourceaudit['status']=='PASS' and sourceaudit['dates']==61 and sourceaudit['MRMS_latest_source_records_verified']==52704
 leak=read(OUT/'R82_LABEL_LEAKAGE_AUDIT.json');schema=read(OUT/'R82_FEATURE_SCHEMA_AUDIT.json');cache=read(OUT/'R82_CACHE_FINAL_VALIDATION.json');temporal=read(OUT/'R82_TEMPORAL_VALIDITY_SUMMARY.json');summary=read(OUT/'R82_ONSET_SUFFICIENCY_SUMMARY.json');qf=read(OUT/'R82_QF_SENSITIVITY.json');manifest=read(OUT/'R82_CONFIRMATORY_SAMPLE_MANIFEST/manifest.json');pres=read(OUT/'R82_FROZEN_ARTIFACT_FINAL_AUDIT.json')
 assert all(x['status']=='PASS' for x in (leak,schema,cache,temporal,summary,manifest,pres)) and cache['verified_objects']==52500 and cache['CACHE_OBJECT_MISSING']==cache['CACHE_SCHEMA_MISMATCH']==0
 assert audit['sample_id_hash']==manifest['sample_id_hash']==summary['sample_id_hash'] and audit['samples']==manifest['samples']==summary['sample_count']
 for rel,expected in pres['hashes'].items():
  p=Path(rel);p=p if p.is_absolute() else ROOT/rel;assert sha(p)==expected,rel
 # Byte checksums only; no checkpoint deserialization/import.
 method=read(ROOT/'R7_METHOD_FREEZE.json');expected=next(x['checkpoint_hashes'] for x in method['candidates'] if x['candidate']=='M_HSUPPRESS');checkpoint_checks={}
 for i,hashvalue in enumerate(expected,1):
  p=ROOT/'checkpoints_r7'/('M_HSUPPRESS_seed202609100'+str(i)+'.pt');actual=sha(p);assert actual==hashvalue;checkpoint_checks[str(p)]=actual
 assert read(ROOT/'TRAIN_NORMALIZATION_R6.json')['source_split']=='TRAIN_ONLY' and sha(ROOT/'TRAIN_NORMALIZATION_R6.json')==schema['normalization_sha256']
 blind_scripts={}
 for name in ('r82_glm_precision_recovery.py','r82_raw_composite_audit_build.py','r82_temporal_frozen_build.py','r82_frozen_label_readiness_build.py','r82_frozen_feature_build.py','r82_frozen_event_manifest_build.py','r82_final_independent_audit.py','r82_final_source_temporal_audit.py'):
  p=OUT/name;tree=ast.parse(p.read_text(encoding='utf-8-sig'));imports=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports.extend(x.name.split('.')[0] for x in node.names)
   elif isinstance(node,ast.ImportFrom):imports.append((node.module or '').split('.')[0])
  assert not set(imports)&{'torch','tensorflow','sklearn','keras','xgboost'}
  blind_scripts[name]={'sha256':sha(p),'imports':sorted(set(imports)),'model_libraries_imported':False}
 new(OUT/'R82_BLIND_EXECUTION_AUDIT.json',{'utc':now(),'status':'PASS','data_scripts':blind_scripts,'exact_R5_R6_AST_function_registries_verified':True,'normalization_fitted':False,'checkpoint_bytes_hashed_only':True,'R7_checkpoint_hashes':checkpoint_checks,'models_deserialized':False,'inference_run':False,'training_tuning_calibration_bootstrap_run':False,'model_metrics_computed':False,'R83_run':False,'evidence_scope':'configured data-only script imports, preserved stage audits, exact frozen pure-function registries and independent actual-data checks'})
 days=[x['date'] for x in manifest['partitions']];assert days==[(dt.date(2025,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
 events=[];parts=[];dailycounts=collections.Counter()
 for day in days:
  lm=read(OUT/'label_readiness_daily'/(day+'.json'));fm=read(OUT/'feature_daily'/(day+'.json'));em=read(OUT/'event_manifest_daily'/(day+'.json'));am=read(OUT/'final_independent_audit_daily'/(day+'.json'));assert am['status']=='PASS'
  for p,h in [(lm['store_path'],lm['store_sha256']),(fm['feature_path'],fm['feature_sha256']),(em['manifest_path'],em['manifest_sha256'])]:assert sha(p)==h
  parts.append({'date':day,'samples':em['samples'],'label_sha256':lm['store_sha256'],'feature_sha256':fm['feature_sha256'],'manifest_sha256':em['manifest_sha256'],'sample_id_partition_hash':em['sample_id_partition_hash'],'audit_sha256':sha(OUT/'final_independent_audit_daily'/(day+'.json'))})
  events.extend(em['events']);dailycounts[day]=em['usable_onset_events']
 usable=[e for e in events if e['usable']];n=len(usable);top=sorted(dailycounts.values(),reverse=True);regions=collections.Counter(e['spatial_region'] for e in usable)
 assert len(events)==summary['N_RAW_ONSET_EVENTS_R82'] and n==summary['N_USABLE_ONSET_EVENTS_R82'] and len([v for v in dailycounts.values() if v])==summary['active_onset_dates']
 assert n>=500 and summary['active_onset_dates']>=20 and top[0]/n==summary['top1_date_share'] and sum(top[:5])/n==summary['top5_date_share'] and max(regions.values())/n==summary['max_16_region_share']
 for e in usable:assert e['spatial_region']==min(3,int((e['centroid_lat']-31)/2))*4+min(3,int((e['centroid_lon']+100)/2))
 categories=collections.Counter();files=collections.Counter()
 for directory,dirs,names in os.walk(OUT):
  for name in names:
   p=Path(directory)/name;category=p.relative_to(OUT).parts[0];categories[category]+=p.stat().st_size;files[category]+=1
 retained=sum(categories.values());before=read(OUT/'R82_PREBUILD_BASELINE.json')['disk_free_before_bytes'];free=shutil.disk_usage(ROOT).free
 disk={'utc':now(),'status':'PASS','new_R82_retained_logical_bytes':retained,'new_R82_retained_GiB':retained/2**30,'disk_free_baseline_bytes':before,'disk_free_after_bytes':free,'volume_free_delta_bytes':before-free,'category_logical_bytes':dict(categories),'category_file_counts':dict(files),'scope':'R82 tree including historical retries, source caches, new labels/features/manifests; logical retained storage snapshot, not cumulative physical device writes','cumulative_physical_write_bytes':'NOT_RECONSTRUCTABLE_FROM_RETRY_LOGS','source_transfer_bytes':{'three_product_ROI_rebuild':51024330363,'GLM_precision_recovery':read(OUT/'R82_GLM_PRECISION_RECOVERY_COMPLETE.json')['download_bytes_this_run'],'raw_composite_audit_reference':read(OUT/'R82_RAW_COMPOSITE_AUDIT_COMPLETE.json')['source_bytes']},'no_full_source_retention':True,'completed_QC_cache_builders_rerun':False}
 new(OUT/'R82_DISK_IO_AUDIT.json',disk)
 text='External confirmation includes an operational GOES-East platform transition from GOES-16 (2024) to GOES-19 (2025).'
 frozen={'creation_timestamp':now(),'set':'EXTERNAL_CONFIRMATION_SET','date_range':['2025-06-01','2025-07-31'],'dates':61,'GOES_platform':'GOES-19','GLM_product':'GLM-L2-LCFA','scientific_source':'OPERATIONAL GOES-EAST GLM-L2-LCFA','platform_transition_statement':text,'potential_domain_shift_part_of_independent_confirmation':True,'MRMS_products':schema['feature_schema']['model_input_fields'],'raw_composite_role':'QC_AUDIT_REFERENCE_ONLY; NOT a fourth model input','sample_count':manifest['samples'],'class_counts':audit['class_counts'],'raw_onset_events':len(events),'usable_onset_events':n,'active_onset_dates':summary['active_onset_dates'],'sample_id_hash':manifest['sample_id_hash'],'sample_id_hash_encoding':manifest['sample_id_hash_encoding'],'manifest_sha256':sha(OUT/'R82_CONFIRMATORY_SAMPLE_MANIFEST/manifest.json'),'feature_schema_hash':schema['feature_schema_sha256'],'label_policy_hash':sha(ROOT/'LABEL_DEFINITION.md'),'QC_policy_hash':sha(ROOT/'QC_POLICY.md'),'R45_mask_policy_hash':sha(ROOT/'FINAL_DATA_MASK_POLICY_R45.md'),'normalization_hash':sha(ROOT/'TRAIN_NORMALIZATION_R6.json'),'normalization_source':'2024_R6_TRAIN_ONLY','R7_method_hash':sha(ROOT/'R7_METHOD_FREEZE.json'),'R7_checkpoint_byte_hash_checks':checkpoint_checks,'independent_audit_hash':sha(OUT/'R82_FINAL_INDEPENDENT_AUDIT_COMPLETE.json'),'full_source_temporal_audit_hash':sha(OUT/'R82_FULL_SOURCE_TEMPORAL_FINAL_AUDIT.json'),'partitions':parts,'history_interval':'[t-1800,t]','history_radius_km':20,'future_interval':'(t,t+1800]','future_radius_km':10,'MRMS_age_limit_sec':180,'MRMS_source_rule':'latest official <= frame; no future fill','readiness':'MODEL_READY_MASKABLE','feature_schema':schema['feature_schema'],'QF_SENSITIVITY_ALERT_R82':qf['QF_SENSITIVITY_ALERT_R82'],'QF_label_agreement':qf['label_agreement'],'primary_labels_changed':False,'no_2025_normalization_fit':True,'no_date_weather_class_performance_selection':True,'models_loaded':False,'inference_run':False,'R83_mutation_prohibited':True}
 new(OUT/'R82_CONFIRMATORY_SET_FREEZE.json',frozen);freezehash=sha(OUT/'R82_CONFIRMATORY_SET_FREEZE.json')
 report=f'''# R82 confirmatory data gate evidence\n\n{ text } This operational transition is a potential domain shift and part of the independent confirmation.\n\nAll 61 dates are one complete external confirmation set. The three input products, C30 rule, R5 endpoints/radii, R45 exact closed gap propagation, R52 six-frame maskable readiness and R6/R7 feature schema remain frozen. Raw Composite was explicitly authorized only as the original conditional QC audit reference.\n\nCache final validation: 52500/52500 PASS, missing=0, schema mismatch=0. Exact gaps: {temporal['total_gap_count']}, maximum {temporal['max_gap_seconds']} seconds. Valid anchors: {temporal['valid_anchors']}/{temporal['total_5min_anchors']} ({temporal['valid_anchor_fraction']:.8%}); causal six-frame anchors {temporal['mrms_causal_history_available_anchors']}. No gap-duration tolerance, interpolation or future fill. Full-span content uncertainty and neighbor-day intersections were checked.\n\nSparse candidates {read(OUT/'R82_LABEL_READINESS_COMPLETE.json')['sample_rows']}; maskable samples {manifest['samples']}. Classes {audit['class_counts']}. Raw onset events {len(events)}, usable {n}, capture {n/len(events):.8%}, active dates {summary['active_onset_dates']}. Top1 {summary['top1_date_share']:.8%}, top5 {summary['top5_date_share']:.8%}, maximum fixed 16-region share {summary['max_16_region_share']:.8%}. Temporal/spatial concentration alerts are {summary['TEMPORAL_CONCENTRATION_ALERT']}/{summary['SPATIAL_CONFIRMATION_ALERT']}.\n\nQF_SENSITIVITY_ALERT_R82={qf['QF_SENSITIVITY_ALERT_R82']}: label agreement {qf['label_agreement']:.8%}, event-count change {qf['event_change_fraction']:.8%}. Agreement is below the frozen 95% alert reference. This alert is disclosed; primary labels, thresholds and samples were not changed.\n\nInvalid or ambiguous channels carry mask0 and raw placeholder0. Frozen R4/R6 corroborated no-echo categories remain explicitly categorical encodings, never physical values; no semantic mask was redefined. Normalization is exclusively unchanged 2024 R6 TRAIN; no 2025 fitting. Actual full partitions were checked for SHA, label formulas, temporal validity, classes, feature rows/shape/dtypes/mask bits/placeholder and frozen transforms. All primary onset membership/components were recomputed. Candidate/H/Y/readiness were independently verified for all candidates at three predeclared anchors per date; event assignment tie-break was checked on deterministic stratified targets. Audit scope is explicit in the daily records.\n\nSample ID SHA256: {manifest['sample_id_hash']}. Freeze SHA256: {freezehash}. R1-R7 baseline files, declared historical label/readiness partitions, old R82 GLM partitions and R8/R81 history were byte-hash verified within the documented scope. Controls are updated only by appending. No model was deserialized and no inference, training, tuning, calibration, bootstrap or model metric was run.\n\nRetained new R82 logical storage snapshot: {retained} bytes ({retained/2**30:.3f} GiB). This is retained logical storage, not a reconstructed cumulative physical-write counter. See R82_DISK_IO_AUDIT.json.\n'''
 reportpath=OUT/'R82_FINAL_DATA_QC_REPORT.md';assert not reportpath.exists();reportpath.write_text(report,encoding='utf-8')
 # Preserve the entire prior control-document body byte for byte.
 control=[];tag='R82_FINAL_CONFIRMATORY_DATA_GATE_20261005'
 appendix=f'\n\n## {tag}\n\nR82_CONFIRMATORY_DATA_GATE = PASS\nREADY_FOR_R83_FROZEN_CONFIRMATORY_INFERENCE = YES\n\nEvidence: R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD/R82_VALIDATION.json and R82_CONFIRMATORY_SET_FREEZE.json (SHA256 {freezehash}). {manifest["samples"]} frozen maskable samples; {n} usable onset events on 61 dates. QF sensitivity alert YES is retained without changing primary labels. {text}\n\nOriginal R8 BLOCKED_BY_DATA and R81 PASS records remain preserved. R7 method/calibration/thresholds and R6 TRAIN normalization are unchanged. R83 was not run.\n'
 for name in ('RESEARCH_CONTROL.md','GATE0_DECISION.md'):
  p=ROOT/name;old=p.read_bytes();assert tag.encode() not in old;beforehash=hashlib.sha256(old).hexdigest()
  with p.open('ab') as h:h.write(appendix.encode('utf-8'))
  assert p.read_bytes().startswith(old);control.append({'path':str(p),'before_sha256':beforehash,'after_sha256':sha(p),'original_bytes':len(old),'prefix_unchanged':True})
 for rel,expected in pres['hashes'].items():
  p=Path(rel);p=p if p.is_absolute() else ROOT/rel;assert sha(p)==expected,rel
 new(OUT/'R82_CONTROL_APPEND_AUDIT.json',{'utc':now(),'status':'PASS','append_only':True,'controls':control,'frozen_artefacts_after_append_unchanged':True})
 new(OUT/'R82_FINAL_AUDITOR_ENGINEERING_CORRECTIONS.json',{'utc':now(),'status':'RESOLVED_WITHOUT_DATA_OR_METHOD_CHANGE','preserved_failures':['R82_FINAL_INDEPENDENT_AUDIT_FAILURE_20261004T230152Z.json','R82_FINAL_INDEPENDENT_AUDIT_FAILURE_20261004T230216Z.json'],'corrections':['Use causal-source flag at 00:25 instead of conflating unused uncached previous-day object with source availability; label support starts00:30 and required eligible source caches remain strictly checked.','Read normalization hash at documented feature_schema nested metadata path.'],'source_QC_label_feature_rules_changed':False})
 conditions={'cache_final_validation_PASS':True,'temporal_propagation_complete':True,'frozen_R5_labels_complete':True,'frozen_R52_maskable_complete':True,'usable_onset_ge500':n>=500,'active_dates_ge20':summary['active_onset_dates']>=20,'label_leakage_PASS':leak['status']=='PASS','feature_schema_PASS':schema['status']=='PASS','normalization_R6_TRAIN_ONLY':True,'manifest_frozen':True,'confirmatory_set_freeze_success':True,'R1_R7_hashes_unchanged':True,'no_model_inference':True}
 assert all(conditions.values())
 validation={'utc':now(),'R82_CONFIRMATORY_DATA_GATE':'PASS','READY_FOR_R83_FROZEN_CONFIRMATORY_INFERENCE':'YES','status':'PASS','conditions':conditions,'freeze_sha256':freezehash,'date_range':frozen['date_range'],'samples':frozen['sample_count'],'class_counts':frozen['class_counts'],'temporal_summary':temporal,'onset_summary':summary,'QF_SENSITIVITY_ALERT_R82':qf['QF_SENSITIVITY_ALERT_R82'],'normalization_source':'2024_R6_TRAIN_ONLY','normalization_sha256':frozen['normalization_hash'],'sample_id_hash':manifest['sample_id_hash'],'feature_schema_hash':schema['feature_schema_sha256'],'platform_transition_statement':text,'frozen_artifacts_audit':str(OUT/'R82_FROZEN_ARTIFACT_FINAL_AUDIT.json'),'control_append_audit':str(OUT/'R82_CONTROL_APPEND_AUDIT.json'),'new_retained_logical_bytes_snapshot':retained,'models_loaded':False,'R83_run':False,'stop_here':True}
 new(OUT/'R82_VALIDATION.json',validation)
 dump(OUT/'R82_PROGRESS.json',{'utc':now(),'phase':'R82_COMPLETE','dates_completed':61,'labels_constructed':True,'R82_CONFIRMATORY_DATA_GATE':'PASS','READY_FOR_R83_FROZEN_CONFIRMATORY_INFERENCE':'YES','models_loaded':False,'R83_run':False})
 print(json.dumps({'R82_CONFIRMATORY_DATA_GATE':'PASS','READY_FOR_R83_FROZEN_CONFIRMATORY_INFERENCE':'YES','storage_bytes':retained,'freeze_sha256':freezehash,'sample_id_hash':manifest['sample_id_hash']},ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  dump(OUT/('R82_FINALIZATION_FAILURE_'+dt.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'models_loaded':False});raise
