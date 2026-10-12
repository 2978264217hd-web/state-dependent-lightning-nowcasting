"""Release I/O commands. Scientific functions are imported unchanged from frozen sources.

None of these preprocessing/inference/evaluation/figure commands was executed
during local submission packaging. Choose a new reproduction workspace.
"""
import argparse, hashlib, importlib, json, os, shutil, subprocess, sys
from pathlib import Path

RELEASE=Path(__file__).resolve().parent
def data_root():
    value=os.environ.get('LIGHTNING_DATA_ROOT')
    if not value:raise RuntimeError('Supply --data-root or LIGHTNING_DATA_ROOT for the separately obtained processed DATA package')
    p=Path(value).resolve()
    if not p.is_dir():raise FileNotFoundError('Processed DATA directory is unavailable')
    return p
R82='R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD'
R83='R8_EXTERNAL_CONFIRMATION/R83_FROZEN_INFERENCE'
STAGES={
    'mrms-inventory':R82+'/r82_mrms_inventory.py',
    'mrms-content-qc':R82+'/r82_mrms_content_qc_retry14.py',
    'mrms-roi-cache':R82+'/r82_mrms_roi_cache_build.py',
    'glm-precision':R82+'/r82_glm_precision_recovery.py',
    'raw-composite-audit':R82+'/r82_raw_composite_audit_build.py',
    'temporal-support':R82+'/r82_temporal_frozen_build.py',
    'label-readiness':R82+'/r82_frozen_label_readiness_build.py',
    'events-and-sample-manifest':R82+'/r82_frozen_event_manifest_build.py',
    'features':R82+'/r82_frozen_feature_build.py',
}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def safe_child(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes intended directory')
    return p
def verify_release():
    hashes=read(RELEASE/'SHA256SUMS.json')
    for rel,digest in hashes.items():
        p=safe_child(RELEASE,rel)
        if not p.is_file() or sha(p)!=digest:raise RuntimeError('Release hash mismatch: '+rel)
def prepare(workspace):
    verify_release()
    root=Path(workspace).resolve()
    if root.is_relative_to(RELEASE) or RELEASE.is_relative_to(root):
        raise ValueError('Choose a workspace outside the release package tree')
    root.mkdir(parents=True,exist_ok=True)
    for row in read(RELEASE/'workspace_map.json'):
        src=safe_child(RELEASE,row['release']);dest=safe_child(root,row['workspace'])
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest)!=sha(src):raise RuntimeError('Refusing to replace changed workspace input: '+row['workspace'])
        else:shutil.copy2(src,dest)
    # The companion processed DATA package is not published in this repository.
    # Stage source-relative metadata only when explicitly supplied by the user.
    if os.environ.get('LIGHTNING_DATA_ROOT'):
        external=data_root()
        for category in external.iterdir():
            if not category.is_dir():continue
            for source in category.rglob('*'):
                if not source.is_file() or source.suffix not in ['.json','.csv','.md']:continue
                relative=source.relative_to(category)
                # Preserve future inference outputs; archived prediction seals
                # require an explicit, separately audited restoration workflow.
                if relative.as_posix().startswith(R83+'/'):continue
                dest=safe_child(root,relative)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    if sha(dest)!=sha(source):raise RuntimeError('Different staged metadata input: '+relative.as_posix())
                else:shutil.copy2(source,dest)
    os.environ['LIGHTNING_WORKSPACE']=str(root)
    for rel in ['scripts',R82,R83]:
        (root/rel).mkdir(parents=True,exist_ok=True)
        sys.path.insert(0,str(root/rel))
    os.chdir(root)
    return root
def frozen_inference(root):
    """Apply existing load_models/dayrun; substitute only release-level input audit I/O."""
    c=importlib.import_module('r83_common');m=importlib.import_module('r83_inference')
    freeze=read(root/R82/'R82_CONFIRMATORY_SET_FREEZE.json')
    assert freeze['sample_count']==179861590 and len(freeze['partitions'])==61
    for part in freeze['partitions']:
        for rel,key in [('labels_r82','label_sha256'),('cache_r82/features','feature_sha256'),('R82_CONFIRMATORY_SAMPLE_MANIFEST','manifest_sha256')]:
            ext='.npz' if rel.endswith('features') else '.nc'
            p=root/R82/rel/(part['date']+ext)
            if not p.is_file():raise FileNotFoundError('Required reconstructed/archived processed input: '+str(p))
            assert sha(p)==part[key], 'Frozen partition bytes differ: '+str(p)
    # No hash claim is made for sanitized container metadata versus original bytes.
    # Scientific normalization/method/checkpoint files keep their original byte hashes.
    assert sha(root/'TRAIN_NORMALIZATION_R6.json')==freeze['normalization_hash']
    assert sha(root/'R7_METHOD_FREEZE.json')==freeze['R7_method_hash']
    c.runtime_schema()
    registry={};env,norm,models,device,cp_hashes=m.load_models(registry)
    (m.OUT/'predictions').mkdir(parents=True,exist_ok=True)
    (m.OUT/'prediction_daily').mkdir(exist_ok=True)
    seal=m.OUT/'R83_PREDICTION_FREEZE.json'
    if seal.exists():raise RuntimeError('Existing prediction seal is preserved; choose a new workspace')
    daily=[m.dayrun(part,env,norm,models,device,cp_hashes) for part in freeze['partitions']]
    assert sum(x['samples'] for x in daily)==freeze['sample_count']
    sid=hashlib.sha256()
    for row in daily:
        for batch in m.pq.ParquetFile(row['prediction_path']).iter_batches(batch_size=500000,columns=['sample_id']):
            sid.update(batch.column(0).to_numpy().astype('<u8',copy=False).tobytes())
    assert sid.hexdigest()==freeze['sample_id_hash']
    c.dump(seal,{'status':'PASS','dates':61,'samples':freeze['sample_count'],'prediction_partitions':daily,'sample_id_hash':sid.hexdigest(),'checkpoint_hashes':cp_hashes,'exact_frozen_functions':registry,'release_input_audit':'Release checksums plus original frozen scientific partition/parameter/checkpoint hashes; metadata paths relocated only.','no_training':True,'no_calibration_fit':True})
def frozen_verification(root):
    m=importlib.import_module('r83_evaluate');freeze=read(root/R83/'R83_PREDICTION_FREEZE.json')
    assert freeze['status']=='PASS' and freeze['dates']==61 and freeze['samples']==179861590
    # The exact original numeric verification functions and ordering are used.
    # Domain-shift/control-log historical bookkeeping is supplied as frozen CSVs.
    m.regression();m.prepare(freeze)
    primary_rows,primary=m.evaluate(False,freeze);primary_ci=m.bootstrap(primary_rows,False)
    qf_rows,qf=m.evaluate(True,freeze);qf_ci=m.bootstrap(qf_rows,True)
    m.dump(m.OUT/'RELEASE_REPRODUCED_VERIFICATION.json',{'primary':list(primary.values()),'primary_intervals':primary_ci,'QF0':list(qf.values()),'QF0_intervals':qf_ci,'no_refitting':True,'exact_frozen_functions':m.REG})
def tables(destination):
    verify_release();dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    if dest.is_relative_to(RELEASE) or dest.is_relative_to(data_root()):raise ValueError('Choose an output folder outside repository and DATA package')
    for p in (data_root()/'verification_tables').glob('manuscript_*_table_*.csv'):
        shutil.copy2(p,dest/p.name)
def figures(destination):
    verify_release()
    data_root()
    os.environ.setdefault('LIGHTNING_WORKSPACE',str(Path.cwd().resolve()))
    os.environ['LIGHTNING_FIGURE_OUTPUT']=str(Path(destination).resolve())
    if Path(destination).resolve().is_relative_to(RELEASE):raise ValueError('Choose output outside release package')
    subprocess.run([sys.executable,str(RELEASE/'figures/r10v2_figures.py')],check=True,env=os.environ.copy())
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','stage','inference','verification','tables','figures'])
    ap.add_argument('--workspace',default='reproduction_workspace')
    ap.add_argument('--data-root',help='Separately obtained processed DATA directory')
    ap.add_argument('--output',default='reproduced_outputs')
    ap.add_argument('--stage',choices=list(STAGES))
    a,stage_args=ap.parse_known_args()
    if stage_args and a.command!='stage':ap.error('Unexpected extra arguments')
    if a.data_root:os.environ['LIGHTNING_DATA_ROOT']=str(Path(a.data_root).resolve())
    if a.command!='prepare':data_root()
    if a.command=='tables':tables(a.output);return
    if a.command=='figures':figures(a.output);return
    root=prepare(a.workspace)
    if a.command=='prepare':return
    if a.command=='inference':frozen_inference(root);return
    if a.command=='verification':frozen_verification(root);return
    if not a.stage:ap.error('stage command requires --stage')
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(str(root/x) for x in ['scripts',R82,R83])
    args=stage_args[1:] if stage_args[:1]==['--'] else stage_args
    subprocess.run([sys.executable,str(root/STAGES[a.stage]),*args],check=True,cwd=root,env=env)
if __name__=='__main__':main()
