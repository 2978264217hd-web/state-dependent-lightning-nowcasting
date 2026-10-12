"""Freeze R4 scope; inventory and reuse existing evidence, never overwrite R1-R3."""
import collections,csv,datetime as dt,hashlib,http.client,json,shutil,sqlite3,sys,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve();OUT=ROOT/'recovery_r4';UTC=dt.timezone.utc
DAYS=[(dt.date(2024,6,1)+dt.timedelta(days=i)).isoformat() for i in range(61)]
FIXED={'2024-07-08','2024-07-10'};QC='MergedReflectivityQCComposite'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return dt.datetime.now(UTC).isoformat()
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def connect():
 c=sqlite3.connect(OUT/'FILE_QC_R4.sqlite',timeout=120);c.execute('PRAGMA journal_mode=WAL');return c
def freeze():
 OUT.mkdir(exist_ok=True)
 if (OUT/'PREREGISTRATION_R4.json').exists():return
 frozen=OUT/'r3_frozen';frozen.mkdir(exist_ok=True);manifest={}
 paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.md','.csv','.json'}]+list((ROOT/'config').glob('*.json'))
 paths += [ROOT/'recovery_r1/full_content_qc/FILE_QC.sqlite',ROOT/'recovery_r2/SEMANTIC_QC_R2.sqlite',ROOT/'recovery_r3/RADAR_INPUT_R3.sqlite']
 for p in paths:
  rel=p.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size}
 dump(frozen/'MANIFEST.json',manifest)
 policy={'phase':'GATE0_FULL_CONTENT_AUDIT_R4','utc':now(),'candidate_dates':[d for d in DAYS if d not in FIXED],
  'fixed_excluded_dates':sorted(FIXED),'fixed_reason':'SENSOR_DATA_INCOMPLETE','mrms_bias_dates':DAYS,
  'required_fields':[QC,'Reflectivity_-10C','EchoTop_30'],'audit_reference':'MergedReflectivityComposite',
  'MAX_MRMS_AGE_SECONDS':180,'alignment':'latest source timestamp<=t, age<=180; no future source, no interpolation',
  'candidate_threshold_dbz':30,'ambiguity_threshold':.05,'Z10_STATUS':'RETAIN','ECHO_TOP_STATUS':'RETAIN',
  'semantic_function':'scripts/analyze_semantic_pilot_r2.py:classify (unchanged as frozen by R3)',
  'semantic_function_sha256':sha(ROOT/'scripts/analyze_semantic_pilot_r2.py'),
  'qc_missing':'-99=NO_TRUSTWORTHY_QC_RETURN_WITH_COVERAGE; candidate FALSE only when source/time/space valid',
  'qc_no_coverage':'-999 unavailable; never zero','glm_feature_qf':'separate from observation completeness; preserve 0/1/3/5 counts',
  'glm_bin':'[t,t+300s); union of readable, trustworthy metadata coverage, no future fill',
  'glm_gap_mask':'history [t-1800,t) or future [t,t+1800) intersects observation invalid interval => sample_valid FALSE, no labels constructed',
  'audit_anchors':'288 per UTC date; future label-supported within-day anchors 00:30..23:30 reported separately',
  'conditional_denominator':'QC>=30 pixel-times; candidate 59 days for monthly/full decisions; excluded 2 days descriptive only',
  'day_screen':'daily ambiguity>5%, missing/invalid sources, version/grid transitions trigger explicit review; no automatic whole-day exclusion for a pixel or short gap',
  'mask_format':'daily compressed NetCDF4 partitions, native ROI grid, masks plus semantic classes and source times; unprocessed partitions absent and explicitly pending',
  'bias_percentile':'100 * count(reference59 <= excluded_value)/59; unavailable until all 59 reference days complete; fixed exclusion unaffected',
  'bias_area':'spherical cell area, mean Earth radius 6371.0088 km; sum QC>=30 cell area times 5/60 h at valid causal anchors; descriptive approximation',
  'stop':'finish full content audit, report gate, stop even if READY_FOR_LABEL_BUILD YES; no labels/models/GI/GO/Gamma/Gate1'}
 dump(OUT/'PREREGISTRATION_R4.json',policy);dump(OUT/'PREREGISTRATION_SEAL.json',{'sha256':sha(OUT/'PREREGISTRATION_R4.json'),'utc':policy['utc']})
 for n in ['sources','raw_roi','payload','daily','logs']:(OUT/n).mkdir(exist_ok=True)
 (ROOT/'SPATIOTEMPORAL_VALIDITY_MASK_R4').mkdir(exist_ok=True)
 print('R4 frozen',policy['utc'],flush=True)
def list_qc(day):
 rows=[];token=None;page=0
 while True:
  page+=1;path=OUT/'sources'/f'qc_listing_{day}_{page}.xml'
  if path.exists():body=path.read_bytes()
  else:
   query={'list-type':'2','prefix':f'CONUS/{QC}_00.50/{day.replace("-","")}/'}
   if token:query['continuation-token']=token
   for attempt in range(3):
    try:
     c=http.client.HTTPSConnection('noaa-mrms-pds.s3.amazonaws.com',timeout=60);c.request('GET','/?'+urllib.parse.urlencode(query));r=c.getresponse();body=r.read();assert r.status==200;c.close();break
    except Exception:
     if attempt==2:raise
   path.write_bytes(body)
  x=ET.fromstring(body);ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
  for e in x.findall('s:Contents',ns):
   key=e.findtext('s:Key',namespaces=ns)
   if not key.endswith('.grib2.gz'):continue
   stamp=dt.datetime.strptime(key.rsplit('_',1)[1].split('.')[0],'%Y%m%d-%H%M%S').replace(tzinfo=UTC).isoformat();assert stamp[:10]==day
   rows.append({'dataset':QC,'date':day,'key':key,'url':'https://noaa-mrms-pds.s3.amazonaws.com/'+key,'bytes':int(e.findtext('s:Size',namespaces=ns)),'start_utc':stamp,'end_utc':stamp,'etag':e.findtext('s:ETag',namespaces=ns)})
  if x.findtext('s:IsTruncated',namespaces=ns)=='false':break
  token=x.findtext('s:NextContinuationToken',namespaces=ns);assert token
 return rows
def main():
 freeze();db=connect()
 db.execute('CREATE TABLE IF NOT EXISTS inventory(key TEXT PRIMARY KEY,dataset TEXT,date TEXT,timestamp REAL,source TEXT)')
 db.execute('CREATE TABLE IF NOT EXISTS files(key TEXT PRIMARY KEY,dataset TEXT,date TEXT,qc_policy_version TEXT,readable INTEGER,reuse_version TEXT,result TEXT)')
 db.execute('CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT,utc TEXT,result TEXT)')
 db.commit()
 if not db.execute('SELECT count(*) FROM inventory').fetchone()[0]:
  with (ROOT/'audit/20260905T103339Z/OBJECT_INVENTORY.csv').open(encoding='utf-8',newline='') as f:
   for s in csv.DictReader(f):
    if s['dataset']=='GLM-L2-LCFA' and s['date'] in FIXED:continue
    s['bytes']=int(s['bytes']);db.execute('INSERT INTO inventory VALUES(?,?,?,?,?)',(s['key'],s['dataset'],s['date'],epoch(s['start_utc']),json.dumps(s)))
  db.commit()
 with ThreadPoolExecutor(12) as pool:
  for rr in pool.map(list_qc,DAYS):
   for s in rr:db.execute('INSERT OR IGNORE INTO inventory VALUES(?,?,?,?,?)',(s['key'],s['dataset'],s['date'],epoch(s['start_utc']),json.dumps(s)))
 db.commit()
 def save(r,version):
  s=r['source']
  if s['dataset']=='GLM-L2-LCFA' and s['date'] in FIXED:return
  r['qc_policy_version']='R4';r['reuse_version']=version
  db.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?,?,?,?)',(s['key'],s['dataset'],s['date'],'R4',int(r.get('readable',False)),version,json.dumps(r)))
 # Pixel caches have priority. Existing source payloads are not fetched again.
 for path,table,version in [('recovery_r3/RADAR_INPUT_R3.sqlite','qc_files','R3'),('recovery_r2/SEMANTIC_QC_R2.sqlite','mrms_files','R2')]:
  old=sqlite3.connect('file:'+(ROOT/path).as_posix()+'?mode=ro',uri=True)
  for txt, in old.execute('SELECT result FROM '+table):save(json.loads(txt),version)
  old.close()
 old=sqlite3.connect('file:'+(ROOT/'recovery_r1/full_content_qc/FILE_QC.sqlite').as_posix()+'?mode=ro',uri=True)
 for txt, in old.execute('SELECT result FROM files'):
  r=json.loads(txt)
  if r['source']['dataset']!='GLM-L2-LCFA':
   sent=r.get('sentinel_counts',{});missing=-1 if r['source']['dataset']=='EchoTop_30' else -99;nocov=-3 if missing==-1 else -999;n=r.get('roi_pixels',0);invalid=round(r.get('missing_fraction',1)*n)
   r['base_counts']={'0':n-invalid,'1':sent.get(str(missing),0),'2':sent.get(str(nocov),0),'3':invalid-sent.get(str(missing),0)-sent.get(str(nocov),0)}
   r['physical_quantiles']=r.get('numeric_summary');r['raw_source_sha256']=next((a['sha256'] for a in reversed(r.get('attempts',[])) if 'sha256' in a),None)
  save(r,'R1')
 old.close();db.commit()
 scope={p:{'objects':n,'bytes':b} for p,n,b in db.execute("SELECT dataset,count(*),sum(json_extract(source,'$.bytes')) FROM inventory GROUP BY dataset")}
 dump(OUT/'SCOPE.json',{'utc':now(),'datasets':scope,'objects':sum(v['objects'] for v in scope.values()),'bytes':sum(v['bytes'] for v in scope.values()),
   'reused_rows':db.execute('SELECT count(*) FROM files').fetchone()[0],'candidate_dates':59,'fixed_bias_dates':2,'original_inventory_sha256':sha(ROOT/'audit/20260905T103339Z/OBJECT_INVENTORY.csv')})
 print((OUT/'SCOPE.json').read_text(),flush=True);db.close()
if __name__=='__main__':main()
