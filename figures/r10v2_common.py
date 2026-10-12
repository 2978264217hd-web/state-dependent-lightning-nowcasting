from pathlib import Path
import json,re,hashlib,html,csv,ast
W=Path(__import__('os').environ['LIGHTNING_DATA_ROOT']).resolve() / 'figure_source_tables'
OUT=Path(__import__('os').environ.get('LIGHTNING_FIGURE_OUTPUT', 'reproduced_figures')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
K=js(W/'key_results.json')
def parse(v):
 for f in [json.loads,ast.literal_eval]:
  try:return f(v)
  except (ValueError,SyntaxError,TypeError):pass
 return v
def val(k,path=None):
 v=parse(K[k]['value_full_precision'])
 if path:
  for x in path.split('/'):v=v[int(x)] if isinstance(v,list) else v[x]
 return v
def fmt(v,integer=False):
 if isinstance(v,(list,tuple)):return '['+', '.join(fmt(x) for x in v)+']'
 if integer:return f'{int(v):,}'
 return f'{v:.6f}' if isinstance(v,(float,int)) else str(v)
def table_data():
 tabs=js(W/'master_docx_readback.json')['tables']
 tabs[0][0]=['Dataset','Dates','Usable samples','I+ / I−','O+ / O−','Usable events','Active dates']
 for r,n in zip(tabs[0][1:],['2024 Training','2024 Validation','2024 Test','2025 Confirmation']):r[0]=n
 tabs[1][0]=['Model / diagnostic','ALL CSI','I CSI','O CSI','I PR-AUC','O PR-AUC']
 for r,n in zip(tabs[1][1:],['MET','LTG','FUSION','Persistence B0','History baseline B1','FUSION: GLM masked','FUSION: MRMS masked']):r[0]=n
 tabs[2][0]=['Dataset / rule','θI','θO','I CSI','O CSI','ALL CSI']
 for r,n in zip(tabs[2][1:],['Validation / GLOBAL','Validation / raw STATE','Validation / transformed STATE','Validation / M_HSUPPRESS','Test / GLOBAL','Test / final STATE']):r[0]=n
 tabs[3][0]=['Labels / state','CSI\nG / S','ΔCSI; 95% interval','PR-AUC\nG / S','POD / FAR','Brier / ECE']
 for r,n in zip(tabs[3][1:],['Primary / ALL','Primary / I','Primary / O','QF=0 / I','QF=0 / O']):r[0]=n
 tabs[3][1][2]='Decrease; see text'
 return tabs
def mdtable(t):return '\n'.join(['| '+' | '.join(x.replace('\n','<br>') for x in r)+' |' for r in [t[0],['---']*len(t[0])]+t[1:]])
def references():
 rr=js(W/'reference_candidates.json');out=[]
 for z in rr:
  m=z['metadata'];authors=m.get('author',[])
  y=m.get('published-print',m.get('published',m.get('issued')))['date-parts'][0][0]
  title=html.unescape(re.sub('<[^>]*>','',m['title'][0])).replace('\xa0',' ')
  # Crossref deposits Brier's name/title in all capitals; normalize typography only.
  if z['id']=='Brier1950':
   authors=[dict(a,family='Brier') for a in authors];title='Verification of forecasts expressed in terms of probability'
  names=[a.get('family',a.get('name','')) for a in authors]
  short=names[0]+(' et al.' if len(names)>2 else ' and '+names[1] if len(names)==2 else '')
  out.append({'id':z['id'],'title':title,'authors':authors,'year':y,'journal':m.get('container-title',[''])[0], 'volume':m.get('volume',''),'issue':m.get('issue',''),'pages':m.get('page',m.get('article-number','')),'doi':m['DOI'],'url':'https://doi.org/'+m['DOI'],'short':short,'role':z['role'],'metadata_sha256':sha(W/f"crossref_{z['id']}.json"),'metadata_verified':True})
 out.append({'id':'NOAATransition','title':"NOAA’s GOES-19 satellite now operational, providing critical new data to forecasters",'authors':[{'name':'NOAA'}],'year':2025,'journal':'NOAA News','volume':'','issue':'','pages':'','doi':None,'url':'https://www.noaa.gov/news-release/noaas-goes-19-satellite-now-operational-providing-critical-new-data-to-forecasters','short':'NOAA','role':'Official dated record of the operational GOES-East transition, not a causal comparison','metadata_verified':True,'published_date':'2025-04-07'})
 return out
