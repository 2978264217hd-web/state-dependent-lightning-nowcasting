from r10v2_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,FancyBboxPatch
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#888888'
F=OUT/'R10_V2_FIGURES';F.mkdir(parents=True,exist_ok=True)
MAP=[];current={}
def V(k,path=None):
 v=val(k,path);current.setdefault('values',[]).append({'key':k,'path':path,'value':v,'source_path':K[k]['source_path'],'source_selector':K[k]['source_selector'],'source_sha256':K[k]['source_sha256']});return v
def start(n,description):
 global current
 current={'figure':n,'description':description,'values':[],'new_metrics_computed':False,'confidence_intervals_recomputed':False}
def save(fig,n):
 fig.savefig(F/f'Figure_{n}.pdf',bbox_inches='tight');fig.savefig(F/f'Figure_{n}.svg',bbox_inches='tight');fig.savefig(F/f'Figure_{n}.png',dpi=450,bbox_inches='tight')
 current['files']={f'Figure_{n}.{e}':sha(F/f'Figure_{n}.{e}') for e in ['pdf','svg','png']};MAP.append(current);plt.close(fig)
def title(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=10)
def grid(ax):ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True)
def bars(ax,labels,series,ylabel='CSI',legend=True):
 x=np.arange(len(labels));width=.75/len(series)
 for j,(name,values,col) in enumerate(series):ax.bar(x+(j-(len(series)-1)/2)*width,values,width,label=name,color=col,edgecolor='black',lw=.4,hatch=['','//','..'][j])
 ax.set_xticks(x,labels);ax.set_ylabel(ylabel);grid(ax)
 if legend:ax.legend(frameon=False)
def forest(ax,labels,keys,paths=None,micro=False):
 for i,(lab,k) in enumerate(zip(labels,keys)):
  delta=V(k[0]);ci=V(k[1],None if paths is None else paths[i]);scale=1000 if micro else 1
  ax.errorbar(delta*scale,i,xerr=np.array([[delta-ci[0]],[ci[1]-delta]])*scale,fmt='o',color=BLUE if i==0 else ORANGE,capsize=4,ms=6,lw=1.5)
 ax.set_yticks(range(len(labels)),labels);ax.axvline(0,c='gray',ls='--',lw=.8);ax.set_xlabel('ΔCSI (STATE − GLOBAL)'+(' × 10³' if micro else ''));ax.set_ylim(len(labels)-.4,-.6);ax.grid(axis='x',color='#dddddd',lw=.6)
def box(ax,xy,w,h,s,col='#eef5f8'):
 p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=.012',ec=BLUE,fc=col,lw=.9);ax.add_patch(p);ax.text(xy[0]+w/2,xy[1]+h/2,s,ha='center',va='center',fontsize=9)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#333333','lw':1.2})
start(1,'Definition schematic: frozen diagnosis → validation-only decision selection → independent confirmation. No empirical values or inferred causal arrows.')
fig,ax=plt.subplots(figsize=(7,4.8));ax.set(xlim=(-.03,1.03),ylim=(-.02,1.04));ax.axis('off')
box(ax,(.01,.71),.28,.24,'Common population\nMET / LTG / FUSION\nFrozen inputs\nand labels')
box(ax,(.36,.71),.28,.24,'Historical state H\nLocal onset: H = 0\nOngoing: H = 1')
box(ax,(.71,.71),.28,.24,'State-specific\nverification\nCSI / PR-AUC\nPersistence comparison')
arrow(ax,(.29,.83),(.36,.83));arrow(ax,(.64,.83),(.71,.83))
box(ax,(.08,.35),.36,.25,'Existing diagnostics\nInput masks\nFailed suppression\nRanking versus decision')
box(ax,(.56,.35),.36,.25,'Validation data only\nRaw / transformed\nstate thresholds\nSelect and freeze final rule')
arrow(ax,(.85,.71),(.74,.58));arrow(ax,(.5,.71),(.26,.58));arrow(ax,(.44,.475),(.56,.475))
box(ax,(.22,.03),.56,.21,'Independent 2025 confirmation\nReuse model, normalization, and decision\nBenefits, false alarms, and label sensitivity',col='#fcf0e8')
arrow(ax,(.74,.37),(.60,.24));ax.text(.5,1,'Study framework',ha='center',fontsize=12,fontweight='bold');save(fig,1)
start(2,'Frozen domain, spatial/time windows, four classes, and study periods. Definition constants only; no new geographic data.')
current['definition_source']=str(ROOT/'R10_PAPER_DRAFT'/'R10_NUMERIC_SOURCE_REGISTRY.json')
fig,axs=plt.subplots(2,2,figsize=(7,5.8),layout='constrained');a,b,c,d=axs.ravel()
title(a,'(a) Fixed verification domain');a.add_patch(Rectangle((-100,31),8,8,fc='#eef5f8',ec=BLUE,lw=2));a.set(xlim=(-101,-91),ylim=(30,40),xlabel='Longitude (°W)',ylabel='Latitude (°N)');a.set_xticks([-100,-96,-92],['100','96','92']);a.text(-96,35,'South-central\nUnited States',ha='center',va='center');a.grid(color='#ddd',lw=.5)
title(b,'(b) Neighborhoods and endpoints');b.axis('off');b.set(xlim=(-1.1,1.1),ylim=(-1.7,1.1));b.set_aspect('equal',adjustable='box');b.add_patch(Circle((0,.2),.7,fill=False,ec=BLUE,lw=1.8));b.add_patch(Circle((0,.2),.35,fill=False,ec=ORANGE,lw=1.8,ls='--'));b.text(0,1.00,'History: 20 km',ha='center',color=BLUE);b.text(0,.015,'Future\n10 km',ha='center',fontsize=8,color=ORANGE);b.text(0,-.85,'H: [t − 30 min, t]\nY: (t, t + 30 min]',ha='center',va='center');b.text(0,-1.45,'History includes t; target excludes t',ha='center',fontsize=8)
title(c,'(c) Four verification classes');c.axis('off');tb=c.table(cellText=[['I−\nLocal-onset negative','I+\nLocal-onset positive'],['O−\nOngoing negative','O+\nOngoing positive']],rowLabels=['H = 0','H = 1'],colLabels=['Y = 0','Y = 1'],loc='center',cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(8);tb.scale(1,3.6)
title(d,'(d) Independent periods');d.axis('off');d.text(.05,.82,'Development: June–July 2024\nGOES-16 / MRMS\nTraining → validation → test',va='top',fontsize=10);d.text(.05,.40,'Confirmation: June–July 2025\nGOES-19 / MRMS\nAll 61 dates; frozen rule',va='top',fontsize=10);d.text(.05,.03,'Year, weather, and platform are confounded.',fontsize=8);save(fig,2)
start(3,'2024 test baseline CSI and historical-lightning-associated gains; frozen Gamma interval only.')
fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained');a,b=axs
title(a,'(a) Baseline CSI: 2024 test');bars(a,['MET','LTG','FUSION','B0','B1'],[('ALL',[V(x) for x in ['K028','K001','K055','K103','K106']],GRAY),('I',[V(x) for x in ['K037','K010','K064','K104','K107']],ORANGE),('O',[V(x) for x in ['K046','K019','K073','K105','K108']],BLUE)]);a.set_ylim(0,.95)
title(b,'(b) Fusion − MET contrasts');values=[V(k) for k in ['K082','K083','K084']];b.axhline(0,c='gray',ls='--',lw=.8);b.scatter([0,1,2],values,c=[ORANGE,BLUE,GREEN],s=45);ci=V('K085');b.errorbar(2,values[2],yerr=[[values[2]-ci[0]],[ci[1]-values[2]]],fmt='none',c=GREEN,capsize=5);b.set_xticks([0,1,2],['G_I','G_O','Gamma']);b.set_ylabel('CSI contrast');grid(b)
for x,y in enumerate(values):b.annotate(f'{y:.6f}',(x,ci[1] if x==2 else y),xytext=(0,-18 if y<0 else 10),textcoords='offset points',ha='center',fontsize=8)
b.set_ylim(-.10,.25);b.set_xlim(-.35,2.35);save(fig,3)
start(4,'Existing 2024 test masking/ranking diagnostics and separate validation operating-point comparison.')
fig,axs=plt.subplots(2,2,figsize=(7,5.6),layout='constrained');a,b,c,d=axs.ravel();labs=['MET','LTG','FUSION','GLM\nmask','MRMS\nmask']
title(a,'(a) Local-onset CSI: test');bars(a,labs,[('I CSI',[V(x) for x in ['K037','K010','K064','K091','K097']],BLUE)],legend=False);a.set_ylim(0,.12)
title(b,'(b) Local-onset ranking: test');bars(b,labs,[('I PR-AUC',[V(x) for x in ['K038','K011','K065','K092','K098']],BLUE)],'PR-AUC',False);b.set_ylim(0,.34)
title(c,'(c) Normal − masked CSI: test');bars(c,['GLM history','MRMS'],[('I',[V('K093'),V('K099')],ORANGE),('O',[V('K096'),V('K102')],BLUE)],'CSI change');c.axhline(0,c='gray',lw=.8)
title(d,'(d) Validation decision comparison');bars(d,['GLOBAL','Raw\nSTATE','Transf.\nSTATE','History\nsuppression'],[('I CSI',[V(x) for x in ['K115','K129','K143','K228']],ORANGE)],legend=False);d.set_ylim(0,.25);save(fig,4)
start(5,'Frozen 2024 test GLOBAL versus STATE and paired frozen intervals; no bootstrap rerun.')
fig,axs=plt.subplots(1,3,figsize=(7,3.15),layout='constrained');a,b,c=axs
title(a,'(a) Test CSI');bars(a,['ALL','I','O'],[('GLOBAL',[V(x) for x in ['K178','K185','K192']],BLUE),('STATE',[V(x) for x in ['K199','K206','K213']],ORANGE)]);a.set_ylim(0,1)
title(b,'(b) Local-onset change');forest(b,['I'],[('K112','K114')],['DELTA_I_TEST']);b.set_xlim(0,.23)
title(c,'(c) Ongoing change');forest(c,['O'],[('K113','K114')],['DELTA_O_TEST'],True);c.set_xlim(-.35,.25);save(fig,5)
start(6,'2025 independent confirmation: primary and QF=0 CSI, frozen paired intervals, detection/false alarms and mixed reliability.')
fig,axs=plt.subplots(2,3,figsize=(7,5.8),layout='constrained');a,b,c,d,e,f=axs.ravel()
title(a,'(a) Primary CSI');bars(a,['ALL','I','O'],[('GLOBAL',[V(x) for x in ['K253','K264','K275']],BLUE),('STATE',[V(x) for x in ['K286','K297','K308']],ORANGE)]);a.set_ylim(0,1)
title(b,'(b) Local-onset change');forest(b,['Primary','QF=0'],[('K239','K241'),('K242','K244')],['DELTA_I_R83','DELTA_I_R83']);b.set_xlim(0,.19)
title(c,'(c) Ongoing change');forest(c,['Primary','QF=0'],[('K240','K241'),('K243','K244')],['DELTA_O_R83','DELTA_O_R83'],True);c.set_xlim(-.35,.35)
title(d,'(d) QF=0 CSI');bars(d,['I','O'],[('GLOBAL',[V('K330'),V('K341')],BLUE),('STATE',[V('K363'),V('K374')],ORANGE)]);d.set_ylim(0,1)
title(e,'(e) Primary onset POD / FAR');bars(e,['POD','FAR'],[('GLOBAL',[V('K266'),V('K267')],BLUE),('STATE',[V('K299'),V('K300')],ORANGE)],'Score');e.set_ylim(0,1)
title(f,'(f) Primary reliability');bars(f,['I\nBrier','O\nBrier','I\nECE','O\nECE'],[('GLOBAL',[V(x) for x in ['K268','K279','K269','K280']],BLUE),('STATE',[V(x) for x in ['K301','K312','K302','K313']],ORANGE)],'Score (lower is better)');save(fig,6)
dump(OUT/'R10_V2_FIGURE_VALUES.json',MAP)
print('Six PDF/SVG/PNG figures generated from frozen values.')
