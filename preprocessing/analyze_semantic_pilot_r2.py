"""Two-day pixel semantic masks and cross-product diagnostics, no labels."""
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__import__('os').environ['LIGHTNING_WORKSPACE']).resolve()
sys.path.insert(0,str(ROOT/'runtime_qc'))
import numpy as np
import netCDF4 as nc
UTC=dt.timezone.utc
PRODUCTS=['Reflectivity_-10C','EchoTop_30','MergedReflectivityComposite']
DATES=['2024-06-01','2024-06-02']
NAMES={0:'PHYSICAL_VALUE',1:'AMBIGUOUS_PRODUCT_MISSING',2:'NO_COVERAGE',3:'DECODE_INVALID',
       4:'VALID_NO_ECHO',5:'VALID_NO_ISOTHERM_ECHO',6:'VALID_NO_30DBZ_ECHO',
       7:'AMBIGUOUS_DERIVED_PRODUCT_MISSING',8:'SOURCE_FILE_UNAVAILABLE_AT_PILOT_BOUNDARY'}
def epoch(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def write_csv(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def classify(z,e,c,zb,eb,cb):
    # No sign-based rejection of reflectivity. Raw fields are never overwritten.
    zs=zb.copy();es=eb.copy();cs=cb.copy()
    cs[(cb==1)&(zb==1)&(eb==1)]=4
    # Conservative corroborated subset only; an isothermal -99 with a physical
    # composite is still ambiguous even where the composite is weak.
    zs[(zb==1)&(cs==4)]=5
    es[eb==1]=7
    es[(eb==1)&(((cb==0)&(c<30))|(cs==4))]=6
    return zs,es,cs

def test_semantics():
    # Regression vectors, not research observations; no fabricated GLM records.
    z=np.array([-5.,-99.,-99.,-999.,5.,-99.]);e=np.array([-1.,-1.,-1.,-3.,8.,-1.]);c=np.array([-7.,-99.,35.,-999.,-99.,10.])
    zb=np.array([0,1,1,2,0,1],dtype='u1');eb=np.array([1,1,1,2,0,1],dtype='u1');cb=np.array([0,1,0,2,1,0],dtype='u1')
    a,b,d=classify(z,e,c,zb,eb,cb)
    assert a.tolist()==[0,5,1,2,0,1] and b.tolist()==[6,6,7,2,0,6] and d.tolist()==[0,4,0,2,1,0]
    assert c[0]==-7 and c[1]==-99 and z[3]==-999

def main():
    test_semantics()
    con=sqlite3.connect('file:'+(ROOT/'recovery_r2/SEMANTIC_QC_R2.sqlite').as_posix()+'?mode=ro',uri=True)
    records=[json.loads(t) for t, in con.execute('SELECT result FROM mrms_files')];con.close()
    assert len(records)==4320 and all(r['readable'] for r in records),'Complete pilot reads required.'
    grouped={p:sorted([r for r in records if r['source']['dataset']==p],key=lambda r:r['source']['start_utc']) for p in PRODUCTS}
    assert all(len(v)==1440 for v in grouped.values())
    rows=[]
    for day in DATES:
        for p in PRODUCTS:
            rr=[r for r in grouped[p] if r['source']['date']==day];assert len(rr)==720
            for code in range(4):
                count=sum(r['base_counts'][str(code)] for r in rr)
                spatial=np.sum([r['spatial_tiles_1degree'][str(code)] for r in rr],axis=0)
                hourly={str(h):sum(r['base_counts'][str(code)] for r in rr if int(r['source']['start_utc'][11:13])==h)/(30*640000) for h in range(24)}
                rawcode=('-1' if p=='EchoTop_30' else '-99') if code==1 else ('-3' if p=='EchoTop_30' else '-999') if code==2 else 'OTHER_FINITE_NON_SENTINEL' if code==0 else 'NONFINITE_OR_GRIB_MISSING'
                rows.append({'date':day,'product':p,'value_category':rawcode,'base_class':['PHYSICAL_VALUE','PRODUCT_MISSING','NO_COVERAGE','DECODE_INVALID'][code],
                    'count':count,'fraction_of_roi':count/(720*640000),'fraction_of_files':sum(r['base_counts'][str(code)]>0 for r in rr)/720,
                    'n_files':720,'roi_pixels_per_file':640000,
                    'spatial_distribution':json.dumps({'tile_counts_north_to_south_west_to_east_8x8':spatial.tolist(),'tile_size_degree':1,'tile_pixels':10000}),
                    'temporal_distribution':json.dumps({'hourly_fraction_of_roi':hourly}),
                    'qc_policy_version':'R2_PILOT'})
    write_csv('MRMS_SENTINEL_DIAGNOSTICS_0601_0602.csv',rows)
    times={p:[epoch(r['source']['start_utc']) for r in grouped[p]] for p in PRODUCTS}
    start=int(epoch('2024-06-01T00:00:00+00:00'));timeline=list(range(start,start+172800,300))
    with np.load(grouped[PRODUCTS[0]][0]['roi_path']) as f:lat=f['lat'];lon=f['lon']
    maskpath=ROOT/'SPATIOTEMPORAL_VALIDITY_MASK_R2_PILOT.nc'
    out=nc.Dataset(maskpath,'w',format='NETCDF4')
    out.createDimension('timestamp',len(timeline));out.createDimension('lat',800);out.createDimension('lon',800)
    v=out.createVariable('timestamp','i8',('timestamp',));v[:]=timeline;v.units='seconds since 1970-01-01 00:00:00 UTC'
    out.createVariable('lat','f8',('lat',))[:]=lat;out.createVariable('lon','f8',('lon',))[:]=lon
    out.qc_policy_version='R2_PILOT';out.semantic_class_legend=json.dumps(NAMES)
    out.note='No ROI-wide missing-fraction threshold. Categorical no-echo is not numeric zero. Pilot only; not training inputs.'
    out.cross_product_alignment='Latest native field with timestamp<=t and age<300s. No interpolation or future field. Source time spread is retained.'
    variables={}
    for p in PRODUCTS:
        group=out.createGroup(p);vv={}
        for name in ['value_raw','value_semantic']:
            vv[name]=group.createVariable(name,'f8',('timestamp','lat','lon'),fill_value=np.nan,zlib=True,complevel=2,chunksizes=(1,100,100))
        for name in ['base_class','coverage_valid','physical_valid','semantic_class','semantic_valid']:
            vv[name]=group.createVariable(name,'u1',('timestamp','lat','lon'),zlib=True,complevel=2,chunksizes=(1,100,100))
        vv['threshold_truth']=group.createVariable('threshold_truth','i1',('timestamp','lat','lon'),zlib=True,complevel=2,chunksizes=(1,100,100))
        vv['source_timestamp']=group.createVariable('source_timestamp','i8',('timestamp',),fill_value=-1)
        vv['source_key']=group.createVariable('source_key',str,('timestamp',))
        group.value_semantic_definition='Numeric values only for PHYSICAL_VALUE; NaN for categorical or unavailable values. Interpret with semantic_class; no-echo threshold_truth=0.'
        group.threshold_truth_definition='-1 unknown; 0 FALSE; 1 TRUE. '+('Composite>=30dBZ' if p==PRODUCTS[2] else 'Z10>=15dBZ' if p==PRODUCTS[0] else '30dBZ echo exists')
        variables[p]=vv
    source_spread=out.createVariable('source_time_spread_seconds','i4',('timestamp',),fill_value=-1)
    totals=collections.defaultdict(collections.Counter);cross=collections.defaultdict(collections.Counter);crossrows=[];r1strict=collections.Counter()
    for j,t in enumerate(timeline):
        day=dt.datetime.fromtimestamp(t,UTC).date().isoformat();raws=[];bases=[];selected=[]
        for p in PRODUCTS:
            idx=bisect.bisect_right(times[p],t)-1
            r=grouped[p][idx] if idx>=0 and t-times[p][idx]<300 else None
            selected.append(r)
            if r:
                with np.load(r['roi_path']) as f:
                    assert np.array_equal(f['lat'],lat) and np.array_equal(f['lon'],lon)
                    raws.append(f['value_raw']);bases.append(f['base_class'])
            else:raws.append(np.full((800,800),np.nan));bases.append(np.full((800,800),3,dtype='u1'))
        sems=classify(*raws,*bases)
        for i,r in enumerate(selected):
            if r is None:sems[i][:]=8
        z,e,c=raws;zb,eb,cb=bases
        masks={'comp_missing_and_z10_physical':(cb==1)&(zb==0),
               'comp_missing_and_et30_physical':(cb==1)&(eb==0),
               'comp_missing_and_both_derived_missing':(cb==1)&(zb==1)&(eb==1),
               'z10_missing_and_comp_ge30':(zb==1)&(cb==0)&(c>=30),
               'z10_missing_and_comp_lt30':(zb==1)&(cb==0)&(c<30),
               'et30_missing_and_comp_ge30':(eb==1)&(cb==0)&(c>=30),
               'et30_missing_and_comp_lt30':(eb==1)&(cb==0)&(c<30),
               'et30_physical_and_comp_lt30':(eb==0)&(cb==0)&(c<30)}
        near_strong=np.zeros(c.shape,dtype=bool);strong=(cb==0)&(c>=30)
        near_strong[1:]|=strong[:-1];near_strong[:-1]|=strong[1:];near_strong[:,1:]|=strong[:,:-1];near_strong[:,:-1]|=strong[:,1:]
        masks['comp_missing_adjacent_4_neighbor_ge30']=(cb==1)&near_strong
        row={'timestamp':dt.datetime.fromtimestamp(t,UTC).isoformat(),'date':day,'qc_policy_version':'R2_PILOT'}
        if all(selected):
            native=[epoch(r['source']['start_utc']) for r in selected];spread=int(max(native)-min(native));source_spread[j]=spread
        else:spread=-1;source_spread[j]=-1
        row['source_time_spread_seconds']=spread
        for name,mask in masks.items():row[name]=int(mask.sum());cross[day][name]+=row[name]
        crossrows.append(row)
        for p,r,raw,base,sem in zip(PRODUCTS,selected,raws,bases,sems):
            vv=variables[p];physical=(base==0);coverage=np.isin(base,[0,1]);valid=np.isin(sem,[0,4,5,6])
            truth=np.full(raw.shape,-1,dtype='i1');truth[np.isin(sem,[4,5,6])]=0
            truth[physical]=(raw[physical]>=(15 if p==PRODUCTS[0] else 30)) if p!='EchoTop_30' else 1
            vv['value_raw'][j]=raw;vv['value_semantic'][j]=np.where(physical,raw,np.nan)
            for name,array in [('base_class',base),('coverage_valid',coverage),('physical_valid',physical),('semantic_class',sem),('semantic_valid',valid),('threshold_truth',truth)]:vv[name][j]=array
            vv['source_timestamp'][j]=int(epoch(r['source']['start_utc'])) if r else -1;vv['source_key'][j]=r['source']['key'] if r else ''
            for code in NAMES:totals[(day,p)][NAMES[code]]+=int(np.count_nonzero(sem==code))
            totals[(day,p)]['negative_physical_pixels']+=int(np.count_nonzero(physical&(raw<0)))
            totals[(day,p)]['denominator_pixels']+=raw.size
            r1strict[(day,p)]+=bool(physical.all())
            assert sum(np.count_nonzero(sem==code) for code in NAMES)==raw.size
            assert np.all(truth[np.isin(sem,[1,2,3,7,8])]==-1)
        if j%48==0:out.sync();print('Semantic mask slots',j,'/',len(timeline),flush=True)
    out.close();write_csv('MRMS_CROSS_PRODUCT_DIAGNOSTICS_R2_PILOT.csv',crossrows)
    result={'qc_policy_version':'R2_PILOT','mrms_native_files':4320,'timeline_slots':576,
        'spatial_mask_path':str(maskpath),'source_time_spread_max_seconds':max(r['source_time_spread_seconds'] for r in crossrows),
        'pixel_results':[{'date':day,'product':p,**dict(totals[(day,p)]),'r1_strict_window_fraction':r1strict[(day,p)]/288} for day in DATES for p in PRODUCTS],
        'cross_product_counts':{day:dict(cross[day]) for day in DATES},'classifier_tests_passed':True,
        'interpretation':'Corroborated no-echo is a product-semantic inference. Uncorroborated missing remains ambiguous; not a physical zero or sensor outage.'}
    (ROOT/'recovery_r2/MRMS_SEMANTICS_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
