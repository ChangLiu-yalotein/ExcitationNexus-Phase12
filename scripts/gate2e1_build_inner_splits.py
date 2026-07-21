#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from gate2e1_common import ROOT, PRIMARY, config, load_protocol, sha, write_json

def stable_key(value,seed): return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
def main():
    c=config(); outdir=ROOT/c['local_root']/ 'inner_splits'; registry_path=ROOT/'data_registry/gate2e1_inner_split_registry.json'
    if registry_path.exists() or outdir.exists(): raise RuntimeError('inner split already frozen')
    outdir.mkdir(parents=True)
    registry={'version':'gate2e1_inner_split_registry_v1','seed':c['inner_split']['seed'],'protocols':{},'official_validation_used':False,'test_access':False}
    for protocol,spec in c['protocols'].items():
        frame,_,_=load_protocol(protocol,False); unit=spec['inner_unit']
        units=frame.groupby(unit).agg(records=('molecule_id','size'),target=(PRIMARY,'mean')).reset_index()
        units['bin']=pd.qcut(units.target.rank(method='first'),q=min(c['inner_split']['primary_quantile_bins'],len(units)),labels=False)
        forced=set(frame.loc[frame.historical_status.eq('HISTORICAL_TRAIN_OVERLAP'),unit])
        checkpoint=set()
        for _,block in units.groupby('bin'):
            eligible=[u for u in block[unit] if u not in forced]; eligible=sorted(eligible,key=lambda x:stable_key(x,c['inner_split']['seed']))
            n=max(1,int(round(len(eligible)*c['inner_split']['checkpoint_fraction']))); checkpoint.update(eligible[:n])
        frame['inner_partition']=np.where(frame[unit].isin(checkpoint),'inner_checkpoint','inner_fit')
        if set(frame.loc[frame.inner_partition.eq('inner_checkpoint'),unit]) & set(frame.loc[frame.inner_partition.eq('inner_fit'),unit]): raise RuntimeError('inner leakage')
        if frame.loc[frame.historical_status.eq('HISTORICAL_TRAIN_OVERLAP'),'inner_partition'].ne('inner_fit').any(): raise RuntimeError('forced fit failure')
        output_columns=list(dict.fromkeys(['molecule_id','inner_partition',unit,'structure_group_id_v1','group_weight']))
        output=outdir/f'{protocol}_inner_split.parquet'; frame[output_columns].sort_values('molecule_id').to_parquet(output,index=False)
        shuffled=frame.sample(frac=1,random_state=7); assert set(shuffled.loc[shuffled[unit].isin(checkpoint),'molecule_id'])==set(frame.loc[frame.inner_partition.eq('inner_checkpoint'),'molecule_id'])
        registry['protocols'][protocol]={'path':str(output.relative_to(ROOT)),'sha256':sha(output),'unit':unit,'fit_records':int(frame.inner_partition.eq('inner_fit').sum()),'checkpoint_records':int(frame.inner_partition.eq('inner_checkpoint').sum()),'fit_units':int(frame.loc[frame.inner_partition.eq('inner_fit'),unit].nunique()),'checkpoint_units':int(frame.loc[frame.inner_partition.eq('inner_checkpoint'),unit].nunique()),'unit_overlap':0,'historical_overlap_checkpoint':0,'row_order_invariant':True}
    write_json(registry_path,registry); print(json.dumps(registry,indent=2))
if __name__=='__main__':main()
