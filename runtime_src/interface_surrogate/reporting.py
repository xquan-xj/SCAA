"""Aggregate fresh seed repetitions without mixing datasets, splits or protocols."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .experiments import fingerprint,read_json


def summarize_runs(run_dirs,*,stage):
    if stage not in {'val','test'}:
        raise ValueError('Summary stage must be val or test')
    groups=defaultdict(list)
    for directory in run_dirs:
        run=Path(directory);m=read_json(run/'manifest.json')
        if m['status']!='completed':
            raise ValueError('Cannot summarize an unfinished run')
        summary=read_json(run/'evaluation'/stage/'summary.json')
        if summary['stage']!=stage or summary['dataset_fingerprint']!=m['dataset_fingerprint']:
            raise ValueError('Evaluation does not match its run')
        for key in ['run_id','split_hash','prepared_metadata_sha256','metric_version']:
            if summary.get(key)!=m.get(key) or key not in summary or key not in m:
                raise ValueError('Evaluation artifact identity differs from its run: '+key)
        cfg=deepcopy(m['config']);cfg.pop('seed')
        group=fingerprint({'config':cfg,'dataset':m['dataset_fingerprint'],'split':m['split_hash'],
                           'prepared_metadata_sha256':m['prepared_metadata_sha256'],
                           'source':m['source_hash'],'metric':m['metric_version'],'stage':stage})
        groups[group].append((m,summary))
    rows=[]
    for group,entries in groups.items():
        m=entries[0][0];seeds=[x['config']['seed'] for x,_ in entries]
        if len(seeds)!=len(set(seeds)):
            raise ValueError('Repeated seed within the same experiment group')
        counts={s['n_samples'] for _,s in entries}
        if len(counts)!=1:
            raise ValueError('Seed runs have different evaluation sample counts')
        row={'group':group,'name':m['config']['name'],'model':m['config']['model']['kind'],
             'stage':stage,'dataset_fingerprint':m['dataset_fingerprint'],'split_hash':m['split_hash'],
             'prepared_metadata_sha256':m['prepared_metadata_sha256'],
             'source_hash':m['source_hash'],'metric_version':m['metric_version'],
             'n_seeds':len(entries),'seeds':','.join(map(str,sorted(seeds))),
             'n_samples_each':next(iter(counts))}
        metrics=sorted(set.intersection(*[set(s) for _,s in entries]))
        for key in metrics:
            values=[s[key] for _,s in entries]
            if key in {'n_samples','excluded_samples','stage','dataset_fingerprint','metric_version'}:
                continue
            if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):
                if not np.isfinite(values).all():raise ValueError('Non-finite summary statistic')
                row[key+'_mean']=float(np.mean(values))
                row[key+'_std']=None if len(values)<2 else float(np.std(values,ddof=1))
        rows.append(row)
    return rows
