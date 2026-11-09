"""Explicit research steps; a matrix is a plan unless --execute is supplied."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .baselines import BASELINE_KINDS,fit_baseline
from .data import prepare_dataset,verify_prepared_dataset
from .evaluation import benchmark_run,evaluate_run
from .experiments import file_hash,freeze_protocol,read_json,write_json
from .fem import (calibrate_local_threshold,compare_pressures,export_case,
                  metric_mechanics_correlations,read_pressure_csv)
from .reporting import summarize_runs
from .settings import resolve_config
from .splits import make_split
from .training import train_experiment
from .screening import baseline_candidates,screen_baselines

ROOT=Path(__file__).resolve().parents[2]


def path(value):
    p=Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def emit(value):
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def csv_rows(filename):
    with path(filename).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('CSV input is empty')
    return rows


def write_csv(filename,rows):
    if not rows:raise ValueError('No rows to write')
    out=path(filename);out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with out.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def strict_bool(value):
    text=str(value).strip().lower()
    if text in {'true','1','yes'}:return True
    if text in {'false','0','no'}:return False
    raise ValueError('mechanics_pass must be an explicit boolean: true/false or1/0')


def verify_resources(raw_dir):
    root=path(raw_dir);manifest=read_json(root/'resources.json')
    hashes=manifest['files_sha256']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='resources.json'}
    if actual!=set(hashes):raise ValueError('Raw resource inventory differs from its manifest')
    for relative,digest in hashes.items():
        target=(root/relative).resolve()
        if not target.is_relative_to(root) or file_hash(target)!=digest:
            raise ValueError('Raw resource SHA256 mismatch: '+relative)
    return {'verified_files':len(hashes),'samples':manifest['sample_count'],
            'label_policy':manifest.get('label_policy','history_column11')}


def parser():
    p=argparse.ArgumentParser(description='Clean EAAI interface morphology study')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('verify-resources');a.add_argument('--raw',default='data/raw')
    a=sub.add_parser('prepare');a.add_argument('--config',default='configs/data.json');a.add_argument('--output')
    a=sub.add_parser('split')
    a.add_argument('--data',required=True);a.add_argument('--output',required=True)
    a.add_argument('--scheme',choices=['random','pressure_block','preload_block'],default='random')
    a.add_argument('--seed',type=int,default=8012)
    a.add_argument('--test-fraction',type=float,default=.1);a.add_argument('--val-fraction',type=float,default=.1)
    a.add_argument('--pressure-block',type=float,nargs=2,default=[27.5,28.5])
    a.add_argument('--preload-block',type=float,nargs=2,default=[3300,3500])
    for name in ['train','baseline','suite']:
        a=sub.add_parser(name)
        a.add_argument('--matrix' if name=='suite' else '--config',required=True)
        a.add_argument('--data',required=True);a.add_argument('--split',required=True)
        a.add_argument('--runs',default='runs');a.add_argument('--device')
        if name=='suite':
            a.add_argument('--execute',action='store_true');a.add_argument('--evaluate-val',action='store_true')
        else:a.add_argument('--seed',type=int)
    a=sub.add_parser('evaluate');a.add_argument('--run',required=True)
    a.add_argument('--stage',choices=['val','test'],required=True);a.add_argument('--lock');a.add_argument('--data');a.add_argument('--device')
    a=sub.add_parser('freeze');a.add_argument('--runs',nargs='+',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('export-fem');a.add_argument('--data',required=True);a.add_argument('--predictions',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('compare-fem');a.add_argument('--prediction',required=True);a.add_argument('--reference',required=True)
    a.add_argument('--threshold',type=float,required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('calibrate');a.add_argument('--scores',required=True);a.add_argument('--mechanics',required=True)
    a.add_argument('--stage',choices=['val'],required=True);a.add_argument('--min-samples',type=int,default=10)
    a.add_argument('--max-false-accept-fraction',type=float,default=0.);a.add_argument('--output',required=True)
    a=sub.add_parser('correlate');a.add_argument('--input',required=True)
    a.add_argument('--mechanics-key',default='pressure_rmse_MPa');a.add_argument('--output',required=True)
    a=sub.add_parser('summarize');a.add_argument('--runs',nargs='+',required=True)
    a.add_argument('--stage',choices=['val','test'],required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('benchmark');a.add_argument('--run',required=True);a.add_argument('--device')
    a.add_argument('--warmup',type=int,default=10);a.add_argument('--repeats',type=int,default=100);a.add_argument('--output',required=True)
    a=sub.add_parser('screen-baselines');a.add_argument('--data',required=True);a.add_argument('--split',required=True)
    a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=8012)
    a.add_argument('--epochs',type=int,default=100);a.add_argument('--execute',action='store_true')
    return p


def launch(args,config):
    cfg=resolve_config(config)
    split=read_json(path(args.split))
    if cfg['model']['kind'] in BASELINE_KINDS:
        return fit_baseline(path(args.data),split,cfg,path(args.runs))
    return train_experiment(path(args.data),split,cfg,path(args.runs),device=args.device)


def main(argv=None):
    args=parser().parse_args(argv);command=args.command
    if command=='verify-resources':return emit(verify_resources(args.raw))
    if command=='screen-baselines':
        candidates=baseline_candidates(args.seed,args.epochs)
        if not args.execute:
            return emit({'execute':False,'stage':'val','test_evaluated':False,'candidates':candidates})
        result=screen_baselines(path(args.data),read_json(path(args.split)),path(args.output),candidates)
        return emit({'stage':'val','test_evaluated':False,'candidates':len(result['candidates']),
                     'selected':result['selected']})
    if command=='prepare':
        cfg=read_json(path(args.config));cfg['raw_dir']=path(cfg['raw_dir'])
        cfg['output_dir']=path(args.output or cfg['output_dir'])
        meta=prepare_dataset(**cfg)
        return emit({'samples':len(meta['samples']),'dataset_fingerprint':meta['dataset_fingerprint']})
    if command=='split':
        meta=verify_prepared_dataset(path(args.data))
        split=make_split(meta['samples'],scheme=args.scheme,seed=args.seed,test_fraction=args.test_fraction,
                         val_fraction=args.val_fraction,pressure_block=args.pressure_block,preload_block=args.preload_block)
        split.update(dataset_fingerprint=meta['dataset_fingerprint'],prepared_metadata_sha256=meta['prepared_metadata_sha256'])
        write_json(path(args.output),split,exclusive=True)
        return emit({key:len(split[key]) for key in ['train','val','test']})
    if command in {'train','baseline'}:
        cfg=read_json(path(args.config))
        if args.seed is not None:cfg['seed']=args.seed
        baseline=cfg.get('model',{}).get('kind') in BASELINE_KINDS
        if baseline!=(command=='baseline'):raise ValueError('Choose baseline for numerical regressors, train for neural models')
        return emit({'run':str(launch(args,cfg))})
    if command=='suite':
        matrix=read_json(path(args.matrix));configs=matrix['configs'];seeds=matrix['seeds']
        if not configs or not seeds or len(configs)!=len(set(configs)) or len(seeds)!=len(set(seeds)):
            raise ValueError('Matrix configs and seeds must be nonempty and unique')
        plans=[]
        for filename in configs:
            for seed in seeds:
                cfg=read_json(path(filename));cfg['seed']=seed;cfg=resolve_config(cfg)
                plans.append({'config_file':filename,'seed':seed,'name':cfg['name'],'model':cfg['model']['kind'],'resolved':cfg})
        if not args.execute:return emit({'execute':False,'planned_runs':plans,'test_evaluation':False})
        results=[]
        for plan in plans:
            run=launch(args,plan['resolved'])
            if args.evaluate_val:evaluate_run(run,stage='val',device=args.device)
            results.append(str(run))
        return emit({'runs':results,'test_evaluation':False})
    if command=='evaluate':
        return emit(evaluate_run(path(args.run),stage=args.stage,lock_path=path(args.lock) if args.lock else None,
                                  processed_dir=path(args.data) if args.data else None,device=args.device))
    if command=='freeze':return emit(freeze_protocol([path(v) for v in args.runs],path(args.output)))
    if command=='export-fem':return emit(export_case(path(args.data),path(args.predictions),path(args.output)))
    if command=='compare-fem':
        result=compare_pressures(read_pressure_csv(path(args.prediction)),read_pressure_csv(path(args.reference)),effective_threshold_MPa=args.threshold)
    elif command=='calibrate':
        scores=csv_rows(args.scores);mechanics=csv_rows(args.mechanics)
        if any(r.get('stage')!='val' for r in scores+mechanics):
            raise ValueError('Calibration tables must explicitly identify validation stage (stage=val)')
        paired={r['id']:strict_bool(r['mechanics_pass']) for r in mechanics}
        score_ids=[r['id'] for r in scores]
        if len(paired)!=len(mechanics) or len(set(score_ids))!=len(scores) or set(paired)!=set(score_ids):
            raise ValueError('Calibration requires matching unique case IDs in both validation tables')
        result=calibrate_local_threshold([float(r['local_score_v2']) for r in scores],
                                         [paired[r['id']] for r in scores],stage=args.stage,
                                         min_samples=args.min_samples,max_false_accept_fraction=args.max_false_accept_fraction)
        result['calibration_source']={'stage':'val','ids':score_ids,
            'scores_sha256':file_hash(path(args.scores)),'mechanics_sha256':file_hash(path(args.mechanics)),
            'run_ids':sorted({r['run_id'] for r in scores if r.get('run_id')})}
    elif command=='correlate':
        rows=csv_rows(args.input)
        for row in rows:
            for k,v in list(row.items()):
                if v is None or not v.strip():row[k]=None
        result=metric_mechanics_correlations(rows,args.mechanics_key)
    elif command=='summarize':
        rows=summarize_runs([path(v) for v in args.runs],stage=args.stage)
        write_csv(args.output,rows);return emit({'groups':len(rows),'output':str(path(args.output))})
    elif command=='benchmark':
        result=benchmark_run(path(args.run),device=args.device,warmup=args.warmup,repeats=args.repeats)
    else:raise AssertionError('Unhandled command')
    write_json(path(args.output),result,exclusive=True);return emit(result)
