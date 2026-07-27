#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = Path('/home/changliu/miniconda3/envs/chemprop-v2-g1c')
RUN = ROOT / 'runs/gate2g1c_chemprop_v2_formal'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='unified_grouped_5fold_oof')
    ap.add_argument('--arm', choices=['blind', 'observable_provenance'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    data_dir = RUN / 'data' / args.protocol / args.arm / f'seed{args.seed}' / f'fold{args.fold}'
    train_csv = data_dir / 'chemprop_train.csv'
    if not train_csv.exists():
        raise SystemExit(f'missing {train_csv}')
    out_dir = RUN / 'formal_runs' / args.protocol / args.arm / f'seed{args.seed}' / f'fold{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ENV / 'bin/chemprop'), 'train',
        '-i', str(train_csv),
        '-o', str(out_dir),
        '--smiles-columns', 'smiles',
        '--target-columns', 'y',
        '--weight-column', 'weight',
        '--splits-column', 'split',
        '--epochs', '80',
        '--patience', '10',
        '--warmup-epochs', '2',
        '--init-lr', '0.0001',
        '--max-lr', '0.001',
        '--final-lr', '0.0001',
        '--batch-size', '64',
        '--num-workers', '0',
        '--accelerator', 'gpu',
        '--devices', '1',
        '--message-hidden-dim', '300',
        '--depth', '3',
        '--dropout', '0.0',
        '--aggregation', 'norm',
        '--aggregation-norm', '100',
        '--ffn-hidden-dim', '300',
        '--ffn-num-layers', '1',
        '--metrics', 'rmse', 'mae', 'r2',
        '--tracking-metric', 'rmse',
        '--pytorch-seed', str(args.seed),
        '--data-seed', str(args.seed),
    ]
    if args.arm == 'observable_provenance':
        registry = json.loads((ROOT / 'data_registry/gate2g1c_chemprop_formal_input_registry.json').read_text())
        job = next(j for j in registry['prepared_jobs'] if j['arm'] == args.arm and j['seed'] == args.seed and j['fold'] == args.fold)
        cmd += ['--descriptors-columns', *job['descriptor_columns']]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    (out_dir / 'launch_command.json').write_text(json.dumps({'cmd': cmd, 'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES']}, indent=2) + '\n')
    subprocess.check_call(cmd, cwd=ROOT, env=env)


if __name__ == '__main__':
    main()
