#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/changliu/ExcitationNexus_Data_v2')
NEXUS = Path('/home/changliu/ExcitationNexus')
RUN = ROOT / 'runs/gate2g1c_chemprop_v2_formal'
RECOVERY_COMMIT = '124dd51a2731af3ab0ef2eee3c2ea3af02857b9e'
PROV = ['method', 'basis', 'program', 'geometry_fidelity', 'target_semantics_version']
TARGET = 'y'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(mid: str) -> str:
    return str(mid).replace('D-', 'D').replace('_A-', '_A')


def slug(value: object) -> str:
    text = str(value)
    text = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_').lower()
    return text or 'empty'


def load_smiles_targets() -> tuple[dict[str, str], dict[str, float], int]:
    smiles: dict[str, str] = {}
    target: dict[str, float] = {}
    new = pd.read_parquet(
        DATA / 'tables/molecule_values_v3.parquet',
        columns=['molecule_id', 'canonical_smiles', 'tddft_coulomb_attraction_eV_eps3p5_proxy'],
    )
    for r in new.itertuples(index=False):
        mid = norm(r.molecule_id)
        smiles[mid] = r.canonical_smiles
        target[f'new15016:{mid}'] = float(r.tddft_coulomb_attraction_eV_eps3p5_proxy)
    old = pd.read_csv(
        NEXUS / 'DA_data/unified_dataset_7316.csv',
        usecols=['molecule_id', 'coulomb_attraction_screened_eV'],
    )
    for r in old.itertuples(index=False):
        target[f'old7316:{norm(r.molecule_id)}'] = float(r.coulomb_attraction_screened_eV)
    final_label_reads = 0
    with (NEXUS / 'equiformer_v3_model/data/external_3371/3371_total_manifest.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            mid = norm(row['sample_id'])
            if row['split_role'] == 'external-dev':
                target[f'external2698:{mid}'] = float(row['label_eV'])
            elif row['split_role'] == 'final-blind':
                final_label_reads += 0
    with (NEXUS / 'DA_data/structure_60k_with_rg_sorted.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            mid = norm(row['Original_ID'])
            smiles.setdefault(mid, row['SMILES'])
    return smiles, target, final_label_reads


def write_json(rel: str, obj: object) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def add_provenance_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    out = frame.copy()
    categories: dict[str, list[str]] = {}
    descriptor_cols: list[str] = []
    for field in PROV:
        vals = sorted(out[field].astype(str).unique())
        categories[field] = vals
        for val in vals:
            col = f'prov__{field}__{slug(val)}'
            out[col] = (out[field].astype(str) == val).astype(int)
            descriptor_cols.append(col)
    return out, descriptor_cols, categories


def build_unified_fold_csv(frame: pd.DataFrame, assign: pd.DataFrame, arm: str, seed: int, fold: int, descriptor_cols: list[str]) -> dict[str, object]:
    if arm not in {'blind', 'observable_provenance'}:
        raise ValueError(arm)
    active = assign[assign.partition == 'outer_validation'].copy()
    test_ids = set(active.loc[active.outer_fold == fold, 'global_record_id'])
    train_pool = active[active.outer_fold != fold].copy()
    if 'inner_fold' not in train_pool.columns:
        raise SystemExit('unified manifest missing frozen inner_fold')
    val_ids = set(train_pool.loc[train_pool.inner_fold == fold, 'global_record_id'])
    train_ids = set(train_pool.loc[train_pool.inner_fold != fold, 'global_record_id'])
    if test_ids & train_ids or test_ids & val_ids or train_ids & val_ids:
        raise SystemExit('split overlap detected')
    subset = frame[frame.global_record_id.isin(test_ids | train_ids | val_ids)].copy()
    subset['split'] = 'unset'
    subset.loc[subset.global_record_id.isin(train_ids), 'split'] = 'train'
    subset.loc[subset.global_record_id.isin(val_ids), 'split'] = 'val'
    subset.loc[subset.global_record_id.isin(test_ids), 'split'] = 'test'
    subset = subset.sort_values(['split', 'global_record_id'], kind='mergesort')
    base_cols = ['smiles', TARGET, 'weight', 'split']
    if arm == 'observable_provenance':
        train_cols = base_cols + descriptor_cols
    else:
        train_cols = base_cols
    out_dir = RUN / 'data' / 'unified_grouped_5fold_oof' / arm / f'seed{seed}' / f'fold{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / 'chemprop_train.csv'
    manifest_csv = out_dir / 'local_manifest.csv'
    subset.rename(columns={'group_weight': 'weight'})[train_cols].to_csv(train_csv, index=False)
    subset[['global_record_id', 'canonical_structure_group_id', 'source_cohort', 'split', 'group_weight']].to_csv(manifest_csv, index=False)
    counts = subset.split.value_counts().to_dict()
    return {
        'arm': arm,
        'seed': seed,
        'fold': fold,
        'train_csv': str(train_csv.relative_to(ROOT)),
        'local_manifest_csv': str(manifest_csv.relative_to(ROOT)),
        'records': int(len(subset)),
        'train_records': int(counts.get('train', 0)),
        'val_records': int(counts.get('val', 0)),
        'test_records': int(counts.get('test', 0)),
        'train_csv_sha256': sha(train_csv),
        'manifest_sha256': sha(manifest_csv),
        'descriptor_columns': descriptor_cols if arm == 'observable_provenance' else [],
    }


def main() -> None:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', RECOVERY_COMMIT, 'HEAD'], cwd=ROOT)
    for registry in ['gate2g1a_sha256.txt', 'gate2g1b_sha256.txt', 'gate2g1c_sha256.txt', 'gate2g1c_chemprop_dependency_recovery_sha256.txt']:
        subprocess.check_call(['sha256sum', '-c', f'data_registry/{registry}'], cwd=ROOT, stdout=subprocess.DEVNULL)
    RUN.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_parquet(ROOT / 'runs/gate2g1b_source_aware_splits/eligible_development_after_final673_quarantine.parquet')
    quarantine = pd.read_parquet(ROOT / 'runs/gate2g1b_source_aware_splits/final673_structure_overlap_quarantine.parquet')
    if len(eligible) != 25008 or len(quarantine) != 22:
        raise SystemExit('G1B eligible/quarantine mismatch')
    smiles, targets, final_reads = load_smiles_targets()
    if final_reads != 0:
        raise SystemExit('final673 label reads detected')
    frame = eligible.copy()
    frame['smiles'] = frame.normalized_molecule_id.map(smiles)
    frame[TARGET] = frame.global_record_id.map(targets)
    if frame.smiles.isna().any() or frame[TARGET].isna().any():
        raise SystemExit('SMILES/target join failure')
    if set(frame.global_record_id) & set(quarantine.global_record_id):
        raise SystemExit('quarantine entered Chemprop frame')
    frame, descriptor_cols, categories = add_provenance_columns(frame)
    master = RUN / 'data/eligible_chemprop_master.csv'
    master.parent.mkdir(parents=True, exist_ok=True)
    frame[['global_record_id', 'normalized_molecule_id', 'smiles', TARGET, 'source_cohort', 'canonical_structure_group_id', 'group_weight'] + PROV].to_csv(master, index=False)
    assign = pd.read_parquet(ROOT / 'runs/gate2g1b_source_aware_splits/unified_grouped_5fold_oof.parquet')
    jobs = []
    for arm in ['blind', 'observable_provenance']:
        for seed in [42, 123, 456]:
            for fold in range(5):
                jobs.append(build_unified_fold_csv(frame, assign, arm, seed, fold, descriptor_cols))
    # Actual-data admission smoke uses only outer-train records from the first formal job.
    smoke_src = pd.read_csv(ROOT / jobs[0]['train_csv'])
    smoke = pd.concat([
        smoke_src[smoke_src.split == 'train'].head(8),
        smoke_src[smoke_src.split == 'val'].head(2),
        smoke_src[smoke_src.split == 'train'].tail(2).assign(split='test'),
    ], ignore_index=True)
    smoke_path = RUN / 'data/admission_smoke_actual_outer_train_only.csv'
    smoke.to_csv(smoke_path, index=False)
    contract = {
        'status': 'GATE2G1C_CHEMPROP_FORMAL_INPUTS_PREPARED',
        'head_at_prepare': head,
        'base_dependency_recovery_commit': RECOVERY_COMMIT,
        'eligible_records': int(len(frame)),
        'eligible_structures': int(frame.canonical_structure_group_id.nunique()),
        'final673_label_reads': final_reads,
        'quarantine_records_in_frame': int(len(set(frame.global_record_id) & set(quarantine.global_record_id))),
        'rdkit_version_g1b_ml_env': json.loads((ROOT / 'data_registry/gate2g1b_aggregate_counts.json').read_text())['rdkit_version'],
        'rdkit_version_chemprop_env_expected': '2026.03.4',
        'provenance_fields': PROV,
        'provenance_categories': categories,
        'raw_source_cohort_token_used': False,
        'prepared_jobs': jobs,
        'master_csv': str(master.relative_to(ROOT)),
        'master_csv_sha256': sha(master),
        'actual_data_smoke_csv': str(smoke_path.relative_to(ROOT)),
        'actual_data_smoke_csv_sha256': sha(smoke_path),
        'formal_training_started': False,
    }
    write_json('data_registry/gate2g1c_chemprop_formal_input_registry.json', contract)
    report = [
        '# Gate 2-G1C Chemprop formal recovery preregistration',
        '',
        'Status: **GATE2G1C_CHEMPROP_FORMAL_INPUTS_PREPARED**.',
        '',
        'Chemprop v2 dependency recovery is complete, but the formal D-MPNN benchmark is not yet complete. This amendment prepares frozen Chemprop inputs for the unified grouped OOF protocol and launches only after graph compatibility and actual-data smoke pass.',
        '',
        f'Eligible records: {len(frame)}; eligible structures: {frame.canonical_structure_group_id.nunique()}.',
        '',
        'The initial formal launch is `unified_grouped_5fold_oof / blind / seed42 / fold0`. OOD Chemprop protocols are pending an explicit inner-validation policy because Gate 2-G1B only materialized `inner_fold` for the unified grouped OOF manifest.',
        '',
        'Observable provenance uses one-hot `method`, `basis`, `program`, `geometry_fidelity`, and `target_semantics_version` descriptors. Raw `source_cohort`, molecule IDs, file paths, final673 membership, and historical split labels are forbidden model inputs.',
    ]
    (ROOT / 'reports/gate2g1c_chemprop_formal_recovery_preregistration.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'status': contract['status'], 'jobs': len(jobs), 'master_csv': contract['master_csv'], 'smoke_csv': contract['actual_data_smoke_csv']}, indent=2))


if __name__ == '__main__':
    main()
