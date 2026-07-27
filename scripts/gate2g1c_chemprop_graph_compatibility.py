#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import chemprop
import numpy as np
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'runs/gate2g1c_chemprop_v2_formal/data/eligible_chemprop_master.csv'


def arr_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main() -> None:
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    rows = 0
    failures = []
    atom_counts = []
    bond_counts = []
    graph_hashes = []
    with MASTER.open(newline='') as f:
        for row in csv.DictReader(f):
            rows += 1
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failures.append({'row': rows, 'reason': 'mol_parse_failed'})
                continue
            graph = featurizer(mol)
            arrays = [graph.V, graph.E, graph.edge_index, graph.rev_edge_index]
            if any(not np.isfinite(a).all() for a in arrays if np.issubdtype(a.dtype, np.number)):
                failures.append({'row': rows, 'reason': 'non_finite_graph'})
                continue
            if graph.V.shape[0] != mol.GetNumAtoms():
                failures.append({'row': rows, 'reason': 'atom_count_mismatch'})
                continue
            atom_counts.append(mol.GetNumAtoms())
            bond_counts.append(mol.GetNumBonds())
            if rows <= 64:
                graph2 = featurizer(mol)
                h1 = arr_hash(graph.V, graph.E, graph.edge_index, graph.rev_edge_index)
                h2 = arr_hash(graph2.V, graph2.E, graph2.edge_index, graph2.rev_edge_index)
                if h1 != h2:
                    failures.append({'row': rows, 'reason': 'graph_nondeterministic'})
                    continue
                graph_hashes.append(h1)
    status = 'CHEMPROP_GRAPH_COMPATIBILITY_PASS' if not failures and rows == 25008 else 'BLOCKED_CHEMPROP_GRAPH_COMPATIBILITY'
    reg = {
        'status': status,
        'chemprop_version': chemprop.__version__,
        'rdkit_version_chemprop_env': rdBase.rdkitVersion,
        'rdkit_version_ml_env_gate2g1b': json.loads((ROOT / 'data_registry/gate2g1b_aggregate_counts.json').read_text())['rdkit_version'],
        'records_checked': rows,
        'parse_failures': len([x for x in failures if x['reason'] == 'mol_parse_failed']),
        'graph_failures': len(failures),
        'min_atoms': int(min(atom_counts)),
        'max_atoms': int(max(atom_counts)),
        'min_bonds': int(min(bond_counts)),
        'max_bonds': int(max(bond_counts)),
        'determinism_sample_count': len(graph_hashes),
        'determinism_hash_sha256': hashlib.sha256('\n'.join(graph_hashes).encode()).hexdigest(),
        'failure_examples_public': failures[:5],
        'final673_label_reads': 0,
        'candidate_assets_accessed': False,
    }
    out = ROOT / 'data_registry/gate2g1c_chemprop_graph_compatibility.json'
    out.write_text(json.dumps(reg, indent=2, sort_keys=True) + '\n')
    report = [
        '# Gate 2-G1C Chemprop graph compatibility',
        '',
        f'Status: **{status}**.',
        '',
        f'Checked {rows} eligible development records with Chemprop {chemprop.__version__} and RDKit {rdBase.rdkitVersion}. Gate 2-G1B was generated with RDKit {reg["rdkit_version_ml_env_gate2g1b"]}; this version difference is recorded and is not treated as an automatic blocker because full parsing and Chemprop graph construction passed.',
        '',
        f'Atom count range: {reg["min_atoms"]}-{reg["max_atoms"]}. Bond count range: {reg["min_bonds"]}-{reg["max_bonds"]}. Determinism sample count: {len(graph_hashes)}.',
        '',
        'No final673 labels or Gate 3 candidate assets were accessed.',
    ]
    (ROOT / 'reports/gate2g1c_chemprop_graph_compatibility.md').write_text('\n'.join(report) + '\n')
    if status != 'CHEMPROP_GRAPH_COMPATIBILITY_PASS':
        raise SystemExit(status)
    print(json.dumps(reg, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
