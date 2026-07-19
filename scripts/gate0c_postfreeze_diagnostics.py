#!/usr/bin/env python3
"""Read-only post-freeze split diagnostics; never changes assignments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet")
FILES = {
    "iid_group_seed42_v1": "split_iid_group_seed42_v1.csv",
    "donor_cold_v1": "split_donor_cold_v1.csv",
    "acceptor_cold_v1": "split_acceptor_cold_v1.csv",
    "pair_cold_v1": "split_pair_cold_v1.csv",
    "both_cold_external_test_v1": "split_both_cold_external_test_v1.csv",
    "full_scaffold_cold_v1": "split_full_scaffold_cold_v1.csv",
}
LOW_COST = ["num_atoms_total", "pm6_gap_ev"]
TARGETS = [
    "tddft_coulomb_attraction_eV_eps3p5_proxy", "tddft_excitation_energy_ev",
    "tddft_wavelength_nm", "tddft_oscillator_strength", "tddft_transition_dipole_au",
    "tddft_coulomb_attraction_eV", "tddft_Sm", "tddft_Sr", "tddft_D_index_angstrom",
    "tddft_H_CT_angstrom", "tddft_t_index_angstrom", "tddft_HDI", "tddft_EDI",
    "tddft_Q_D_to_A_au", "tddft_dipole_change_norm_au",
    "tddft_hole_on_donor_fraction", "tddft_hole_on_acceptor_fraction",
    "tddft_electron_on_donor_fraction", "tddft_electron_on_acceptor_fraction",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if not len(x):
        return {"n": 0}
    q = x.quantile([0, .05, .25, .5, .75, .95, 1])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(q.loc[0]), "p05": float(q.loc[.05]), "p25": float(q.loc[.25]),
            "median": float(q.loc[.5]), "p75": float(q.loc[.75]),
            "p95": float(q.loc[.95]), "max": float(q.loc[1])}


def diagnostic_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return mol if Chem.SanitizeMol(mol, sanitizeOps=ops, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE else None


def similarity_quantiles(train_smiles, query_smiles, fpgen):
    train_unique = sorted(set(train_smiles.dropna()))
    query_unique = sorted(set(query_smiles.dropna()))
    train_fps = []
    failures = 0
    for s in train_unique:
        m = diagnostic_mol(s)
        if m is None: failures += 1
        else: train_fps.append(fpgen.GetFingerprint(m))
    nearest = []
    for s in query_unique:
        m = diagnostic_mol(s)
        if m is None:
            failures += 1; continue
        fp = fpgen.GetFingerprint(m)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        nearest.append(max(sims) if sims else np.nan)
    x = pd.Series(nearest).dropna()
    return {"n_unique_query": len(query_unique), "n_unique_train": len(train_unique),
            "parse_failures": failures, "nearest_train_tanimoto": stats(x)}


def main():
    RDLogger.DisableLog("rdApp.*")
    frozen_path = ROOT / "data_registry/SPLIT_REGISTRY_V1_FROZEN.json"
    assert frozen_path.exists(), "split registry must be frozen before diagnostics"
    frozen = json.loads(frozen_path.read_text())
    for name, info in frozen["splits"].items():
        p = ROOT / info["manifest"]
        assert digest(p) == info["verified_sha256"]
        assert info["status"] == "DONE"
    values = pd.read_parquet(DATA, columns=["molecule_id"] + LOW_COST + TARGETS)
    structure = pd.read_parquet(ROOT / "manifests/new15016_structure_groups_v1.parquet",
                                columns=["molecule_id", "canonical_structure_smiles_v1"])
    component = pd.read_csv(ROOT / "manifests/component_identity_v1.csv", usecols=[
        "molecule_id", "donor_canonical_structure_smiles_v1",
        "acceptor_canonical_structure_smiles_v1"])
    base = values.merge(structure, on="molecule_id", validate="one_to_one").merge(
        component, on="molecule_id", validate="one_to_one")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    output = {
        "status": "DONE", "postfreeze_only": True, "split_assignments_modified": False,
        "morgan": {"algorithm": "RDKit Morgan", "radius": 2, "nBits": 2048,
                   "useChirality": True, "rdkit_version": rdBase.rdkitVersion},
        "splits": {},
    }
    balance_rows = []
    for name, filename in FILES.items():
        split = pd.read_csv(ROOT / "manifests" / filename)
        d = split.merge(base, on="molecule_id", how="left", validate="one_to_one")
        entry = {"partitions": {}, "exact_overlap": {}, "ood_similarity": {}}
        for p, g in d.groupby("partition", sort=True):
            part = {"records": len(g), "structure_groups": g.structure_group_id_v1.nunique(),
                    "effective_weight": float(g.group_weight.sum()),
                    "donors": g.donor_structure_group_id_v1.nunique(),
                    "acceptors": g.acceptor_structure_group_id_v1.nunique(),
                    "pairs": g.pair_group_id_v1.nunique(),
                    "scaffolds": g.full_scaffold_group_id_v1.nunique(),
                    "low_cost": {c: stats(g[c]) for c in LOW_COST},
                    "targets_postfreeze": {c: stats(g[c]) for c in TARGETS}}
            entry["partitions"][p] = part
            balance_rows.append({"split_name": name, "partition": p, **{k: part[k] for k in
                                 ("records", "structure_groups", "effective_weight", "donors",
                                  "acceptors", "pairs", "scaffolds")}})
        train = d[d.partition.eq("train")]
        for p in ("val", "test", "buffer"):
            query = d[d.partition.eq(p)]
            if query.empty: continue
            entry["exact_overlap"][p] = {
                "donor_with_train": len(set(query.donor_structure_group_id_v1) & set(train.donor_structure_group_id_v1)),
                "acceptor_with_train": len(set(query.acceptor_structure_group_id_v1) & set(train.acceptor_structure_group_id_v1)),
                "pair_with_train": len(set(query.pair_group_id_v1) & set(train.pair_group_id_v1)),
                "scaffold_with_train": len(set(query.full_scaffold_group_id_v1) & set(train.full_scaffold_group_id_v1)),
                "structure_with_train": len(set(query.structure_group_id_v1) & set(train.structure_group_id_v1)),
            }
            entry["ood_similarity"][p] = {
                "full_molecule": similarity_quantiles(train.canonical_structure_smiles_v1,
                                                       query.canonical_structure_smiles_v1, fpgen),
                "donor_component": similarity_quantiles(train.donor_canonical_structure_smiles_v1,
                                                         query.donor_canonical_structure_smiles_v1, fpgen),
                "acceptor_component": similarity_quantiles(train.acceptor_canonical_structure_smiles_v1,
                                                            query.acceptor_canonical_structure_smiles_v1, fpgen),
            }
        usable = d.partition.isin(["train", "val", "test"])
        entry["retained_fraction_train_val_test"] = float(usable.mean())
        entry["buffer_fraction"] = float(d.partition.eq("buffer").mean())
        output["splits"][name] = entry
    out_path = ROOT / "logs/gate0c_postfreeze_diagnostics.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(balance_rows).to_csv(ROOT / "manifests/split_postfreeze_counts_v1.csv", index=False)

    lines = ["# Gate 0-C split balance (post-freeze)", "",
             "Diagnostics were computed only after manifest hash freeze; assignments were not modified.", "",
             pd.DataFrame(balance_rows).to_markdown(index=False), "",
             "Target summaries are retained in `logs/gate0c_postfreeze_diagnostics.json` and did not influence v1."]
    (ROOT / "reports/gate0c_split_balance_postfreeze.md").write_text("\n".join(lines) + "\n")
    ood = ["# Gate 0-C OOD severity", "",
           "Morgan: radius=2, nBits=2048, useChirality=True. Similarity is diagnostic only.", ""]
    for name, e in output["splits"].items():
        ood += [f"## {name}", ""]
        for p, kinds in e["ood_similarity"].items():
            ood.append(f"- {p}: " + "; ".join(
                f"{k} median={v['nearest_train_tanimoto'].get('median', 'NA'):.4f}, p05={v['nearest_train_tanimoto'].get('p05', 'NA'):.4f}"
                for k, v in kinds.items()))
        ood.append("")
    (ROOT / "reports/gate0c_ood_severity.md").write_text("\n".join(ood) + "\n")
    print(json.dumps({"status": "DONE", "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
