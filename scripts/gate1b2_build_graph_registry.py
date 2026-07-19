#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from excitationnexus_phase12.dft_graph_dataset import graph_content_hash
from excitationnexus_phase12.role_resolution import load_structure_fields

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
RAW = Path("/home/changliu/ExcitationNexus_Data_v2/raw_compact/dft/results")
MANIFEST = ROOT / "manifests/split_iid_group_seed42_v1.csv"
ROLE_MANIFEST = ROOT / "manifests/role_resolution_v1.csv"
CACHE = ROOT / "runs/gate1b2_3d_admission/dft_graph_cache_v1.npz"
REGISTRY = ROOT / "data_registry/dft_3d_graph_registry_v1.parquet"

Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
     "S": 16, "Cl": 17, "Se": 34, "Br": 35, "I": 53}
ROLE = {"donor": 0, "acceptor": 1, "unknown": 2}
BOND = {"single": 1, "double": 2, "triple": 3, "aromatic": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdb_atoms(path: Path) -> tuple[list[str], np.ndarray]:
    elements, coords = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            elements.append((line[76:78].strip() or line[12:16].strip()[0]).capitalize())
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return elements, np.asarray(coords, dtype=np.float64)


def main() -> None:
    started = time.perf_counter()
    manifest = pd.read_csv(MANIFEST).sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    roles = pd.read_csv(ROLE_MANIFEST, usecols=["molecule_id", "dft_json_sha256", "dft_pdb_sha256"])
    frame = manifest.merge(roles, on="molecule_id", validate="one_to_one")
    all_z, all_pos, all_role, all_bonds, all_bond_types, rows = [], [], [], [], [], []
    atom_offset = bond_offset = 0
    for cache_index, row in enumerate(frame.itertuples(index=False)):
        mid = str(row.molecule_id); folder = RAW / mid
        json_path, pdb_path = folder / f"{mid}_dft.json", folder / f"{mid}_dft.pdb"
        fields = load_structure_fields(json_path)
        z = np.asarray([Z[atom["element"]] for atom in fields["atoms"]], dtype=np.uint8)
        pos64 = np.asarray([atom["coords"] for atom in fields["atoms"]], dtype=np.float64)
        role = np.asarray([ROLE[atom["role"]] for atom in fields["atoms"]], dtype=np.uint8)
        bonds = np.asarray([[bond["a"] - 1, bond["b"] - 1] for bond in fields["bonds"]], dtype=np.int32).reshape(-1, 2)
        bond_types = np.asarray([BOND.get(bond["type"], 1) for bond in fields["bonds"]], dtype=np.uint8)
        if not np.isfinite(pos64).all() or len(z) == 0 or np.any(bonds < 0) or np.any(bonds >= len(z)):
            raise RuntimeError(f"invalid graph arrays for {mid}")
        pdb_elements, pdb_pos = pdb_atoms(pdb_path)
        json_elements = [atom["element"] for atom in fields["atoms"]]
        if pdb_elements != json_elements or pdb_pos.shape != pos64.shape:
            raise RuntimeError(f"PDB/JSON atom identity mismatch for {mid}")
        pdb_delta = float(np.max(np.abs(pdb_pos - pos64)))
        if pdb_delta > 0.000501:
            raise RuntimeError(f"PDB/JSON rounding mismatch for {mid}: {pdb_delta}")
        graph_hash = graph_content_hash(z, pos64, role, bonds, bond_types)
        counts = Counter(role.tolist())
        rows.append({
            "molecule_id": mid, "partition": row.partition, "cache_index": cache_index,
            "atom_offset_start": atom_offset, "atom_offset_end": atom_offset + len(z),
            "bond_offset_start": bond_offset, "bond_offset_end": bond_offset + len(bonds),
            "num_atoms": len(z), "num_bonds": len(bonds), "donor_atoms": counts[0],
            "acceptor_atoms": counts[1], "unknown_atoms": counts[2],
            "coordinates_finite": True, "pdb_json_max_abs_delta_angstrom": pdb_delta,
            "graph_content_sha256": graph_hash, "dft_json_sha256": row.dft_json_sha256,
            "dft_pdb_sha256": row.dft_pdb_sha256, "schema_version": "dft_graph_v1",
        })
        all_z.append(z); all_pos.append(pos64.astype(np.float32)); all_role.append(role)
        all_bonds.append(bonds); all_bond_types.append(bond_types)
        atom_offset += len(z); bond_offset += len(bonds)
    registry = pd.DataFrame(rows)
    if len(registry) != 15016 or registry.molecule_id.nunique() != 15016:
        raise RuntimeError("graph registry identity failure")
    if registry.partition.value_counts().to_dict() != {"train": 10387, "test": 2319, "val": 2309, "historical_quarantine": 1}:
        raise RuntimeError("graph registry partition mismatch")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, z=np.concatenate(all_z), pos=np.concatenate(all_pos),
                        role=np.concatenate(all_role), bonds=np.concatenate(all_bonds),
                        bond_types=np.concatenate(all_bond_types))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True); registry.to_parquet(REGISTRY, index=False)
    shuffled = registry.sample(frac=1, random_state=20260719).sort_values("molecule_id", kind="mergesort")
    stable = registry.sort_values("molecule_id", kind="mergesort")
    if stable[["molecule_id", "graph_content_sha256"]].reset_index(drop=True).equals(
            shuffled[["molecule_id", "graph_content_sha256"]].reset_index(drop=True)) is False:
        raise RuntimeError("row-order graph hash invariance failed")
    spec = {
        "version": "v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_fields": ["element/atomic_number", "DFT S0 coordinates", "explicit original role", "bond topology"],
        "forbidden_fields": ["DFT scalar properties", "PM6 scalar properties", "TDDFT/Multiwfn", "target", "IDs as model inputs", "partition embedding"],
        "cache_path": str(CACHE), "cache_sha256": sha256(CACHE), "cache_uploaded_to_github": False,
        "registry_path": str(REGISTRY), "registry_sha256": sha256(REGISTRY),
        "records": len(registry), "atoms": int(atom_offset), "bonds": int(bond_offset),
        "partition_counts": registry.partition.value_counts().to_dict(),
        "pdb_json_max_delta_angstrom": float(registry.pdb_json_max_abs_delta_angstrom.max()),
        "manifest_sha256": sha256(MANIFEST), "role_manifest_sha256": sha256(ROLE_MANIFEST),
        "target_fields_read": False, "final673_accessed": False,
    }
    spec_path = ROOT / "data_registry/dft_3d_graph_spec_v1.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    report = ROOT / "reports/gate1b2_graph_integrity.md"
    report.write_text(
        "# Gate 1-B2 DFT graph integrity\n\n"
        f"Target-free registry covers {len(registry):,} records, {atom_offset:,} atoms, and {bond_offset:,} source bonds. "
        f"Partition counts are `{json.dumps(spec['partition_counts'], sort_keys=True)}`. Maximum PDB/JSON coordinate rounding delta is {spec['pdb_json_max_delta_angstrom']:.7f} Å.\n\n"
        f"Local cache SHA-256: `{spec['cache_sha256']}`; registry SHA-256: `{spec['registry_sha256']}`. The large cache is excluded from GitHub. Quarantine is registered but cannot form a Dataset. No scalar or target field was read.\n"
    )
    print(json.dumps({"status": "GRAPH_REGISTRY_COMPLETE", "records": len(registry),
                      "atoms": atom_offset, "bonds": bond_offset, "cache_sha256": spec["cache_sha256"],
                      "wall_seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
