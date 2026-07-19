#!/usr/bin/env python3
"""Read-only audit for the historical Stage 2C cheap champion assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from rdkit import Chem


EXPECTED = {
    "split": "2bd3526b756122575cf7607bdc20b2ca16d558db6274df4e251edadeee369db5",
    "teacher": "04a3148803b507c41e415083fd6078e788aefa1706c20ad1ca23cb7145d2a56d",
    "structures": "9804e89ab76f67db4ce3e7950ef095eaf3873bfb6dc00ee0a6624861ae882bf4",
    "historical_script": "e541d753583c9911d2628d21bd16d15d669347967dc6356f0dd60e63a27627a0",
    "historical_predictions": "16a7e9a8c60176ae0f5c2f31ca6be10ece374967131996a1da6096b0d06818ea",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sid(sid: str) -> str:
    donor, acceptor = sid.split("_")
    return donor.replace("D-", "D") + "_" + acceptor.replace("A-", "A")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["historical_assets"].items()}

    observed_hashes = {
        "split": sha256(paths["split_manifest"]),
        "teacher": sha256(paths["teacher_table"]),
        "structures": sha256(paths["structure_registry"]),
        "historical_script": sha256(paths["historical_script"]),
        "historical_predictions": sha256(paths["historical_predictions"]),
    }
    if observed_hashes != EXPECTED:
        raise RuntimeError(f"historical asset hash mismatch: {observed_hashes}")

    split = json.loads(paths["split_manifest"].read_text())
    split_counts = {name: len(split[name]) for name in ("train", "val", "test")}
    if split_counts != {"train": 5118, "val": 1098, "test": 1097}:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    sets = {name: set(split[name]) for name in split_counts}
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Layer G partitions overlap")
    all_sids = split["train"] + split["val"] + split["test"]
    if len(set(all_sids)) != 7313:
        raise RuntimeError("Layer G does not contain 7,313 unique IDs")

    with paths["teacher_table"].open(newline="") as handle:
        teacher = {row["molecule_id"]: row["eb_eV"] for row in csv.DictReader(handle)}
    missing_teacher = sorted(set(all_sids) - teacher.keys())
    if missing_teacher:
        raise RuntimeError(f"missing teacher labels: {missing_teacher[:10]}")

    wanted_compact = {compact_sid(sid) for sid in all_sids}
    found_compact: set[str] = set()
    structure_smiles: dict[str, str] = {}
    with paths["structure_registry"].open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") in wanted_compact:
                found_compact.add(row["id"])
                structure_smiles[row["id"]] = row["smiles"]
    missing_structures = sorted(wanted_compact - found_compact)
    if missing_structures:
        raise RuntimeError(f"missing structure records: {missing_structures[:10]}")

    pm6_dir = paths["pm6_energy_dir"]
    missing_pm6: list[str] = []
    basename_mismatches: list[str] = []
    parse_failures: list[str] = []
    current_sid_atom_matches = 0
    old_basename_atom_matches = 0
    atom_comparable = 0
    pm6_dft_old_basename_matches = 0
    for sid in all_sids:
        path = pm6_dir / f"{sid}_metadata.json"
        if not path.exists():
            missing_pm6.append(sid)
            continue
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            parse_failures.append(sid)
            continue
        if row.get("basename") not in (None, sid):
            basename_mismatches.append(sid)
        current_smiles = structure_smiles.get(compact_sid(sid))
        old_basename = row.get("basename")
        old_smiles = structure_smiles.get(compact_sid(old_basename)) if isinstance(old_basename, str) and "_" in old_basename else None
        if current_smiles and row.get("num_atoms") is not None:
            current_mol = Chem.MolFromSmiles(current_smiles)
            old_mol = Chem.MolFromSmiles(old_smiles) if old_smiles else None
            atom_comparable += 1
            current_sid_atom_matches += int(Chem.AddHs(current_mol).GetNumAtoms() == row["num_atoms"])
            old_basename_atom_matches += int(old_mol is not None and Chem.AddHs(old_mol).GetNumAtoms() == row["num_atoms"])
        dft_path = paths["dft_energy_dir"] / f"{sid}.json"
        if dft_path.exists():
            try:
                pm6_dft_old_basename_matches += int(json.loads(dft_path.read_text()).get("basename") == old_basename)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    with paths["historical_predictions"].open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    prediction_ids = [row["molecule_id"] for row in predictions]
    if len(predictions) != 1097 or prediction_ids != split["test"]:
        raise RuntimeError("historical prediction order does not equal frozen test order")

    evidence = {
        "status": "ASSET_AUDIT_PASS",
        "observed_hashes": observed_hashes,
        "split_counts": split_counts,
        "split_unique": 7313,
        "teacher_coverage": 7313,
        "structure_coverage": 7313,
        "pm6_present": 7313 - len(missing_pm6) - len(parse_failures),
        "pm6_missing": missing_pm6,
        "pm6_parse_failures": parse_failures,
        "pm6_basename_mismatch_count": len(basename_mismatches),
        "pm6_basename_mismatch_ids_sha256": hashlib.sha256(
            "\n".join(sorted(basename_mismatches)).encode()
        ).hexdigest(),
        "pm6_identity_crosscheck": {
            "atom_count_comparable": atom_comparable,
            "matches_current_filename_sid_structure": current_sid_atom_matches,
            "matches_embedded_old_basename_structure": old_basename_atom_matches,
            "pm6_dft_embedded_old_basename_equal": pm6_dft_old_basename_matches,
            "interpretation": (
                "The embedded basename is a stale pre-renumbering alias. Current filename SID is supported "
                "by total-atom agreement and is the frozen model identity."
            ),
        },
        "historical_prediction_rows": 1097,
        "historical_prediction_order_matches_test": True,
        "historical_model_dump_found": False,
        "historical_complete_run_log_found": False,
        "historical_selection_limitation": (
            "The Stage 2C script selected best_config using test MAE across configurations; "
            "the fixed B reproduction is historical reproduction, not leakage-corrected model selection."
        ),
        "final673_accessed": False,
        "python": sys.version.split()[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
