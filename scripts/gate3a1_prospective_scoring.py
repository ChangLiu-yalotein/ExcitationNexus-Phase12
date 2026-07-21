#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
from excitationnexus_phase12.prospective_scoring import (
    C0_COLUMNS, DESC_NAMES, assert_c0_contract, c0_matrix, deterministic_rank_percentile,
    fit_preprocessor, identity_cap_ok, load_config, load_deployment_labels,
    load_preprocessor, ordered_hash, save_preprocessor, sha256, transform, verify_inputs, write_json,
)


def resolve(path: str) -> Path:
    return ROOT / path


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_base(config: dict) -> None:
    if git_value("rev-parse", "HEAD") != config["base_git_head"]:
        raise RuntimeError("Gate 3-A1 must start from the frozen Gate 3-A0 HEAD")
    if git_value("rev-parse", "origin/main") != config["base_git_head"]:
        raise RuntimeError("origin/main mismatch")
    if git_value("rev-parse", f"{config['required_tag']}^{{}}") != config["base_git_head"]:
        raise RuntimeError("Gate 3-A0 tag mismatch")
    if git_value("status", "--porcelain"):
        expected = {
            "configs/gate3a1_prospective_scoring_v1.json",
            "scripts/gate3a1_prospective_scoring.py",
            "src/excitationnexus_phase12/prospective_scoring.py",
            "tests/test_gate3a1_contract.py",
        }
        current = {line[3:] for line in git_value("status", "--porcelain").splitlines()}
        if not current.issubset(expected):
            raise RuntimeError(f"unexpected worktree change before preregistration: {sorted(current - expected)}")


def preregister(config: dict) -> None:
    require_base(config)
    verify_inputs(ROOT, config)
    assert_c0_contract(json.loads(resolve(config["inputs"]["gate1b1_config"]["path"]).read_text())["features"]["M1_C0_open"])
    candidate = pd.read_parquet(resolve(config["inputs"]["candidate_universe"]["path"]), columns=["pair_hash", "status", "in_pair_cold_domain"])
    counts = candidate["status"].value_counts().to_dict()
    selected = candidate[candidate["status"].eq("NOVEL_PAIR_KNOWN_COMPONENTS") & candidate["in_pair_cold_domain"].astype(bool)]
    if len(candidate) != 54208 or len(selected) != 36523 or selected["pair_hash"].nunique() != 36523:
        raise RuntimeError("Gate 3-A0 candidate boundary mismatch")
    expected_counts = {
        "DUPLICATE_PRODUCT_FROM_DIFFERENT_ALIASES": 958, "INVALID_VALENCE": 308,
        "NOVEL_PAIR_KNOWN_COMPONENTS": 36523, "OBSERVED_EXACT_STRUCTURE": 14957,
        "OUTSIDE_COMPONENT_SUPPORT": 1462,
    }
    if counts != expected_counts:
        raise RuntimeError("Gate 3-A0 status counts mismatch")
    manifest = pd.read_csv(resolve(config["inputs"]["iid_manifest"]["path"]))
    labels, label_evidence = load_deployment_labels(ROOT, config, manifest)
    if len(labels) != 15015:
        raise RuntimeError("deployment training label coverage mismatch")

    config_path = ROOT / "configs/gate3a1_prospective_scoring_v1.json"
    locked_paths = [
        config_path,
        ROOT / "scripts/gate3a1_prospective_scoring.py",
        ROOT / "src/excitationnexus_phase12/prospective_scoring.py",
        ROOT / "tests/test_gate3a1_contract.py",
    ]
    files = {str(path.relative_to(ROOT)): sha256(path) for path in locked_paths}
    aggregate = hashlib.sha256("".join(f"{key}:{value}\n" for key, value in sorted(files.items())).encode()).hexdigest()
    lock = {
        "status": "FROZEN_BEFORE_FEATURE_BUILD_MODEL_FIT_AND_CANDIDATE_SCORING",
        "locked_utc": utc(), "files": files, "aggregate_sha256": aggregate,
        "gate3a0_status_counts": expected_counts, "eligible_candidate_count": 36523,
        "deployment_label_coverage": label_evidence,
        "bootstrap_seeds": config["stability"]["seeds"], "selection": config["selection"],
        "post_lock_policy": "Any scientific change requires explicit v2; v1 is never overwritten.",
        "test_artifact_access": False, "final673_access": False, "main_parquet_read": False,
    }
    write_json(ROOT / "data_registry/gate3a1_preregistration_lock_v1.json", lock)
    (ROOT / "reports/gate3a1_preregistration.md").write_text(
        "# Gate 3-A1 preregistration\n\n"
        "Status: **FROZEN BEFORE FEATURE BUILD, MODEL FIT, AND CANDIDATE SCORING**.\n\n"
        "The sole scorer is the Gate 1-B1 XGBoost-C0 contract: 20 RDKit descriptors plus a 512-bit "
        "radius-2 non-chiral full-molecule Morgan fingerprint. One deterministic deployment model and "
        "20 fixed structure-group bootstrap models are frozen before a single candidate-scoring call. "
        "Bootstrap spread is ranking stability only, not calibrated uncertainty. The 4+4+4+4 selection, "
        "0.80 extreme inclusion threshold, identity caps, diversity rule, and hash tie-breaks are locked.\n\n"
        "The 36,523 objects are seen-component/unseen-pair computations. They are not donor-OOD, "
        "acceptor-OOD, new-scaffold discovery, catalytic activity, or experimental evidence. "
        "`PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY` and "
        "`BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH` remain mandatory.\n"
    )
    write_json(ROOT / "data_registry/gate3a1_label_coverage_registry.json", {
        "status": "FROZEN_LABEL_COVERAGE_COMPLETE", **label_evidence,
        "source_artifacts": [{"path": x["path"], "sha256": x["sha256"]} for x in config["label_sources"]],
        "local_artifact": config["local_outputs"]["training_labels"],
    })
    resolve(config["local_outputs"]["root"]).mkdir(parents=True, exist_ok=True)
    labels.sort_values("molecule_id").to_parquet(resolve(config["local_outputs"]["training_labels"]), index=False)
    registry = json.loads((ROOT / "data_registry/gate3a1_label_coverage_registry.json").read_text())
    registry["local_artifact_sha256"] = sha256(resolve(config["local_outputs"]["training_labels"]))
    write_json(ROOT / "data_registry/gate3a1_label_coverage_registry.json", registry)
    print(json.dumps({"status": lock["status"], "eligible_candidates": 36523, "training_labels": 15015}, indent=2))


def nearest_similarity(candidate_bits: np.ndarray, observed_bits: np.ndarray) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, includeChirality=False)
    del generator
    observed = []
    for row in observed_bits:
        fp = DataStructs.ExplicitBitVect(512)
        for bit in np.flatnonzero(row):
            fp.SetBit(int(bit))
        observed.append(fp)
    result = np.empty(len(candidate_bits), dtype=np.float32)
    for i, row in enumerate(candidate_bits):
        fp = DataStructs.ExplicitBitVect(512)
        for bit in np.flatnonzero(row):
            fp.SetBit(int(bit))
        result[i] = max(DataStructs.BulkTanimotoSimilarity(fp, observed))
    return result


def build_features(config: dict) -> None:
    lock = json.loads((ROOT / "data_registry/gate3a1_preregistration_lock_v1.json").read_text())
    if sha256(ROOT / "configs/gate3a1_prospective_scoring_v1.json") != lock["files"]["configs/gate3a1_prospective_scoring_v1.json"]:
        raise RuntimeError("preregistration config changed")
    verify_inputs(ROOT, config)
    universe = pd.read_parquet(resolve(config["inputs"]["candidate_universe"]["path"]))
    candidate = universe[universe["status"].eq("NOVEL_PAIR_KNOWN_COMPONENTS") & universe["in_pair_cold_domain"].astype(bool)].copy()
    candidate = candidate.sort_values("pair_hash", kind="mergesort").reset_index(drop=True)
    observed = universe[universe["status"].eq("OBSERVED_EXACT_STRUCTURE")].copy()
    observed = observed.sort_values("pair_hash", kind="mergesort").drop_duplicates("full_structure_hash").reset_index(drop=True)
    if len(candidate) != 36523 or candidate.pair_hash.nunique() != 36523:
        raise RuntimeError("candidate filter failure")
    x_candidate = c0_matrix(candidate.canonical_smiles.astype(str).tolist())
    x_observed = c0_matrix(observed.canonical_smiles.astype(str).tolist())
    observed_train = pd.read_parquet(resolve(config["inputs"]["observed_features"]["path"]))
    manifest = pd.read_csv(resolve(config["inputs"]["iid_manifest"]["path"]), usecols=["molecule_id", "partition"])
    train_features = manifest.loc[manifest.partition.ne("historical_quarantine"), ["molecule_id"]].merge(
        observed_train, on="molecule_id", validate="one_to_one"
    )
    if len(train_features) != 15015 or not np.isfinite(train_features[C0_COLUMNS].to_numpy(np.float64)).all():
        raise RuntimeError("observed feature boundary failure")
    # Exact empirical percentiles for the 20 continuous descriptors.
    robust = np.empty((len(candidate), 20), dtype=np.float32)
    for j in range(20):
        reference = np.sort(train_features[C0_COLUMNS[j]].to_numpy(np.float64))
        robust[:, j] = np.searchsorted(reference, x_candidate[:, j], side="right") / len(reference)
    similarity = nearest_similarity(x_candidate[:, 20:].astype(np.uint8), x_observed[:, 20:].astype(np.uint8))
    candidate_out = candidate.drop(columns=["canonical_smiles"]).copy()
    candidate_out["canonical_smiles"] = candidate["canonical_smiles"].astype(str)
    for j, name in enumerate(C0_COLUMNS):
        candidate_out[name] = x_candidate[:, j]
    candidate_out["nearest_observed_morgan_similarity"] = similarity
    candidate_out["descriptor_outside_1_99_count"] = ((robust < 0.01) | (robust > 0.99)).sum(axis=1).astype(np.int16)
    candidate_out.to_parquet(resolve(config["local_outputs"]["candidate_features"]), index=False)
    observed_out = observed[["pair_hash", "donor_identity_hash", "acceptor_identity_hash", "full_structure_hash", "canonical_smiles"]].copy()
    for j, name in enumerate(C0_COLUMNS):
        observed_out[name] = x_observed[:, j]
    observed_out.to_parquet(resolve(config["local_outputs"]["observed_control_features"]), index=False)
    domain = {
        "status": "CANDIDATE_FEATURES_FROZEN", "candidate_count": len(candidate_out),
        "observed_control_structures": len(observed_out), "columns": C0_COLUMNS,
        "column_count": 532, "column_order_sha256": ordered_hash(C0_COLUMNS),
        "candidate_feature_sha256": sha256(resolve(config["local_outputs"]["candidate_features"])),
        "observed_control_feature_sha256": sha256(resolve(config["local_outputs"]["observed_control_features"])),
        "rdkit_version": rdkit.__version__, "dtype": "float32",
        "morgan": {"radius": 2, "nBits": 512, "includeChirality": False},
        "finite": True, "duplicate_candidate_hashes": 0,
        "nearest_observed_similarity": {q: float(np.quantile(similarity, float(q))) for q in ("0", "0.1", "0.5", "0.9", "1")},
        "descriptor_outside_1_99": {
            "mean_columns_per_candidate": float(candidate_out.descriptor_outside_1_99_count.mean()),
            "fraction_any": float((candidate_out.descriptor_outside_1_99_count > 0).mean()),
        },
        "domain_interpretation": "target-free diagnostics only; AD_SCORE_NOT_VALIDATED remains frozen",
    }
    write_json(ROOT / "data_registry/gate3a1_feature_registry.json", domain)
    (ROOT / "reports/gate3a1_candidate_feature_domain.md").write_text(
        "# Gate 3-A1 candidate feature and domain audit\n\n"
        f"All {len(candidate_out):,} in-domain unseen pairs produced the exact 532-column C0 contract with no "
        f"sanitize, duplicate-hash, NaN, or Inf failure. Nearest-observed full-Morgan similarity has median "
        f"{domain['nearest_observed_similarity']['0.5']:.6f}; "
        f"{domain['descriptor_outside_1_99']['fraction_any']:.2%} have at least one descriptor outside the "
        "observed 1st-99th percentile band. These are target-free input-domain diagnostics, not a validated "
        "applicability domain. Candidate molecular sizes remain unusually large, and structural alerts are "
        "heuristics rather than synthesis or safety evidence.\n"
    )
    print(json.dumps({"status": domain["status"], "candidates": len(candidate_out), "controls": len(observed_out)}, indent=2))


def xgb_params(config: dict, device: str) -> dict:
    allowed = {k: v for k, v in config["xgboost"].items() if k not in {"row_subsampling", "column_subsampling"}}
    allowed["device"] = device
    return allowed


def train_models(config: dict, device: str, physical_gpu: int | None) -> None:
    if not (ROOT / "data_registry/gate3a1_feature_registry.json").is_file():
        raise RuntimeError("feature registry missing")
    model_registry_path = ROOT / "data_registry/gate3a1_model_registry.json"
    if model_registry_path.exists():
        raise RuntimeError("models already frozen")
    manifest = pd.read_csv(resolve(config["inputs"]["iid_manifest"]["path"]))
    manifest["molecule_id"] = manifest["molecule_id"].astype(str)
    eligible = manifest[manifest.partition.ne("historical_quarantine")].copy()
    labels = pd.read_parquet(resolve(config["local_outputs"]["training_labels"]))
    features = pd.read_parquet(resolve(config["inputs"]["observed_features"]["path"]))
    frame = eligible.merge(labels, on="molecule_id", validate="one_to_one").merge(features, on="molecule_id", validate="one_to_one")
    target = config["primary_target"]
    x = frame[C0_COLUMNS].to_numpy(np.float64)
    y = frame[target].to_numpy(np.float64)
    base_weight = frame.group_weight.to_numpy(np.float64)
    groups = frame.structure_group_id_v1.astype(str).to_numpy()
    unique_groups = np.array(sorted(np.unique(groups)))
    if len(frame) != 15015 or len(unique_groups) != 14638 or not np.isclose(base_weight.sum(), 14638):
        raise RuntimeError("deployment training boundary failure")
    model_root = resolve(config["local_outputs"]["models"])
    model_root.mkdir(parents=True, exist_ok=False)
    entries = []

    def fit_one(label: str, weights: np.ndarray, seed: int) -> dict:
        active = weights > 0
        prep = fit_preprocessor(x[active], weights[active])
        prep_path = model_root / f"{label}_preprocessor.npz"
        save_preprocessor(prep_path, prep)
        model = XGBRegressor(**xgb_params(config, device))
        started = time.perf_counter()
        model.fit(transform(x[active], prep), y[active], sample_weight=weights[active])
        elapsed = time.perf_counter() - started
        path = model_root / f"{label}.json"
        model.save_model(path)
        return {
            "label": label, "seed": seed, "model_path": str(path.relative_to(ROOT)),
            "model_sha256": sha256(path), "preprocessor_path": str(prep_path.relative_to(ROOT)),
            "preprocessor_sha256": sha256(prep_path), "active_records": int(active.sum()),
            "sample_weight_sum": float(weights.sum()), "wall_seconds": elapsed,
        }

    entries.append(fit_one("final", base_weight, 42))
    for seed in config["stability"]["seeds"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        values, counts = np.unique(sampled, return_counts=True)
        multiplicity = dict(zip(values.tolist(), counts.tolist()))
        weights = base_weight * np.array([multiplicity.get(group, 0) for group in groups], dtype=np.float64)
        if not np.isclose(weights.sum(), len(unique_groups)):
            raise RuntimeError("bootstrap group-weight sum mismatch")
        entries.append(fit_one(f"bootstrap_{seed}", weights, seed))
    registry = {
        "status": "FINAL_AND_STABILITY_MODELS_FROZEN_BEFORE_CANDIDATE_INFERENCE",
        "model_count": len(entries), "final_models": 1, "stability_models": 20,
        "device": device, "physical_gpu": physical_gpu, "models": entries,
        "feature_columns_sha256": ordered_hash(C0_COLUMNS), "training_records": len(frame),
        "training_structure_groups": len(unique_groups), "target_aggregation": False,
        "replicates_retained": True, "group_weight": "1/structure_group_size times bootstrap multiplicity",
        "hyperparameter_search": False, "candidate_prediction_run": False,
        "test_artifact_access": False, "final673_access": False,
        "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "python": platform.python_version()},
    }
    write_json(model_registry_path, registry)
    print(json.dumps({"status": registry["status"], "models": len(entries), "device": device}, indent=2))


def bitvectors(matrix: np.ndarray) -> list:
    result = []
    for row in matrix:
        fp = DataStructs.ExplicitBitVect(512)
        for bit in np.flatnonzero(row):
            fp.SetBit(int(bit))
        result.append(fp)
    return result


def add_selected(row: pd.Series, category: str, selected: list, donor_counts: dict, acceptor_counts: dict) -> None:
    item = row.to_dict()
    item["category"] = category
    selected.append(item)
    donor_counts[row.donor_identity_hash] = donor_counts.get(row.donor_identity_hash, 0) + 1
    acceptor_counts[row.acceptor_identity_hash] = acceptor_counts.get(row.acceptor_identity_hash, 0) + 1


def score_once(config: dict) -> None:
    sentinel = resolve(config["local_outputs"]["scoring_sentinel"])
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Gate 3-A1 candidate scoring is fail-closed after first invocation") from exc
    with os.fdopen(fd, "w") as handle:
        handle.write(f"started={utc()}\n")
    registry = json.loads((ROOT / "data_registry/gate3a1_model_registry.json").read_text())
    if registry["status"] != "FINAL_AND_STABILITY_MODELS_FROZEN_BEFORE_CANDIDATE_INFERENCE" or len(registry["models"]) != 21:
        raise RuntimeError("frozen model registry failure")
    candidate = pd.read_parquet(resolve(config["local_outputs"]["candidate_features"]))
    if len(candidate) != 36523 or candidate.pair_hash.nunique() != 36523:
        raise RuntimeError("candidate feature boundary failure")
    matrix = candidate[C0_COLUMNS].to_numpy(np.float64)
    predictions = []
    for entry in registry["models"]:
        model_path = ROOT / entry["model_path"]
        prep_path = ROOT / entry["preprocessor_path"]
        if sha256(model_path) != entry["model_sha256"] or sha256(prep_path) != entry["preprocessor_sha256"]:
            raise RuntimeError("model artifact changed after freeze")
        model = XGBRegressor()
        model.load_model(model_path)
        predictions.append(model.predict(transform(matrix, load_preprocessor(prep_path))).astype(np.float64))
    prediction = np.vstack(predictions)
    point, boot = prediction[0], prediction[1:]
    hashes = candidate.pair_hash.astype(str).to_numpy()
    ranks = np.vstack([deterministic_rank_percentile(row, hashes) for row in boot])
    frac = float(config["stability"]["extreme_pool_fraction"])
    candidate["point_prediction"] = point
    candidate["bootstrap_median"] = np.median(boot, axis=0)
    candidate["bootstrap_q10"] = np.quantile(boot, 0.10, axis=0)
    candidate["bootstrap_q90"] = np.quantile(boot, 0.90, axis=0)
    candidate["prediction_std"] = np.std(boot, axis=0, ddof=1)
    candidate["rank_percentile_median"] = np.median(ranks, axis=0)
    candidate["low_extreme_inclusion_frequency"] = np.mean(ranks <= frac, axis=0)
    candidate["high_extreme_inclusion_frequency"] = np.mean(ranks >= 1.0 - frac, axis=0)
    candidate.to_parquet(resolve(config["local_outputs"]["scoring"]), index=False)

    selected, donor_counts, acceptor_counts = [], {}, {}
    threshold = float(config["stability"]["extreme_inclusion_frequency_min"])
    low = candidate[candidate.low_extreme_inclusion_frequency.ge(threshold)].sort_values(
        ["point_prediction", "pair_hash"], ascending=[True, True], kind="mergesort"
    )
    for _, row in low.iterrows():
        if identity_cap_ok(row, donor_counts, acceptor_counts):
            add_selected(row, "predicted_low_proxy", selected, donor_counts, acceptor_counts)
        if sum(x["category"] == "predicted_low_proxy" for x in selected) == 4:
            break
    high = candidate[candidate.high_extreme_inclusion_frequency.ge(threshold)].sort_values(
        ["point_prediction", "pair_hash"], ascending=[False, True], kind="mergesort"
    )
    for _, row in high.iterrows():
        if identity_cap_ok(row, donor_counts, acceptor_counts):
            add_selected(row, "predicted_high_proxy", selected, donor_counts, acceptor_counts)
        if sum(x["category"] == "predicted_high_proxy" for x in selected) == 4:
            break

    used = {x["pair_hash"] for x in selected}
    pool = candidate[~candidate.pair_hash.isin(used)].copy().reset_index(drop=True)
    pool_fps = bitvectors(pool[C0_COLUMNS[20:]].to_numpy(np.uint8))
    exploration_indices = []
    similarity_to_selected = np.zeros(len(pool), dtype=np.float64)
    while len(exploration_indices) < 4:
        eligible = np.array([
            i not in exploration_indices and identity_cap_ok(pool.iloc[i], donor_counts, acceptor_counts)
            for i in range(len(pool))
        ])
        if not eligible.any():
            break
        if not exploration_indices:
            order = np.lexsort((pool.pair_hash.astype(str).to_numpy(), pool.nearest_observed_morgan_similarity.to_numpy()))
            chosen = next(int(i) for i in order if eligible[i])
        else:
            prior = pool_fps[exploration_indices[-1]]
            similarity_to_selected = np.maximum(similarity_to_selected, np.asarray(DataStructs.BulkTanimotoSimilarity(prior, pool_fps)))
            keys = np.lexsort((pool.pair_hash.astype(str).to_numpy(), pool.nearest_observed_morgan_similarity.to_numpy(), similarity_to_selected))
            chosen = next(int(i) for i in keys if eligible[i])
        exploration_indices.append(chosen)
        add_selected(pool.iloc[chosen], "diversity_exploration", selected, donor_counts, acceptor_counts)

    observed = pd.read_parquet(resolve(config["local_outputs"]["observed_control_features"]))
    observed_fps = bitvectors(observed[C0_COLUMNS[20:]].to_numpy(np.uint8))
    used_control_structures = set()
    exploration = [x for x in selected if x["category"] == "diversity_exploration"]
    for item in exploration:
        candidate_fp = bitvectors(np.asarray([[item[name] for name in C0_COLUMNS[20:]]], dtype=np.uint8))[0]
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(candidate_fp, observed_fps))
        order = np.lexsort((observed.pair_hash.astype(str).to_numpy(), -similarities))
        for index in order:
            row = observed.iloc[int(index)]
            if row.full_structure_hash in used_control_structures or not identity_cap_ok(row, donor_counts, acceptor_counts):
                continue
            control = row.copy()
            control["matched_similarity"] = float(similarities[int(index)])
            add_selected(control, "matched_observed_control", selected, donor_counts, acceptor_counts)
            used_control_structures.add(row.full_structure_hash)
            break

    category_counts = pd.Series([x["category"] for x in selected]).value_counts().to_dict()
    complete = category_counts == {
        "predicted_low_proxy": 4, "predicted_high_proxy": 4,
        "diversity_exploration": 4, "matched_observed_control": 4,
    }
    unique_structures = len({x["full_structure_hash"] for x in selected}) == len(selected)
    caps = max(donor_counts.values(), default=0) <= 2 and max(acceptor_counts.values(), default=0) <= 2
    status = "COMPUTATIONAL_SHORTLIST_FROZEN" if complete and unique_structures and caps else "RANKING_UNSTABLE_NO_SHORTLIST"
    local = pd.DataFrame(selected)
    local.to_parquet(resolve(config["local_outputs"]["shortlist"]), index=False)
    public_items = []
    for index, item in enumerate(selected, start=1):
        public_items.append({
            "shortlist_id": f"G3A1-{index:02d}", "category": item["category"],
            "anonymous_pair_hash": item["pair_hash"], "anonymous_full_structure_hash": item["full_structure_hash"],
            "stable_extreme_gate": bool(
                item["category"] not in {"predicted_low_proxy", "predicted_high_proxy"} or
                max(float(item.get("low_extreme_inclusion_frequency", 0)), float(item.get("high_extreme_inclusion_frequency", 0))) >= threshold
            ),
        })
    shortlist_registry = {
        "status": status, "category_counts": category_counts, "items": public_items,
        "unique_full_structures": unique_structures, "identity_caps_pass": caps,
        "canonical_smiles_published": False, "full_ranking_published": False,
        "local_shortlist_path": config["local_outputs"]["shortlist"],
        "local_shortlist_sha256": sha256(resolve(config["local_outputs"]["shortlist"])),
        "interpretation": "computational proxy shortlist only",
    }
    write_json(ROOT / "data_registry/gate3a1_shortlist_registry.json", shortlist_registry)
    scoring_registry = {
        "status": "CANDIDATE_SCORING_CONSUMED_ONCE", "candidate_count": len(candidate),
        "model_registry_sha256": sha256(ROOT / "data_registry/gate3a1_model_registry.json"),
        "scoring_artifact_sha256": sha256(resolve(config["local_outputs"]["scoring"])),
        "point_prediction": {q: float(np.quantile(point, float(q))) for q in ("0", "0.1", "0.5", "0.9", "1")},
        "prediction_std": {q: float(np.quantile(candidate.prediction_std, float(q))) for q in ("0", "0.5", "0.9", "1")},
        "stable_low_pool": int(len(low)), "stable_high_pool": int(len(high)),
        "second_invocation": "FAIL_CLOSED", "candidate_labels_read": False,
        "test_artifact_access": False, "final673_access": False,
    }
    write_json(ROOT / "data_registry/gate3a1_scoring_unlock_v1.json", scoring_registry)
    (ROOT / "reports/gate3a1_ranking_stability.md").write_text(
        "# Gate 3-A1 ranking stability\n\n"
        f"One frozen deployment model and 20 fixed structure-group bootstrap models scored {len(candidate):,} candidates once. "
        f"The 1% extreme-pool stability gate (frequency at least {threshold:.0%}) contains {len(low):,} low-proxy and "
        f"{len(high):,} high-proxy candidates. Bootstrap quantiles, rank frequency, and standard deviation describe "
        "ranking stability only; they are not conformal, calibrated, confidence, prediction, or experimental intervals.\n"
    )
    (ROOT / "reports/gate3a1_computational_shortlist.md").write_text(
        "# Gate 3-A1 computational shortlist\n\n"
        f"Status: **{status}**. Category counts: {json.dumps(category_counts, sort_keys=True)}. "
        "The low group means only predicted low screened electron-hole Coulomb proxy under the frozen model. "
        "It does not mean high catalytic efficiency, best photocatalyst, superior charge separation, or experimental activity. "
        "High-proxy entries are computational controls, not experimental negatives. Public assets contain anonymous hashes only; "
        "SMILES, structures, full ranking, and per-candidate values remain local and Git-ignored.\n"
    )
    (ROOT / "reports/gate3a1_final_decision.md").write_text(
        "# Gate 3-A1 final decision\n\n"
        f"Decision: **{status}**.\n\n"
        "The scope remains seen donors plus seen acceptors in unseen pair combinations. It is not donor-OOD, acceptor-OOD, "
        "or new-chemistry extrapolation. Twelve observed reconstruction failures and 753 atom-origin-unverifiable records remain "
        "documented boundaries. Candidate molecular size is large, structural alerts are heuristics, and no synthesis route, "
        "reaction condition, catalytic property, or experimental validation exists. Therefore "
        "`PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY` and `BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH` remain active.\n"
    )
    evidence = {
        "status": status, "scoring_once": True, "candidate_count": len(candidate), "shortlist_count": len(selected),
        "category_counts": category_counts, "models_frozen_before_scoring": True,
        "feature_coverage": 1.0, "identity_caps_pass": caps, "unique_structures": unique_structures,
        "training": "one final plus 20 fixed structure-group bootstrap XGBoost-C0 models",
        "new_architecture_search": False, "candidate_labels_read": False, "test_artifact_access": False,
        "final673_access": False, "main_parquet_read": False, "gpu_model_selection": False,
        "publication_boundary": ["PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY", "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH"],
        "ended_utc": utc(),
    }
    write_json(ROOT / "logs/gate3a1_evidence.json", evidence)
    print(json.dumps({"status": status, "category_counts": category_counts, "stable_low": len(low), "stable_high": len(high)}, indent=2))


def finalize(config: dict) -> None:
    required = [
        "configs/gate3a1_prospective_scoring_v1.json",
        "data_registry/gate3a1_preregistration_lock_v1.json",
        "data_registry/gate3a1_label_coverage_registry.json",
        "data_registry/gate3a1_feature_registry.json",
        "data_registry/gate3a1_model_registry.json",
        "data_registry/gate3a1_scoring_unlock_v1.json",
        "data_registry/gate3a1_shortlist_registry.json",
        "reports/gate3a1_preregistration.md",
        "reports/gate3a1_candidate_feature_domain.md",
        "reports/gate3a1_ranking_stability.md",
        "reports/gate3a1_computational_shortlist.md",
        "reports/gate3a1_final_decision.md",
        "logs/gate3a1_evidence.json",
        "scripts/gate3a1_prospective_scoring.py",
        "src/excitationnexus_phase12/prospective_scoring.py",
        "tests/test_gate3a1_contract.py",
    ]
    lines = [f"{sha256(ROOT / path)}  {path}" for path in required]
    (ROOT / "data_registry/gate3a1_sha256.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "GATE3A1_SHA_FROZEN", "files": len(required)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preregister", "features", "train", "score", "finalize"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    config = load_config(ROOT)
    if args.stage == "preregister":
        preregister(config)
    elif args.stage == "features":
        build_features(config)
    elif args.stage == "train":
        train_models(config, args.device, args.physical_gpu)
    elif args.stage == "score":
        score_once(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
