#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n")


def metric(arm: dict) -> float:
    values = arm["validation"]
    return float(values.get("identity_macro_mae", values.get("acceptor_identity_macro_mae", values.get("structure_group_macro_mae"))))


def main() -> None:
    embedding = read("data_registry/gate2d2_v2_embedding_registry.json")
    models = read("data_registry/gate2d2_v2_model_registry.json")
    metrics = read("logs/gate2d2_v2_validation_metrics.json")
    mechanism = read("logs/gate2d2_v2_acceptor_mechanism.json")
    smoke = read("logs/gate2d2_v2_long_sequence_smoke.json")
    if metrics["decision"] != "REPRESENTATION_SIGNAL_INCONCLUSIVE" or models["new_models"] != 12:
        raise RuntimeError("finalization inputs incomplete")

    rows = []
    for protocol, info in models["protocols"].items():
        arms = info["arms"]
        rows.append(f"| {protocol} | {metric(arms['A_C0_512_reference']):.9f} | {metric(arms['B_MF_Full_RP512']):.9f} | {metric(arms['C_MF_Role_RP512']):.9f} |")
    write("reports/gate2d2_v2_validation_results.md", """# Gate 2-D2 v2 validation results

All values are validation-only hierarchical MAE in eV under each protocol's frozen inference unit. No test artifact was read.

| Protocol | C0-512 A | MF-Full-RP512 B | MF-Role-RP512 C |
|---|---:|---:|---:|
""" + "\n".join(rows) + f"""

Primary acceptor-cold comparisons (10,000 acceptor-identity bootstrap replicates):

- C−A: {metrics['primary']['acceptor_C_minus_A']['point']:+.9f} eV; 95% CI {metrics['primary']['acceptor_C_minus_A']['ci95']}.
- C−B: {metrics['primary']['acceptor_C_minus_B']['point']:+.9f} eV; 95% CI {metrics['primary']['acceptor_C_minus_B']['ci95']}.
- IID C−A: {metrics['primary']['iid_C_minus_A']['point']:+.9f} eV; 95% CI {metrics['primary']['iid_C_minus_A']['ci95']}.

Arm C beats the equal-budget continuous full-molecule control on acceptor-cold validation, but it neither reaches the frozen 0.003 eV improvement over C0 nor establishes IID non-inferiority. The evidence therefore does not admit this representation.
""")

    category_lines = []
    for kind, info in embedding["categories"].items():
        diagnostic = info["diagnostic"]
        category_lines.append(f"| {kind} | {info['identities']} | {info['max_token_length']} | {info['over_202_count']} | {info['raw_exact_collision_count']} | {diagnostic['cosine_distance_spearman']:.6f} | {diagnostic['pairwise_relative_error_median']:.6f} | {diagnostic['empirical_rank']} |")
    write("reports/gate2d2_v2_embedding_integrity.md", """# Gate 2-D2 v2 embedding integrity

The immutable safetensors SHA-256 matched the preregistered value. The true frozen-tokenizer maxima are 417/208/378 for full/donor/acceptor inputs; this target-free correction was locked before any molecular forward. The long-sequence gate passed without truncation.

| Input | Identities | Max tokens | >202 | Raw exact aliases | Cosine-distance Spearman | Median relative distance error | Empirical rank |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(category_lines) + f"""

- Repeated forward maximum absolute difference: {smoke['repeat_max_abs']}.
- Single-versus-padded-batch maximum absolute difference: {smoke['single_vs_padded_batch_max_abs']}.
- Full extraction batch-size sensitivity: {embedding['batch_size_sample_max_abs']}.
- Trainable encoder parameters: {embedding['trainable_parameters']}; optimizer parameters: {embedding['optimizer_parameters']}.

Every exact raw-embedding collision was traced to an identical frozen-tokenizer ID sequence. No distinct token sequence produced an identical embedding. These aliases are a representation limitation, not an extraction failure. Inputs above 202 tokens remain `OUTSIDE_PRETRAINING_LENGTH_SUPPORT` despite successful forward execution.
""")

    write("reports/gate2d2_v2_acceptor_mechanism.md", f"""# Gate 2-D2 v2 acceptor mechanism

- Acceptor identities in validation: {mechanism['acceptor_identities']}.
- Morgan nearest-train similarity versus Arm C identity error Spearman: {mechanism['morgan_similarity_vs_error_spearman']}.
- Frozen embedding nearest-train distance versus Arm C identity error Spearman: {mechanism['embedding_distance_vs_error_spearman']}.
- Lowest Morgan-similarity quartile mean Arm C identity error: {mechanism['quartiles']['Q1_low']['mean_error']:.9f} eV.
- Full/donor/acceptor identities above the documented pretraining length support: {mechanism['outside_pretraining_length_support']}.

Continuous role separation improved over the equal-budget full continuous arm, but the improvement over frozen C0 was smaller than the preregistered minimum and its CI crossed zero. Neither Morgan similarity nor embedding distance supplied a strong monotonic acceptor-risk explanation in the 32-identity validation set. Raw structures and SMILES are not published; worst identities remain anonymous hashes in local aggregate evidence.
""")

    write("reports/gate2d2_v2_final_decision.md", f"""# Gate 2-D2 v2 final decision

Final decision: `{metrics['decision']}`.

The primary C−A point estimate is {metrics['primary']['acceptor_C_minus_A']['point']:+.9f} eV with CI {metrics['primary']['acceptor_C_minus_A']['ci95']}; it misses both the −0.0030 eV effect threshold and the CI-below-zero rule. C−B is {metrics['primary']['acceptor_C_minus_B']['point']:+.9f} eV with CI {metrics['primary']['acceptor_C_minus_B']['ci95']}, supporting a role-separation signal relative to the continuous full control. IID C−A has CI {metrics['primary']['iid_C_minus_A']['ci95']}, whose upper bound exceeds +0.0020 eV, so non-inferiority also fails.

This is neither admission nor a definitive negative result. It is an inconclusive validation-only signal. No test prediction/label, source Parquet, buffer/quarantine, final673, encoder fine-tuning, projection search, or post-result model change was used.
""")

    tracked = [
        "configs/gate2d2_frozen_molformer_admission_v2.json",
        "configs/gate2d2_v2_sequence_length_correction.json",
        "data_registry/gate2d2_v2_fixed_projection_matrices.npz",
        "data_registry/gate2d2_v2_preregistration_lock.json",
        "data_registry/gate2d2_v2_projection_registry.json",
        "data_registry/gate2d2_v2_sequence_length_correction_lock.json",
        "data_registry/gate2d2_v2_embedding_registry.json",
        "data_registry/gate2d2_v2_model_registry.json",
        "logs/gate2d2_v2_long_sequence_smoke.json",
        "logs/gate2d2_v2_validation_metrics.json",
        "logs/gate2d2_v2_acceptor_mechanism.json",
        "logs/gate2d2_v2_evidence.json",
        "scripts/gate2d2_v2_analyze_validation.py",
        "scripts/gate2d2_v2_extract_embeddings.py",
        "scripts/gate2d2_v2_finalize.py",
        "scripts/gate2d2_v2_freeze_amendment.py",
        "scripts/gate2d2_v2_freeze_sequence_length_correction.py",
        "scripts/gate2d2_v2_long_sequence_smoke.py",
        "scripts/gate2d2_v2_model.py",
        "scripts/gate2d2_v2_train_validation_only.py",
        "tests/test_gate2d2_v2_contract.py",
        "tests/test_gate2d2_v2_outputs.py",
        "reports/gate2d2_v2_amendment.md",
        "reports/gate2d2_v2_sequence_length_correction.md",
        "reports/gate2d2_v2_embedding_integrity.md",
        "reports/gate2d2_v2_validation_results.md",
        "reports/gate2d2_v2_acceptor_mechanism.md",
        "reports/gate2d2_v2_final_decision.md"
    ]
    lines = [f"{sha256(ROOT / path)}  {path}" for path in tracked]
    write("data_registry/gate2d2_v2_sha256.txt", "\n".join(lines))
    print(json.dumps({"status": "GATE2D2_V2_FINALIZED", "decision": metrics["decision"], "sha_entries": len(lines)}, indent=2))


if __name__ == "__main__":
    main()
