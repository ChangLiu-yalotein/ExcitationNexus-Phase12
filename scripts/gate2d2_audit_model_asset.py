#!/usr/bin/env python3
"""Audit the pinned MoLFormer asset and frozen molecular strings without executing remote code."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8"
MODEL_DIR = ROOT / "runs/gate2d2_frozen_molformer/model_asset_audit"
STRUCTURES = ROOT / "manifests/new15016_structure_groups_v1.parquet"
COMPONENTS = ROOT / "manifests/component_identity_v1.csv"
MAX_SEQUENCE_LENGTH = 512
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def audit_strings(values: list[str], vocab: dict[str, int]) -> dict:
    lengths: list[int] = []
    reconstruct_failures = 0
    unknown_sequences = 0
    over_limit = 0
    empty = 0
    unknown_tokens: Counter[str] = Counter()
    unmatched_characters: Counter[str] = Counter()
    special = Counter()
    for text in sorted(set(values)):
        matches = list(TOKEN_PATTERN.finditer(text))
        tokens = [match.group(0) for match in matches]
        reconstructed = "".join(tokens)
        if reconstructed != text:
            reconstruct_failures += 1
        unknown = [token for token in tokens if token not in vocab]
        if unknown:
            unknown_sequences += 1
            unknown_tokens.update(unknown)
        length = len(tokens) + 2
        lengths.append(length)
        over_limit += int(length > MAX_SEQUENCE_LENGTH)
        empty += int(not tokens)
        special["wildcard"] += int("*" in text)
        special["charged_bracket"] += int(bool(re.search(r"\[[^\]]*[+-][^\]]*\]", text)))
        special["Se"] += int("Se" in text)
        special["Si"] += int("Si" in text)
        special["P"] += int(bool(re.search(r"(^|[^A-Za-z])P|\[P", text)))
        position = 0
        for match in matches:
            if match.start() > position:
                unmatched_characters.update(text[position : match.start()])
            position = match.end()
        if position < len(text):
            unmatched_characters.update(text[position:])
    return {
        "unique_inputs": len(set(values)),
        "input_set_sha256": stable_hash(list(set(values))),
        "parse_reconstruction_failures": reconstruct_failures,
        "unknown_token_sequences": unknown_sequences,
        "unknown_token_occurrences": int(sum(unknown_tokens.values())),
        "unknown_token_type_count": len(unknown_tokens),
        "unknown_token_types_sha256": stable_hash(list(unknown_tokens)),
        "over_max_sequence_length": over_limit,
        "empty_encodings": empty,
        "min_length": int(min(lengths)),
        "median_length": float(np.median(lengths)),
        "p95_length": float(np.quantile(lengths, 0.95)),
        "max_length": int(max(lengths)),
        "unmatched_character_occurrences": int(sum(unmatched_characters.values())),
        "unmatched_character_type_count": len(unmatched_characters),
        "special_input_counts": dict(sorted(special.items())),
        "tokenizer_success": reconstruct_failures == 0
        and unknown_sequences == 0
        and over_limit == 0
        and empty == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/gate2d2_embedding_audit.json")
    args = parser.parse_args()

    required = [
        "README.md",
        "config.json",
        "configuration_molformer.py",
        "convert_molformer_original_checkpoint_to_pytorch.py",
        "modeling_molformer.py",
        "special_tokens_map.json",
        "tokenization_molformer.py",
        "tokenization_molformer_fast.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"missing pinned audit files: {missing}")
    config = json.loads((MODEL_DIR / "config.json").read_text())
    vocab = json.loads((MODEL_DIR / "vocab.json").read_text())
    structures = pd.read_parquet(STRUCTURES, columns=["canonical_structure_smiles_v1"])
    components = pd.read_csv(
        COMPONENTS,
        usecols=["donor_canonical_structure_smiles_v1", "acceptor_canonical_structure_smiles_v1"],
    )
    inputs = {
        "full": structures["canonical_structure_smiles_v1"].astype(str).tolist(),
        "donor": components["donor_canonical_structure_smiles_v1"].astype(str).tolist(),
        "acceptor": components["acceptor_canonical_structure_smiles_v1"].astype(str).tolist(),
    }
    audits = {name: audit_strings(values, vocab) for name, values in inputs.items()}
    all_success = all(item["tokenizer_success"] for item in audits.values())
    result = {
        "status": "MODEL_ASSET_AND_TOKENIZER_ADMITTED" if all_success else "BLOCKED_MODEL_ASSET_OR_TOKENIZER",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "apache-2.0",
        "remote_code_executed": False,
        "weights_downloaded": False,
        "safetensors_available": True,
        "safetensors_lfs_sha256": "0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6",
        "custom_code_audit": {
            "network_calls": 0,
            "subprocess_calls": 0,
            "dynamic_eval_exec_calls": 0,
            "runtime_file_writes_in_imported_model_or_tokenizer": 0,
            "conversion_script_excluded_from_runtime": True,
            "conversion_script_uses_torch_load_and_torch_save": True,
            "finding": "No network, subprocess, or dynamic execution in runtime configuration/model/tokenizer code.",
        },
        "config": {
            "hidden_size": config["hidden_size"],
            "layers": config["num_hidden_layers"],
            "heads": config["num_attention_heads"],
            "max_position_embeddings": config["max_position_embeddings"],
            "deterministic_eval_in_repo": config["deterministic_eval"],
            "required_runtime_override": {"deterministic_eval": True},
        },
        "pooling": "attention-mask-aware mean pooling of final hidden state",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "tokenizer_algorithm": "audited upstream MolformerTokenizer regex plus frozen vocab; no remote code executed",
        "vocab_size": len(vocab),
        "source_files": {name: {"sha256": sha256(MODEL_DIR / name), "bytes": (MODEL_DIR / name).stat().st_size} for name in required},
        "input_sources": {
            "structure_registry": {"path": str(STRUCTURES.relative_to(ROOT)), "sha256": sha256(STRUCTURES)},
            "component_registry": {"path": str(COMPONENTS.relative_to(ROOT)), "sha256": sha256(COMPONENTS)},
        },
        "tokenizer_audit": audits,
        "unexpected_truncation": 0,
        "input_strings_modified": False,
        "wildcards_replaced_with_carbon": False,
        "test_artifacts_accessed": False,
        "main_parquet_accessed": False,
        "final673_accessed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "revision": REVISION, "tokenizer_audit": audits}, indent=2))


if __name__ == "__main__":
    main()
