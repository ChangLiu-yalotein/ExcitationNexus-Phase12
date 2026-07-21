#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gate2d2_v2_sequence_length_correction.json"
LOCK = ROOT / "data_registry/gate2d2_v2_sequence_length_correction_lock.json"
REPORT = ROOT / "reports/gate2d2_v2_sequence_length_correction.md"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    if LOCK.exists() or REPORT.exists():
        raise RuntimeError("sequence-length correction already frozen; refuse overwrite")
    config = json.loads(CONFIG.read_text())
    if config["frozen_tokenizer_maxima"] != {"full": 417, "donor": 208, "acceptor": 378}:
        raise RuntimeError("unexpected corrected maxima")
    lock = {
        "status": "GATE2D2_V2_TOKEN_LENGTH_CORRECTION_FROZEN_BEFORE_FORWARD",
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "v2_preregistration_lock_sha256": sha256(ROOT / "data_registry/gate2d2_v2_preregistration_lock.json"),
        "molecular_forward_count_before_lock": 0,
        "embedding_count_before_lock": 0,
        "validation_model_count_before_lock": 0,
        "validation_metric_count_before_lock": 0,
        "test_artifacts_accessed": False,
        "main_parquet_accessed": False,
        "final673_accessed": False
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(
        "# Gate 2-D2 v2 sequence-length correction\n\n"
        "This target-free amendment was frozen before any molecular forward, embedding, or validation result. "
        "The v1 audit counted tokens with a regex; the immutable model tokenizer gives maxima 417/208/378 "
        "for full/donor/acceptor inputs rather than 399/208/372. The hard gate is strengthened to test the "
        "true maxima. No model, molecular string, projection, XGBoost contract, bootstrap rule, or admission "
        "threshold changed.\n"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
