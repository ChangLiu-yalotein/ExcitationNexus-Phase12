from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
TABLE_SHA256 = "e7587b1546039f099a4dbd0d352e98885bb2ebdbdcfa18884dd4355eed815a83"
MANIFEST_SHA256 = {
    "iid_group_seed42_v1": "f4572f2c1896d4228dd9eff67220adb7d0a02ad79b70c66766e6da876541c3f2",
    "donor_cold_v1": "505e4bbe4ed20900ed66721d20df5442854c0f0ea04f5885df330bde111427cd",
    "acceptor_cold_v1": "659992cc7413094fd17da2281902aa74f261307d98648a973254bd472b5ecc65",
    "pair_cold_v1": "06aaee7d7756957bc42f7d4e221ffd5b20916be69b3116c3bbbe017eb02f4825",
    "both_cold_external_test_v1": "f74f40c4a2f5e7d91214ebdabc3e8f2d1a48a3c106509bb8ab7a3c846b83cb16",
    "full_scaffold_cold_v1": "d395020eccfa6e8b73d50bc788a11261d308403a731a9e448040ffe7d238a897",
}
MANIFEST_FILES = {
    "iid_group_seed42_v1": "split_iid_group_seed42_v1.csv",
    "donor_cold_v1": "split_donor_cold_v1.csv",
    "acceptor_cold_v1": "split_acceptor_cold_v1.csv",
    "pair_cold_v1": "split_pair_cold_v1.csv",
    "both_cold_external_test_v1": "split_both_cold_external_test_v1.csv",
    "full_scaffold_cold_v1": "split_full_scaffold_cold_v1.csv",
}
ALLOWED_PARTITIONS = frozenset({"train", "val", "test"})
FORBIDDEN_INPUT_PATTERNS = (
    "tddft_*", "multiwfn_*", "target_*", "*_label", "*source_path*",
    "*method*", "*basis*", "*program*", "*termination*", "*parser_version*",
    "*final673*", "in_final*", "partition", "partition_code", "split_code",
)
COULOMB_EQUIVALENTS = re.compile(r"(coulomb|j_eh|eps3p5|screened.*proxy)", re.I)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs(table_path: str | Path, manifest_dir: str | Path) -> dict[str, str]:
    observed = {"table": sha256_file(table_path)}
    if observed["table"] != TABLE_SHA256:
        raise RuntimeError("BLOCKED_INPUT_HASH_MISMATCH:table")
    for name, filename in MANIFEST_FILES.items():
        observed[name] = sha256_file(Path(manifest_dir) / filename)
        if observed[name] != MANIFEST_SHA256[name]:
            raise RuntimeError(f"BLOCKED_INPUT_HASH_MISMATCH:{name}")
    return observed


def assert_input_fields_allowed(fields: Iterable[str], *, tier: str, pm6_dipole_enabled: bool = False) -> None:
    if tier not in {"table_only", "tier0_2d", "tier1_pm6_3d", "tier2_dft_3d"}:
        raise ValueError(f"unknown input tier: {tier}")
    for raw in fields:
        field = str(raw).lower()
        if any(fnmatch.fnmatch(field, p) for p in FORBIDDEN_INPUT_PATTERNS) or COULOMB_EQUIVALENTS.search(field):
            raise ValueError(f"TARGET_LEAKAGE_FIREWALL:{raw}")
        if field == "pm6_dipole_debye" and not pm6_dipole_enabled:
            raise ValueError("PM6_DIPOLE_DISABLED")
        if field.startswith("dft_") and tier != "tier2_dft_3d":
            raise ValueError(f"DFT_FEATURE_REQUIRES_TIER2:{raw}")


@dataclass(frozen=True)
class TaskGraph:
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    auxiliary: tuple[str, ...]
    report_only: tuple[str, ...]
    disabled: tuple[str, ...]

    @property
    def optimization_tasks(self) -> tuple[str, ...]:
        return self.primary + self.secondary + self.auxiliary

    @property
    def all_reportable_tasks(self) -> tuple[str, ...]:
        return self.optimization_tasks + self.report_only

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TaskGraph":
        data = json.loads(Path(path or ROOT / "data_registry/TARGET_TASK_GRAPH_V1.json").read_text())
        return cls(tuple(data["primary"]), tuple(data["secondary"]),
                   tuple(data["masked_auxiliary"]), tuple(data["report_only_deterministic"]),
                   tuple(x for x in data["disabled"] if isinstance(x, str)))


def allowed_scalar_fields(tier: str, pm6_dipole_enabled: bool = False) -> tuple[str, ...]:
    if tier in {"table_only", "tier0_2d"}:
        fields = ("num_atoms_total",)
    elif tier == "tier1_pm6_3d":
        fields = ("pm6_homo_hartree", "pm6_lumo_hartree", "pm6_gap_ev")
        if pm6_dipole_enabled:
            fields += ("pm6_dipole_debye",)
    elif tier == "tier2_dft_3d":
        fields = ("dft_homo_hartree", "dft_lumo_hartree", "dft_gap_ev", "dft_dipole_debye")
    else:
        raise ValueError(tier)
    assert_input_fields_allowed(fields, tier=tier, pm6_dipole_enabled=pm6_dipole_enabled)
    return fields
