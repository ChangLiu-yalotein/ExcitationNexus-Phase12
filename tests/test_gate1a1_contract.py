import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gate1a1_feature_contract_is_frozen_and_no_dipole():
    registry = json.loads((ROOT / "data_registry/gate1a1_feature_columns_v1.json").read_text())
    columns = registry["columns"]
    assert registry["count"] == len(columns) == 541
    assert len(columns) == len(set(columns))
    assert columns[:2] == ["pair_MolWt", "pair_MolLogP"]
    assert columns[20] == "pair_morgan_0"
    assert columns[532:] == [
        "pm6_homo_hartree", "pm6_lumo_hartree", "pm6_homo_lumo_gap_hartree",
        "pm6_homo_lumo_gap_ev", "pm6_pm6_energy_hartree", "pm6_num_atoms",
        "pm6_normal_termination", "pm6_n_warnings", "pm6_missing_flag",
    ]
    assert not any("dipole" in column.lower() for column in columns)
    digest = hashlib.sha256(("\n".join(columns) + "\n").encode()).hexdigest()
    assert digest == registry["ordered_columns_sha256"]


def test_gate1a1_preregistration_records_historical_selection_limitation():
    config = json.loads((ROOT / "configs/gate1a1_cheap_reproduction_v1.json").read_text())
    assert config["formal_runs"] == 1
    assert config["hyperparameter_search"] is False
    assert config["early_stopping"] is None
    assert config["selection_limitation"]["historical_test_used_to_select_best_across_configs"] is True
    assert config["final673_access"] is False
    assert config["new15016_access"] is False
