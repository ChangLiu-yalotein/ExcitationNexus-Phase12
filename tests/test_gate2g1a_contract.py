import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gate2g1a_preregistration_boundaries():
    config = json.loads((ROOT / "configs/gate2g1a_unified25703_data_contract_v1.json").read_text())
    assert config["execution"] == "CPU_ONLY_DATA_GOVERNANCE_NO_TRAINING_NO_GPU"
    assert config["final673_label_access"] is False
    assert config["legacy3371_policy"] == "MEMBERSHIP_ALIAS_ONLY_DO_NOT_APPEND"
    assert config["gate3_shortlist_status"] == "EXPLORATORY_BASELINE_SHORTLIST_FROZEN"


def test_gate2g1a_dataset_registry_when_present():
    path = ROOT / "data_registry/unified25703_dataset_registry.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    assert reg["total_records"] == 25703
    assert reg["development_records"] == 25030
    assert reg["final_confirmation_records"] == 673
    assert reg["legacy3371_decomposition"]["external2698"] == 2698
    assert reg["legacy3371_decomposition"]["final673"] == 673
    assert reg["legacy3371_decomposition"]["sum"] == 3371


def test_gate2g1a_final673_public_boundary_when_present():
    source = ROOT / "data_registry/unified25703_source_registry.json"
    if not source.exists():
        return
    reg = json.loads(source.read_text())
    assert reg["final673_boundary"]["label_access"] is False
    assert reg["final673_boundary"]["public_membership"] == "aggregate counts only"
    final_report = (ROOT / "reports/gate2g1a_final673_boundary.md").read_text()
    assert "Targets, IDs, SMILES, and per-sample membership are not published" in final_report
