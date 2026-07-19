import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    path = ROOT / "scripts/gate1a3_launch_b21_seeds123_456.py"
    spec = importlib.util.spec_from_file_location("gate1a3_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate1a3_config_is_frozen_to_two_seeds():
    path = ROOT / "configs/gate1a3_b21_seeds123_456_reproduction_v1.json"
    if not path.exists():
        return
    config = json.loads(path.read_text())
    assert config["seeds"] == [123, 456]
    assert config["training_contract"]["runs_per_seed"] == 1
    assert config["training_contract"]["epochs"] == 80
    assert config["final673_access"] is False
    assert config["new15016_access"] is False


def test_gpu_free_requires_all_three_conditions():
    module = load_launcher()
    source = (ROOT / "scripts/gate1a3_launch_b21_seeds123_456.py").read_text()
    assert 'uuid not in busy_uuids and int(memory) < 1024 and int(util) <= 5' in source
    assert module.ROOT == ROOT


def test_worker_outputs_are_seed_isolated():
    path = ROOT / "scripts/gate1a3_launch_b21_seeds123_456.py"
    source = path.read_text()
    assert "gate1a3-b21-seed{seed}-20260719" in source
    assert 'runs/gate1a3_b21_seed123' not in source
    assert "start_new_session=True" in source


def test_frozen_final_status_and_thresholds():
    path = ROOT / "runs/gate1a3_b21_multiseed/published/gate1a3_multiseed_metrics.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    assert result["status"] == "FAILED_REPRODUCTION"
    assert result["formal_training_runs"] == {"123": 1, "456": 1}
    assert result["test_inference_runs"] == {"123": 1, "456": 1}
    assert result["per_seed"]["42"]["within_0p001_eV"] is True
    assert result["per_seed"]["123"]["within_0p001_eV"] is False
    assert result["per_seed"]["456"]["within_0p001_eV"] is False
    assert result["aggregate"]["mean_within_0p001_eV"] is False
    assert result["final673_accessed"] is False
    assert result["new15016_accessed"] is False
