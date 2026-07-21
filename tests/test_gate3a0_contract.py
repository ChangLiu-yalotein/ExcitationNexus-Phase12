import json
import subprocess
from pathlib import Path

from rdkit import Chem
from excitationnexus_phase12.combinatorial_assembly import (
    assemble_components, canonical_graph, parse_component_site, sha256_file, stable_json_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def test_preregistration_lock_and_firewall():
    config_path = ROOT / "configs/gate3a0_prospective_pair_feasibility_v1.json"
    lock = json.loads((ROOT / "data_registry/gate3a0_preregistration_lock_v1.json").read_text())
    config = json.loads(config_path.read_text())
    assert sha256_file(config_path) == lock["config_sha256"]
    assert config["execution"] == {
        "cpu_only": True, "training": False, "property_prediction": False,
        "candidate_ranking": False, "target_access": False, "test_artifact_access": False,
        "final673_access": False, "raw_data_write": False,
    }
    assert config["pair_cold_domain"]["donor_min_calculation_records"] == 5
    assert config["pair_cold_domain"]["acceptor_min_calculation_records"] == 5


def test_marker_rule_and_synthetic_product():
    donor = parse_component_site("c1ccccc1[A]", "donor", "D")
    acceptor = parse_component_site("[A]C", "acceptor", "A")
    product = assemble_components(donor, acceptor)
    assert canonical_graph(product) == canonical_graph(Chem.MolFromSmiles("Cc1ccccc1"))
    assert donor.placeholder_degree == acceptor.placeholder_degree == 1


def test_marker_contract_rejects_missing_or_multiple_markers():
    for value in ("CC", "C[A]C[A]"):
        try:
            parse_component_site(value, "donor", "x")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid marker contract was accepted")


def test_stable_hash_is_order_independent_after_sorting():
    rows = [("b", "2"), ("a", "1")]
    assert stable_json_sha256(sorted(rows)) == stable_json_sha256(sorted(reversed(rows)))


def test_generated_registry_contract():
    assembly_path = ROOT / "data_registry/gate3a0_assembly_audit_registry.json"
    universe_path = ROOT / "data_registry/gate3a0_candidate_universe_registry.json"
    if not assembly_path.exists() or not universe_path.exists():
        return
    assembly = json.loads(assembly_path.read_text())
    universe = json.loads(universe_path.read_text())
    assert assembly["metrics"]["record_coverage"] == 1.0
    assert assembly["metrics"]["verified_atom_origin_mismatch"] == 0
    assert assembly["metrics"]["deterministic_repeat"]
    assert universe["cartesian_pairs"] == 154 * 352
    assert sum(universe["status_counts"].values()) == universe["cartesian_pairs"]
    assert universe["in_domain_novel_pairs"] > 0


def test_local_candidate_artifacts_are_git_ignored():
    path = "runs/gate3a0_prospective_pair_feasibility/candidate_universe_v1.parquet"
    assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
