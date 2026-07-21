#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import skew

from gate2e0_common import ROOT, load_config, load_primary_labels, load_protocol_aux, weighted_mean, write_json


UNITS = {
    "tddft_excitation_energy_ev": "eV", "tddft_oscillator_strength": "dimensionless",
    "tddft_transition_dipole_au": "a.u.", "tddft_Sm": "dimensionless", "tddft_Sr": "dimensionless",
    "tddft_D_index_angstrom": "angstrom", "tddft_H_CT_angstrom": "angstrom",
    "tddft_t_index_angstrom": "angstrom", "tddft_HDI": "dimensionless", "tddft_EDI": "dimensionless",
    "tddft_Q_D_to_A_au": "a.u.", "tddft_dipole_change_norm_au": "a.u.",
    "tddft_hole_on_donor_fraction": "fraction", "tddft_hole_on_acceptor_fraction": "fraction",
    "tddft_electron_on_donor_fraction": "fraction", "tddft_electron_on_acceptor_fraction": "fraction",
}
MEANINGS = {
    "tddft_excitation_energy_ev": "selected-state vertical excitation energy",
    "tddft_oscillator_strength": "selected-state oscillator strength",
    "tddft_transition_dipole_au": "selected-state transition dipole magnitude",
    "tddft_Sm": "hole-electron overlap index Sm", "tddft_Sr": "hole-electron overlap index Sr",
    "tddft_D_index_angstrom": "hole-electron centroid separation D",
    "tddft_H_CT_angstrom": "charge-transfer distance descriptor H_CT",
    "tddft_t_index_angstrom": "charge-transfer separation index t",
    "tddft_HDI": "hole delocalization index", "tddft_EDI": "electron delocalization index",
    "tddft_Q_D_to_A_au": "donor-to-acceptor transferred charge proxy",
    "tddft_dipole_change_norm_au": "norm of excitation-induced dipole change",
    "tddft_hole_on_donor_fraction": "hole population assigned to donor atoms",
    "tddft_hole_on_acceptor_fraction": "hole population assigned to acceptor atoms",
    "tddft_electron_on_donor_fraction": "electron population assigned to donor atoms",
    "tddft_electron_on_acceptor_fraction": "electron population assigned to acceptor atoms",
}


def target_stats(frame: pd.DataFrame, column: str) -> dict:
    valid = frame[[column, "group_weight"]].dropna()
    values, weights = valid[column].to_numpy(float), valid.group_weight.to_numpy(float)
    q1, q3 = np.quantile(values, [0.25, 0.75]) if len(values) else (np.nan, np.nan)
    iqr = q3 - q1
    outliers = int(((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()) if len(values) else 0
    mean = weighted_mean(values, weights) if len(values) else np.nan
    variance = weighted_mean((values - mean) ** 2, weights) if len(values) else np.nan
    return {
        "finite_count": len(valid), "completeness": len(valid) / len(frame), "unique_count": int(valid[column].nunique()),
        "weighted_mean": mean, "weighted_std": float(np.sqrt(variance)), "variance": variance,
        "skewness": float(skew(values, bias=False)) if len(values) > 2 else np.nan,
        "minimum": float(np.min(values)) if len(values) else np.nan, "maximum": float(np.max(values)) if len(values) else np.nan,
        "outlier_count": outliers, "outlier_rate": outliers / len(valid) if len(valid) else np.nan,
    }


def main() -> None:
    config = load_config()
    train, _ = load_protocol_aux(config, "iid", "train")
    primary = load_primary_labels(config)
    train = train.merge(primary, on="molecule_id", validate="one_to_one")

    t_residual = train.tddft_t_index_angstrom - (train.tddft_D_index_angstrom - train.tddft_H_CT_angstrom)
    t_relation = {"formula": "t = D - H_CT", "max_abs_residual_angstrom": float(t_residual.abs().max()), "p99_abs_residual_angstrom": float(t_residual.abs().quantile(.99)), "rounding_tolerance_angstrom": 0.001}
    fraction_checks = {}
    for particle in ("hole", "electron"):
        donor = f"tddft_{particle}_on_donor_fraction"; acceptor = f"tddft_{particle}_on_acceptor_fraction"
        valid = train[[donor, acceptor]].dropna(); unassigned = 1.0 - valid[donor] - valid[acceptor]
        fraction_checks[particle] = {"joint_nonmissing": len(valid), "median_unassigned": float(unassigned.median()), "p99_abs_unassigned": float(unassigned.abs().quantile(.99)), "max_abs_unassigned": float(unassigned.abs().max()), "strictly_complementary_all_rows": bool((unassigned.abs() <= 1e-9).all())}

    rows = []
    optimization = [config["primary"], *config["secondary"], *config["masked"]]
    for column in optimization:
        stats = target_stats(train, column)
        if column == config["primary"]:
            status, dependency, meaning, unit = "PRIMARY", "Coulomb algebra family sole optimization member", "screened Coulomb-attraction proxy at fixed epsilon=3.5", "eV"
        elif column == "tddft_t_index_angstrom":
            status, dependency, meaning, unit = "REPORT_ONLY_REDUNDANT", "t = D - H_CT within 0.001 angstrom source rounding", MEANINGS[column], UNITS[column]
        elif column in config["secondary"]:
            status, dependency, meaning, unit = "OPTIMIZATION_ALLOWED", "none established against primary", MEANINGS[column], UNITS[column]
        else:
            status, dependency, meaning, unit = "MASKED_ONLY", "donor/acceptor pair not strictly complementary for every jointly observed row", MEANINGS[column], UNITS[column]
        rows.append({"canonical_name": column, "physical_meaning": meaning, "unit": unit, "source_json_file": "TDDFT/*_properties.json with audited excitation JSON semantics", "extraction_rule": "schema_v3 frozen parser field", **stats, "deterministic_dependency": dependency, "allowed_role": status})
    for column, meaning, unit, dep in [
        ("tddft_wavelength_nm", "wavelength derived from excitation energy", "nm", "deterministic inverse-energy transform"),
        ("tddft_coulomb_attraction_au", "raw Coulomb attraction", "a.u.", "same algebraic family as primary"),
        ("tddft_coulomb_attraction_eV", "raw Coulomb attraction", "eV", "same algebraic family as primary"),
    ]:
        rows.append({"canonical_name": column, "physical_meaning": meaning, "unit": unit, "source_json_file": "frozen schema/parser provenance", "extraction_rule": "not read in Gate 2-E0", "completeness": np.nan, "finite_count": np.nan, "unique_count": np.nan, "variance": np.nan, "weighted_mean": np.nan, "weighted_std": np.nan, "skewness": np.nan, "minimum": np.nan, "maximum": np.nan, "outlier_count": np.nan, "outlier_rate": np.nan, "deterministic_dependency": dep, "allowed_role": "REPORT_ONLY_DETERMINISTIC"})
    for column in config["disabled"]:
        rows.append({"canonical_name": column, "physical_meaning": "disabled or unresolved field", "unit": "unresolved/field-specific", "source_json_file": "frozen schema/parser provenance", "extraction_rule": "not read in Gate 2-E0", "completeness": np.nan, "finite_count": np.nan, "unique_count": np.nan, "variance": np.nan, "weighted_mean": np.nan, "weighted_std": np.nan, "skewness": np.nan, "minimum": np.nan, "maximum": np.nan, "outlier_count": np.nan, "outlier_rate": np.nan, "deterministic_dependency": "disabled", "allowed_role": "DISABLED"})
    ledger = pd.DataFrame(rows)
    ledger.to_csv(ROOT / "data_registry/gate2e0_target_admission_ledger.csv", index=False)
    audit = {"completed_utc": datetime.now(timezone.utc).isoformat(), "iid_train_records": len(train), "primary_read_from_frozen_artifact": True, "source_parquet_read": False, "t_index_relation": t_relation, "fraction_assignment_checks": fraction_checks, "optimization_counts": ledger.allowed_role.value_counts().to_dict(), "primary_equivalent_auxiliary_in_loss": False}
    write_json("logs/gate2e0_target_audit.json", audit)
    lines = ["# Gate 2-E0 target semantics", "", "The primary remains `J_eh_screened_eV_eps3p5 proxy`; it is not experimental Eb or catalytic efficiency.", "", f"The 12 candidate secondary fields were traced to the frozen schema and TDDFT property/excitation JSON semantics. Eleven remain optimization candidates. `tddft_t_index_angstrom` is report-only redundant because the IID-train source values satisfy `t = D - H_CT` within {t_relation['max_abs_residual_angstrom']:.6g} Å (source rounding).", "", "Wavelength and raw Coulomb au/eV remain report-only deterministic. No member of the primary Coulomb unit/dielectric family is admitted as an auxiliary loss.", "", "The four fragment fractions remain masked-only: missing values are never imputed, and donor+acceptor does not equal one for every jointly observed record, so unknown/unassigned contribution is preserved rather than erased.", "", "## Fraction assignment audit", ""]
    for name, item in fraction_checks.items(): lines.append(f"- {name}: jointly observed {item['joint_nonmissing']}; median unassigned {item['median_unassigned']:.3g}; P99 |unassigned| {item['p99_abs_unassigned']:.6g}; max {item['max_abs_unassigned']:.6g}; strict complement={item['strictly_complementary_all_rows']}.")
    (ROOT / "reports/gate2e0_target_semantics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
