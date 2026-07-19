import pytest
from excitationnexus_phase12.contracts import assert_input_fields_allowed

@pytest.mark.parametrize('field',['tddft_excitation_energy_ev','multiwfn_x','target_proxy','foo_label','dft_source_path','dft_method','normal_termination','final673_member','partition','J_eh_screened_eV_eps3p5_proxy'])
def test_forbidden_inputs_fail(field):
    with pytest.raises(ValueError): assert_input_fields_allowed([field],tier='tier1_pm6_3d')

def test_allowed_pm6_no_dipole():
    assert_input_fields_allowed(['pm6_homo_hartree','pm6_lumo_hartree','pm6_gap_ev'],tier='tier1_pm6_3d')

def test_pm6_dipole_flag():
    with pytest.raises(ValueError): assert_input_fields_allowed(['pm6_dipole_debye'],tier='tier1_pm6_3d',pm6_dipole_enabled=False)
    assert_input_fields_allowed(['pm6_dipole_debye'],tier='tier1_pm6_3d',pm6_dipole_enabled=True)
