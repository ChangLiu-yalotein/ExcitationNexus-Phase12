from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
TABLE=Path('/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet')
RAW=Path('/home/changliu/ExcitationNexus_Data_v2/raw_compact')

@pytest.fixture(scope='session')
def paths(): return {'root':ROOT,'table':TABLE,'raw':RAW,'manifests':ROOT/'manifests'}
