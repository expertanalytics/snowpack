import pytest
from fenics import UnitIntervalMesh
from snowbird import bulk_temperature


@pytest.fixture(scope="session")
def simple_mesh():
    return UnitIntervalMesh(10)


@pytest.fixture(scope="session")
def simple_bulk_temp_parameters():
    return bulk_temperature.BulkTemperatureParameters(V=None, mesh=simple_mesh(), dt=0.01)
