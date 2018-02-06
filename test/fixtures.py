import pytest
from fenics import UnitIntervalMesh, FunctionSpace
from snowbird import bulk_temperature


@pytest.fixture(scope="session")
def simple_mesh():
    return UnitIntervalMesh(10)


@pytest.fixture(scope="session")
def simple_bulk_temp_parameters():
    return bulk_temperature.BulkTemperatureParameters(V=None, mesh=simple_mesh(), dt=0.01)


@pytest.fixture(scope="session")
def simple_V(simple_mesh):
    return FunctionSpace(simple_mesh, "Lagrange", 1)

