from snowpack import bulk_temperature
from .fixtures import simple_mesh as mesh

def test_bulk_temperature_parameter(mesh):
    params = bulk_temperature.BulkTemperatureParameter(V=None, mesh=mesh, dt=0.01)
    assert params is not None