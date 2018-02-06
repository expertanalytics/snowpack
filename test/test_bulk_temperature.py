from test.fixtures import simple_mesh, simple_V, simple_bulk_temp_parameters as params
from snowbird import bulk_temperature


def test_create_bulk_temperature(params, simple_mesh, simple_V):
    dt = 0.01
    bt = bulk_temperature.BulkTemperature(V=simple_V, mesh=simple_mesh, dt=dt, params=params)
    assert isinstance(bt, bulk_temperature.BulkTemperature)


def test_simulate_bulk_temperature(params, simple_mesh, simple_V):
    t = 0
    dt = 0.01
    num_steps = 10

    bt = bulk_temperature.BulkTemperature(V=simple_V, mesh=simple_mesh, dt=dt, params=params)
    bt.initialize(time=t)

    for n in range(num_steps):
        bt.step(time=t, dt=dt)

