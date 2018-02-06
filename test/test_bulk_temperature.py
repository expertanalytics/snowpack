from test.fixtures import simple_mesh as mesh, simple_bulk_temp_parameters as params
from snowbird import bulk_temperature
import fenics


def test_create_bulk_temperature_parameters(mesh):
    params = bulk_temperature.BulkTemperatureParameters(V=None, mesh=mesh, dt=0.01)
    assert isinstance(params.V, fenics.FunctionSpace)


def test_create_bulk_temperature(params):
    bt = bulk_temperature.BulkTemperature(params=params)
    assert isinstance(bt, bulk_temperature.BulkTemperature)


def test_simulate_bulk_temperature(params):
    bt = bulk_temperature.BulkTemperature(params=params)
    V = params.V
    dt = params.dt
    T_prev = params.T_prev
    a, L = bt.construct_variation_problem()

    T = fenics.Function(V)
    num_steps = 10
    t = 0
    for n in range(num_steps):
        bc = bt.make_bcs(V, 0, -10)
        t, T_prev, T = bulk_temperature.step(time=t, dt=dt, t_prev=T_prev, t=T, a=a, l=L, bc=bc)
