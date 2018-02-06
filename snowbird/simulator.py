import numpy as np
import matplotlib.pylab as plt
from fenics import FunctionSpace, Function, UnitIntervalMesh
from snowbird import mass_air_phase
from snowbird import bulk_temperature


def solve():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "Lagrange", 1)
    dt = 0.01
    bt_params = bulk_temperature.BulkTemperatureParameters(V=V, mesh=mesh, dt=dt)
    bt = bulk_temperature.BulkTemperature(params=bt_params)
    a_T, L_T = bt.construct_variation_problem()
    T = Function(V)
    bt_params.T_prev.vector().set_local(np.random.uniform(0, -10, bt_params.T_prev.vector().size()))
    map_params = mass_air_phase.MassAirPhaseParameters(V=V, mesh=mesh, dt=dt)
    map_params.T_s = T
    ma_ph = mass_air_phase.MassAirPhase(params=map_params)
    a_P, L_P = ma_ph.construct_variation_problem()
    P = Function(V)
    num_steps = 10
    t = 0
    T_sols = [bt_params.T_prev.vector().array()]
    P_sols = []
    for n in range(num_steps):
        bc = bt.make_bcs(V, 0, -10)
        _, bt_params.T_prev, T = bulk_temperature.step(time=t, dt=dt, t_prev=bt_params.T_prev, t=T, a=a_T, l=L_T, bc=bc)
        _, map_params.P_prev, P = mass_air_phase.step(time=t, dt=dt, p_prev=map_params.P_prev, p=P, a=a_P, l=L_P)
        t += dt
        T_sols.append(T.vector().array())
        P_sols.append(P.vector().array())
    for T_s in T_sols:
        plt.plot(T_s)
    plt.figure()
    for P_s in P_sols:
        plt.plot(P_s)

    plt.show()

if __name__ == "__main__":
    solve()
