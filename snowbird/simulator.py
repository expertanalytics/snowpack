import matplotlib.pylab as plt
from fenics import Function, UnitIntervalMesh
from snowbird import mass_air_phase
from snowbird import bulk_temperature


def solve():
    num_steps = 10
    t = 0
    dt = 0.01

    mesh = UnitIntervalMesh(10)


    bt_params = bulk_temperature.BulkTemperatureParameters(V=V, mesh=mesh, dt=dt)
    bt = bulk_temperature.BulkTemperature(V=V, mesh=mesh, dt=dt, params=bt_params)
    bt.initialize(time=t)

    map_params = mass_air_phase.MassAirPhaseParameters(V=V, mesh=mesh, dt=dt)
    map_params.T_s = bt.t
    ma_ph = mass_air_phase.MassAirPhase(params=map_params)
    a_P, L_P = ma_ph.construct_variation_problem()
    P = Function(V)

    T_sols = [bt.t_prev.vector().get_local()]
    P_sols = []
    for n in range(num_steps):

        bt.step(time=t, dt=dt)
        _, map_params.P_prev, P = mass_air_phase.step(time=t, dt=dt, p_prev=map_params.P_prev, p=P, a=a_P, l=L_P)
        t += dt
        T_sols.append(bt.t.vector().get_local())
        P_sols.append(P.vector().get_local())
    for T_s in T_sols:
        plt.plot(T_s)
    plt.figure()
    for P_s in P_sols:
        plt.plot(P_s)

    plt.show()

if __name__ == "__main__":
    solve()
