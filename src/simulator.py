import numpy as np
import matplotlib.pylab as plt
from fenics import FunctionSpace, Function
from snow_mesh import simple_mesh
import mass_air_phase 
import bulk_temperature


def solve():
    mesh = simple_mesh
    V = FunctionSpace(mesh, "Lagrange", 1)
    dt_val = 0.01
    _, u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt = bulk_temperature.simple_setup(mesh, dt_val, V)
    a_T, L_T = bulk_temperature.construct_variation_problem(u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt)
    T = Function(V)
    T_prev.vector().set_local(np.random.uniform(0, -10, T_prev.vector().size()))
    _, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s = mass_air_phase.simple_setup(mesh, dt_val, V)
    a_P, L_P = mass_air_phase.construct_variation_problem(mesh, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T)
    P = Function(V)
    num_steps = 10
    t = 0
    T_sols = [T_prev.vector().array()]
    P_sols = []
    #T.vector().array()
    for n in range(num_steps):
        bc = bulk_temperature.make_bcs(V, 0, -10)
        _, T_prev, T = bulk_temperature.step(t, dt, T_prev, T, a_T, L_T, bc)
        _, P_prev, P = mass_air_phase.step(t, dt, P_prev, P, a_P, L_P)
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
