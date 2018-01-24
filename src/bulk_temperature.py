from fenics import *
from snow_mesh import simple_mesh
import matplotlib.pylab as plt


def simple_setup():
    V = FunctionSpace(simple_mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    T_prev = Function(V)
    P_s = Constant(1.0)
    c_s = Constant(1.0)
    k_e = Constant(1.0)
    Q_pc = Constant(1.0)
    Q_sw = Constant(1.0)
    Q_mm = Constant(1.0)
    dt = Constant(1.0)
    return V, u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt


def construct_variation_problem(u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt):
    u_t = (u - T_prev)/dt
    f = Q_pc + Q_sw + Q_mm
    F = P_s*c_s*u_t*v*dx + k_e*dot(grad(u), grad(v))*dx - f*v*dx
    return lhs(F), rhs(F)


def surface_boundary(x, on_boundary):
        return on_boundary and x[0] > 0


def ground_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0.0)


def make_bcs(V, surface_val, ground_val):
    bc1 = DirichletBC(V, Constant(-ground_val), ground_boundary)
    bc2 = DirichletBC(V, Constant(surface_val), surface_boundary)
    return [bc1, bc2]


def step(t, dt, T_prev, T, a, L, bc):
        t += dt
        solve(a == L, T, bc)
        T_prev.assign(T)
        return t, T_prev, T


def main():
    V, u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt = simple_setup()
    a, L = construct_variation_problem(u, v, T_prev, P_s, c_s, k_e, Q_pc, Q_sw, Q_mm, dt)
    T = Function(V)
    num_steps = 10
    t = 0
    sols = []
    for n in range(num_steps):
        bc = make_bcs(V, 0, -10)
        t, T_prev, T = step(t, dt, T_prev, T, a, L, bc)
        sols.append(T.vector().array())
    for s in sols:
        plt.plot(s)

    plt.show()
if __name__ == "__main__":
    main()
