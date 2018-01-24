from __future__ import print_function
from fenics import *
from snow_mesh import simple_mesh
import matplotlib.pylab as plt


def simple_setup():
    V = FunctionSpace(simple_mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    P_prev = Function(V)
    theta_a = Constant(1.0)
    D_e = Constant(1.0)
    P_d = Constant(1.0)
    my_v = Constant(1.0)
    my_d = Constant(1.0)
    alpha_th = Constant(1.0)
    M_mm = Constant(1.0)
    q_h = Constant(1.0)
    T_s = project(Constant(-5.0), V)
    dt = Constant(1.0)
    return V, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s


def make_ds(mesh):
    Surface = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 1))
    Allboundaries = DomainBoundary()
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    Surface.mark(boundaries, 1)
    ds = Measure("ds", subdomain_data = boundaries)
    return ds

def construct_variation_problem(u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s):
    ds = make_ds(simple_mesh)
    u_t = (u - P_prev)/dt
    term = alpha_th*u*P_d*(my_v + my_d)**2/((P_prev + P_d)*my_v*my_d)/T_s*grad(T_s)
    F = theta_a*u_t*v*dx + theta_a*D_e*dot(grad(u) - term, grad(v))*dx - M_mm*v*dx + q_h*v*ds(1)
    return lhs(F), rhs(F)


def surface_boundary(x, on_boundary):
        return on_boundary and x[0] > 0


def step(t, dt, P_prev, P, a, L):
        t += dt
        solve(a == L, P)
        P_prev.assign(P)
        return t, P_prev, P


def main():
    V, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s = simple_setup()
    a, L = construct_variation_problem(u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s)
    P = Function(V)
    num_steps = 10
    t = 0
    sols = []
    for n in range(num_steps):
        t, P_prev, P = step(t, dt, P_prev, P, a, L)
        sols.append(P.vector().array())
    for s in sols:
        print(s)
        plt.plot(s)

    plt.show()
if __name__ == "__main__":
    main()
