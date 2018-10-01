#from dolfin import FunctionSpace
#from dolfin import TrialFunction
#from dolfin import TestFunction
#from dolfin import Function
#from dolfin import Constant
#from dolfin import project
#from dolfin import AutoSubDomain
#from dolfin import Measure
#from dolfin import SubDomain
#from dolfin import MeshFunction
#from dolfin import lhs
#from dolfin import rhs
#from dolfin import solve
#from dolfin import near
##from ufl import dx, dot, grad
from firedrake import *
from snowpack.snow_mesh import simple_mesh
import matplotlib.pylab as plt


def simple_setup(mesh, dt_val, V=None):
    V = V or FunctionSpace(mesh, "Lagrange", 1)
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
    dt = Constant(dt_val)
    return V, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s




def make_ds(mesh):
    #surface = MyFuckingSubDomain()
    #Allboundaries = DomainBoundary()
    boundaries = MeshFunction(mesh=mesh, dim=mesh.geometry().dim() - 1, value_type="size_t")
    boundaries.set_all(0)
    surface.mark(boundaries, 1)
    ds = Measure("ds", subdomain_data=boundaries)
    return ds

def construct_variation_problem(mesh, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s):
    ds = make_ds(mesh)
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
    V, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s = simple_setup(simple_mesh, 0.01)
    a, L = construct_variation_problem(simple_mesh, u, v, dt, P_prev, theta_a, D_e, P_d, my_v, my_d, alpha_th, M_mm, q_h, T_s)
    P = Function(V)
    num_steps = 10
    t = 0
    sols = []
    for i in range(num_steps):
        print(i)
        t, P_prev, P = step(t, dt, P_prev, P, a, L)
        sols.append(P.vector().array())
    for s in sols:
        print(s)
        plt.plot(s)

    plt.show()
if __name__ == "__main__":
    main()
