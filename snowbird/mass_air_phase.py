from typing import Optional, Tuple
from fenics import FunctionSpace
from fenics import TrialFunction
from fenics import TestFunction
from fenics import Function
from fenics import Constant
from fenics import AutoSubDomain
from fenics import FacetFunction
from fenics import Measure
from fenics import Mesh
from fenics import UnitIntervalMesh
from fenics import dot, grad, dx, rhs, lhs, near, solve, project
from ufl.form import Form
import matplotlib.pylab as plt


class MassAirPhaseParameters:

    def __init__(self, *,  V: Optional[FunctionSpace], mesh: Mesh, dt: float):
        V = V or FunctionSpace(mesh, "Lagrange", 1)
        self.V = V
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.P_prev = Function(V)
        self.theta_a = Constant(1.0)
        self.D_e = Constant(1.0)
        self.P_d = Constant(1.0)
        self.my_v = Constant(1.0)
        self.my_d = Constant(1.0)
        self.alpha_th = Constant(1.0)
        self.M_mm = Constant(1.0)
        self.q_h = Constant(1.0)
        self.T_s = project(Constant(-5.0), V)
        self.dt = Constant(dt)
        self.mesh = mesh



class MassAirPhase:

    def __init__(self, *, params: MassAirPhaseParameters):
        self.params = params

    @staticmethod
    def make_ds(mesh):
        surface = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 1))
        boundaries = FacetFunction("size_t", mesh)
        boundaries.set_all(0)
        surface.mark(boundaries, 1)
        ds = Measure("ds", subdomain_data=boundaries)
        return ds

    def construct_variation_problem(self) -> Tuple[Form, Form]:
        p = self.params
        ds = self.make_ds(p.mesh)
        u_t = (p.u - p.P_prev)/p.dt
        term = p.alpha_th*p.u*p.P_d*(p.my_v + p.my_d)**2/((p.P_prev + p.P_d)*p.my_v*p.my_d)/p.T_s*grad(p.T_s)
        F = p.theta_a*u_t*p.v*dx + p.theta_a*p.D_e*dot(grad(p.u) - term, grad(p.v))*dx - p.M_mm*p.v*dx + p.q_h*p.v*ds(1)
        return lhs(F), rhs(F)

    @staticmethod
    def surface_boundary(x, on_boundary):
            return on_boundary and x[0] > 0


def step(*, time: float, dt: float, p_prev: Function, p: Function,
         a: Form, l: Form) -> Tuple[float, Function, Function]:
        time += dt
        solve(a == l, p)
        p_prev.assign(p)
        return time, p_prev, p


def test_solve() -> None:
    dt = 0.01
    mesh = UnitIntervalMesh(10)
    params = MassAirPhaseParameters(V=None, mesh=mesh, dt=dt)
    mass_air_phase = MassAirPhase(params=params)
    a, L = mass_air_phase.construct_variation_problem()
    print(type(a))
    print(type(L))
    P = Function(params.V)
    P_prev = params.P_prev
    num_steps = 10
    t = 0
    sols = []
    for n in range(num_steps):
        t, P_prev, P = step(time=t, dt=dt, p_prev=P_prev, p=P, a=a, l=L)
        sols.append(P.vector().get_local())
    for s in sols:
        plt.plot(s)

    plt.show()


if __name__ == "__main__":
    test_solve()
