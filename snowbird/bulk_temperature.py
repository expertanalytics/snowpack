from typing import Optional, Tuple, Any, List
from fenics import FunctionSpace
from fenics import TrialFunction
from fenics import TestFunction
from fenics import Function
from fenics import Constant
from fenics import DirichletBC
from fenics import Mesh
from fenics import dot, grad, dx, rhs, lhs, near, solve
from .snow_mesh import simple_mesh
import matplotlib.pylab as plt


class BulkTemperatureParameters:

    def __init__(self, *, V: Optional[FunctionSpace], mesh: Mesh, dt: float) -> None:
        V = V or FunctionSpace(mesh, "Lagrange", 1)
        self.V = V
        self.T = TrialFunction(V)
        self.v = TestFunction(V)
        self.T_prev = Function(V)
        self.P_s = Constant(1.0)
        self.c_s = Constant(1.0)
        self.k_e = Constant(1.0)
        self.Q_pc = Constant(1.0)
        self.Q_sw = Constant(1.0)
        self.Q_mm = Constant(1.0)
        self.dt = Constant(dt)



class BulkTemperature:

    def __init__(self, *, params: BulkTemperatureParameters) -> None:
        self.params = params

    def construct_variation_problem(self) -> Tuple[Any, Any]:
        p = self.params
        u_t = (p.T - p.T_prev)/p.dt
        f = p.Q_pc + p.Q_sw + p.Q_mm
        F = p.P_s*p.c_s*u_t*p.v*dx + p.k_e*dot(grad(p.T), grad(p.v))*dx - f*p.v*dx
        return lhs(F), rhs(F)


    @staticmethod
    def surface_boundary(x, on_boundary: bool) -> bool:
            return on_boundary and x[0] > 0


    @staticmethod
    def ground_boundary(x, on_boundary: bool) -> bool:
            return on_boundary and near(x[0], 0.0)


    @classmethod
    def make_bcs(cls, V, surface_val, ground_val) -> List[DirichletBC]:
        bc1 = DirichletBC(V, Constant(ground_val), cls.ground_boundary)
        bc2 = DirichletBC(V, Constant(surface_val), cls.surface_boundary)
        return [bc1, bc2]


def step(t, dt, T_prev, T, a, L, bc):
        t += dt
        solve(a == L, T, bc)
        T_prev.assign(T)
        return t, T_prev, T


def main():
    p = BulkTemperatureParameters(V=None, mesh=simple_mesh, dt=0.01)
    bulk_temp = BulkTemperature(params=p)
    V = p.V
    dt = p.dt
    T_prev = p.T_prev
    a, L = bulk_temp.construct_variation_problem()

    T = Function(V)
    num_steps = 10
    t = 0
    sols = []
    for n in range(num_steps):
        bc = bulk_temp.make_bcs(V, 0, -10)
        t, T_prev, T = step(t, dt, T_prev, T, a, L, bc)
        sols.append(T.vector().array())
    for s in sols:
        plt.plot(s)

    plt.show()
if __name__ == "__main__":
    main()
