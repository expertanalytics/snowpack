from typing import Optional, Tuple, Any, List
from fenics import FunctionSpace
from fenics import TrialFunction
from fenics import TestFunction
from fenics import Function
from fenics import Constant
from fenics import DirichletBC
from fenics import Mesh
from fenics import UnitIntervalMesh
from fenics import dot, grad, dx, rhs, lhs, near, solve
from ufl.form import Form
import matplotlib.pylab as plt
import numpy as np


class BulkTemperatureParameters:

    def __init__(self, V: FunctionSpace, mesh: Mesh, dt: float) -> None:

        self.P_s = Constant(1.0)
        self.c_s = Constant(1.0)
        self.k_e = Constant(1.0)
        self.Q_pc = Constant(1.0)
        self.Q_sw = Constant(1.0)
        self.Q_mm = Constant(1.0)


class BulkTemperature:

    def __init__(self, *, V: FunctionSpace, mesh: Mesh, dt: float, params: BulkTemperatureParameters) -> None:
        self.params = params

        self.V = V
        self.t = Function(V)
        self.T = TrialFunction(V)
        self.v = TestFunction(V)
        self.t_prev = Function(V)
        self.dt = Constant(dt)

        self.a, self.l = self.construct_variation_problem()

    def construct_variation_problem(self) -> Tuple[Form, Form]:
        p = self.params
        u_t = (self.T - self.t_prev) / self.dt
        f = p.Q_pc + p.Q_sw + p.Q_mm
        F = p.P_s * p.c_s * u_t * self.v * dx + p.k_e * dot(grad(self.T), grad(self.v)) * dx - f * self.v * dx
        return lhs(F), rhs(F)

    def initialize(self, *, time: float):
        self.initialize_random(time=time)

    def initialize_random(self, *, time: float) -> None:
        values = np.random.uniform(0, -10, self.t_prev.vector().size())
        values[0] = self.ground_value(time=time)
        values[-1] = self.surface_value(time=time)
        self.t_prev.vector().set_local(values)

    @staticmethod
    def surface_boundary(x, on_boundary: bool) -> bool:
        return on_boundary and x[0] > 0

    @staticmethod
    def ground_boundary(x, on_boundary: bool) -> bool:
        return on_boundary and near(x[0], 0.0)

    def surface_value(self, *, time) -> float:
        return -10

    def ground_value(self, *, time) -> float:
        return 0

    def make_bcs(self, *, time) -> Tuple[DirichletBC, DirichletBC]:
        bc1 = DirichletBC(self.V, Constant(self.ground_value(time=time)), self.ground_boundary)
        bc2 = DirichletBC(self.V, Constant(self.surface_value(time=time)), self.surface_boundary)
        return bc1, bc2

    def step(self, *, time: float, dt: float):

        time += dt
        bc = self.make_bcs(time=time)
        solve(self.a == self.l, self.t, bc)
        self.t_prev.assign(self.t)



