import bluesky.plan_stubs as bps
import numpy as np
from ophyd import Component as Cpt
from ophyd import Device, Signal


def get_dofs(n=2):
    return [Signal(name=f"x{i+1}", value=0) for i in range(n)]


def get_device(name="dofs", n=2):
    components = {}

    for i in range(n):
        components[f"x{i+1}"] = Cpt(Signal, value=i + 1)

    cls = type("DOF", (Device,), components)

    device = cls(name=name)

    return [getattr(device, attr) for attr in device.read_attrs]


class BaseOptimizationTest:
    MIN_SNR = 1e2

    n_dof = 2
    dofs = get_dofs(n=2)
    bounds = np.array([[-5.0, +5.0], [-5.0, +5.0]])

    DEPENDENT_COMPONENTS = [dof.name for dof in dofs]

    def initialize(self):
        yield from bps.null()  # do nothing

    def parse_entry(self, entry):
        X = np.array([getattr(entry, cpt) for cpt in self.DEPENDENT_COMPONENTS])
        fitness = self.fitness_func(X)
        return ("fitness"), (fitness)


class Ackley(BaseOptimizationTest):
    """
    Ackley's function in $n$ dimensions (https://en.wikipedia.org/wiki/Ackley_function)
    """

    def __init__(self, n_dof=2):
        self.n_dof = n_dof
        self.dofs = get_dofs(n=n_dof)
        self.bounds = np.array([[-5.0, +5.0] for i in range(self.n_dof)])
        self.DEPENDENT_COMPONENTS = [dof.name for dof in self.dofs]

    @staticmethod
    def fitness_func(X):
        return (
            20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(X**2)))
            + 5e-2 * np.exp(0.5 * np.sum(np.cos(2 * np.pi * X)))
            - 20
            - np.e
        )


class Himmelblau(BaseOptimizationTest):
    """
    Himmelblau's function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
    """

    n_dof = 2
    dofs = get_dofs(n=2)
    bounds = np.array([[-5.0, +5.0], [-5.0, +5.0]])

    DEPENDENT_COMPONENTS = [dof.name for dof in dofs]

    @staticmethod
    def fitness_func(X):
        return -((X[0] ** 2 + X[1] - 11) ** 2 + (X[0] + X[1] ** 2 - 7) ** 2)


class ConstrainedHimmelblau(Himmelblau):
    """
    Himmelblau's function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
    Return NaN if ||X|| > dim(X)
    """

    bounds = np.array([[-10.0, +10.0], [-10.0, +10.0]])

    @staticmethod
    def fitness_func(X):
        if np.sqrt(np.sum(np.square(X))) > 8:
            return np.nan
        return -((X[0] ** 2 + X[1] - 11) ** 2 + (X[0] + X[1] ** 2 - 7) ** 2)


class Rosenbrock(BaseOptimizationTest):
    """
    Rosenbrock function in $n$ dimensions (https://en.wikipedia.org/wiki/Rosenbrock_function)
    """

    def __init__(self, n_dof=2):
        self.n_dof = n_dof
        self.dofs = get_dofs(n=n_dof)
        self.bounds = np.array([[-2.0, +2.0] for i in range(self.n_dof)])
        self.DEPENDENT_COMPONENTS = [dof.name for dof in self.dofs]

    @staticmethod
    def fitness_func(X):
        return -np.log((100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2).sum(axis=-1))


class ConstrainedRosenbrock(Rosenbrock):
    """
    Rosenbrock function in $n$ dimensions (https://en.wikipedia.org/wiki/Rosenbrock_function)
    Return NaN if ||X|| > dim(X)
    """

    @staticmethod
    def fitness_func(X):
        if (X**2).sum(axis=-1) > len(X):
            return np.nan
        return -np.log((100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2).sum(axis=-1))


class StyblinskiTang(BaseOptimizationTest):
    """
    Styblinski-Tang function in $n$ dimensions (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """

    def __init__(self, n_dof=2):
        self.n_dof = n_dof
        self.dofs = get_dofs(n=n_dof)
        self.bounds = np.array([[-5.0, +5.0] for i in range(self.n_dof)])
        self.DEPENDENT_COMPONENTS = [dof.name for dof in self.dofs]

    @staticmethod
    def fitness_func(X):
        return -(X**4 - 16 * X**2 + 5 * X).sum(axis=-1)
