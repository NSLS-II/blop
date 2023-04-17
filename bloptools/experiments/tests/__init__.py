import bluesky.plan_stubs as bps
import numpy as np
from ophyd import Signal


def get_dofs(n=2):
    return [Signal(name=f"x{i+1}", value=0) for i in range(n)]


class Ackley:
    """
    Ackley's function in $n$ dimensions (https://en.wikipedia.org/wiki/Ackley_function)
    """

    MIN_SNR = 1e1

    def __init__(self, n_dof=2):
        self.n_dof = n_dof
        self.dofs = [getattr(self.dofs_256, f"x{i+1}") for i in range(self.n_dof)]
        self.bounds = np.array([[0.0, 1.0] for i in range(self.n_dof)])

        self.DEPENDENT_COMPONENTS = [f"x{i+1}" for i in range(self.n_dof)]

    def initialize(self):
        yield from bps.null()  # do nothing

    def parse_entry(self, entry):
        # get the ingredient from our dependent variables
        x = 4 * (np.c_[[getattr(entry, f"dofs_x{i+1}") for i in range(self.n_dof)]] - 0.5)

        fitness = np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) + 5e-2 * np.exp(0.5 * np.sum(np.cos(2 * np.pi * x)))

        return ("fitness"), (fitness)


class Himmelblau:
    """
    Himmelblau's function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
    """

    MIN_SNR = 1e1

    def __init__(self):
        self.n_dof = 2
        self.dofs = get_dofs(n=2)
        self.bounds = np.array([[0.0, 1.0] for i in range(self.n_dof)])

        self.DEPENDENT_COMPONENTS = [f"x{i+1}" for i in range(self.n_dof)]

    def initialize(self):
        yield from bps.null()  # do nothing

    def parse_entry(self, entry):
        # get the ingredient from our dependent variables
        x = 10 * (np.array([getattr(entry, f"x{i+1}") for i in range(self.n_dof)]) - 0.5)

        fitness = -1e-2 * ((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)

        return ("fitness"), (fitness)
