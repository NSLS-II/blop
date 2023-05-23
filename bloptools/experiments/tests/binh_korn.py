import bluesky.plan_stubs as bps
import numpy as np

from .. import BaseTask
from . import get_dofs

dofs = get_dofs(n=2)
bounds = np.array([[0, +5.0], [0, +3.0]])


class MinBinhKorn1(BaseTask):
    name = "min_f1"

    def get_fitness(processed_entry):
        return -getattr(processed_entry, "f1")


class MinBinhKorn2(BaseTask):
    name = "min_f2"

    def get_fitness(processed_entry):
        return -getattr(processed_entry, "f2")


def initialize():
    yield from bps.null()  # do nothing


def postprocess(entry):
    """
    Binh and Korn function (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """

    X = np.array([getattr(entry, dof.name) for dof in dofs])

    g1 = (X[0] - 5) ** 2 + X[1] ** 2
    g2 = (X[0] - 8) ** 2 + (X[1] + 3) ** 2

    if (g1 > 25) or (g2 < 7.7):
        return {"f1": np.nan, "f2": np.nan}

    f1 = 4 * (X[0] ** 2 + X[1] ** 2)
    f2 = (X[0] - 5) ** 2 + (X[1] - 5) ** 2

    return {"f1": f1, "f2": f2}
