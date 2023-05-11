import bluesky.plan_stubs as bps
import numpy as np

from .. import BaseTask
from . import get_dofs

dofs = get_dofs(n=2)
bounds = np.array([[-8.0, +8.0], [-8.0, +8.0]])


class MinHimmelblau(BaseTask):
    name = "minimize_himmelblau"

    def get_fitness(processed_entry):
        return -np.log(1 + 1e-1 * getattr(processed_entry, "himmelblau"))


def initialize():
    yield from bps.null()  # do nothing


def postprocess(entry):
    """
    Himmelblau's function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
    """
    X = np.array([getattr(entry, dof.name) for dof in dofs])

    if np.sqrt(np.square(X).sum()) > 5 * np.sqrt(2):
        return {"himmelblau": np.nan}

    himmelblau = (X[0] ** 2 + X[1] - 11) ** 2 + (X[0] + X[1] ** 2 - 7) ** 2
    return {"himmelblau": himmelblau}
