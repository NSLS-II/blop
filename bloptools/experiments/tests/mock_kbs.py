import bluesky.plan_stubs as bps
import numpy as np

from .. import BaseTask
from . import get_dofs

dofs = get_dofs(n=4)
bounds = np.array([[-5.0, +5.0], [-5.0, +5.0], [-5.0, +5.0], [-5.0, +5.0]])


class MinBeamWidth(BaseTask):
    name = "min_beam_width"

    def get_fitness(processed_entry):
        return -np.log(getattr(processed_entry, "x_width"))


class MinBeamHeight(BaseTask):
    name = "min_beam_height"

    def get_fitness(processed_entry):
        return -np.log(getattr(processed_entry, "y_width"))


def initialize():
    yield from bps.null()  # do nothing


def postprocess(entry):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
    """

    X = np.array([getattr(entry, dof.name) for dof in dofs])

    sigma_x = np.sqrt(1 + 0.25 * (X[1] - X[0]) ** 2 + 16 * (X[0] + X[1] - 2) ** 2)
    sigma_y = np.sqrt(1 + 0.25 * (X[2] - X[3]) ** 2 + 16 * (X[2] + X[3] - 2) ** 2)

    return {"x_width": 2 * sigma_x, "y_width": 2 * sigma_y}
