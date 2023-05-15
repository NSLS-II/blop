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

    products = {}

    if np.isin(["x1", "x2"], entry.index).all():
        sigma_x = np.sqrt(1 + 0.25 * (entry.x1 - entry.x2) ** 2 + 16 * (entry.x1 + entry.x2 - 2) ** 2)
        products["x_width"] = 2 * sigma_x

    if np.isin(["x3", "x4"], entry.index).all():
        sigma_y = np.sqrt(1 + 0.25 * (entry.x3 - entry.x4) ** 2 + 16 * (entry.x3 + entry.x4 - 2) ** 2)
        products["y_width"] = 2 * sigma_y

    return products
