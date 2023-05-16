import bluesky.plan_stubs as bps
import bluesky.plans as bp
import numpy as np

from .. import BaseTask
from . import get_dofs

dofs = get_dofs(n=4)
bounds = np.array([[-5.0, +5.0], [-5.0, +5.0], [-5.0, +5.0], [-5.0, +5.0]])


class MinBeamWidth(BaseTask):
    name = "min_beam_width"

    def get_fitness(processed_entry):
        return -np.log(1 + 1e-1 * getattr(processed_entry, "x_width"))


class MinBeamHeight(BaseTask):
    name = "min_beam_height"

    def get_fitness(processed_entry):
        return -np.log(1 + 1e-1 * getattr(processed_entry, "y_width"))


def initialize():
    yield from bps.null()  # do nothing


def acquisition(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


def digestion(db, uid):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
    """

    table = db[uid].table()
    products = {"x_width": [], "y_width": []}

    for index, entry in table.iterrows():
        sigma_x = np.sqrt(1 + 0.25 * (entry.x1 - entry.x2) ** 2 + 16 * (entry.x1 + entry.x2 - 2) ** 2)
        sigma_y = np.sqrt(1 + 0.25 * (entry.x3 - entry.x4) ** 2 + 16 * (entry.x3 + entry.x4 - 2) ** 2)

        products["x_width"].append(2 * sigma_x)
        products["y_width"].append(2 * sigma_y)

    return products
