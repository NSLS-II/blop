import bluesky.plan_stubs as bps
import bluesky.plans as bp
import numpy as np

from bloptools.tasks import Task


def initialize():
    yield from bps.null()  # do nothing


class MaxBeamFlux(Task):
    name = "max_flux"

    def get_fitness(processed_entry):
        return getattr(processed_entry, "flux")


def acquisition(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


def flux_digestion(db, uid):
    """
    This method eats the output of a Bluesky scan, and returns a dict with inputs for the tasks
    """

    table = db[uid].table(fill=True)

    products_keys = [
        "image",
        "vertical_extent",
        "horizontal_extent",
        "flux",
        "x_pos",
        "y_pos",
        "x_width",
        "y_width",
    ]
    products = {key: [] for key in products_keys}

    for index, entry in table.iterrows():
        products["apb_ch4"].append(getattr(entry, "apb_ch4"))

    return products
