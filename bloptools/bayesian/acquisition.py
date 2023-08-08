import bluesky.plans as bp
import numpy as np


def default_acquisition_plan(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid
