import time as ttime

import bluesky.plan_stubs as bps
import bluesky.plans as bp  # noqa F401
import numpy as np

from ... import utils

TARGET_BEAM_WIDTH_X = 0
TARGET_BEAM_WIDTH_Y = 0
BEAM_PROP = 0.5
MIN_SEPARABILITY = 0.1
OOB_REL_BUFFER = 1 / 32


def initialize(shutter, detectors):
    timeout = 10

    yield from bps.mv(shutter.close_cmd, 1)
    yield from bps.sleep(2.0)

    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (shutter.status.get(as_string=True) == "Open"):
        print(f"Shutter not closed, retrying ... (closed_status = {shutter.status.get(as_string=True)})")
        yield from bps.sleep(1.0)
        yield from bps.mv(shutter.close_cmd, 1)

    uid = yield from bp.count(detectors)

    yield from bps.mv(shutter.open_cmd, 1)
    yield from bps.sleep(2.0)

    timeout = 10
    start_time = ttime.monotonic()
    while (ttime.monotonic() - start_time < timeout) and (shutter.status.get(as_string=True) != "Open"):
        print(f"Shutter not open, retrying ... (closed_status = {shutter.status.get(as_string=True)})")
        yield from bps.sleep(1.0)
        yield from bps.mv(shutter.open_cmd, 1)

    yield from bps.sleep(10.0)

    return uid


def fitness(entry, args):
    image = getattr(entry, args["image"])
    # if (args['flux'] not in entry.index) or (args['flux'] is not None):
    #    flux = 1e0
    # else:
    flux = getattr(entry, args["flux"])

    background = args["background"]

    x_min, x_max, y_min, y_max, separability = utils.get_beam_stats(
        image - background, beam_prop=args["beam_prop"]
    )

    n_y, n_x = image.shape

    # u, s, v = np.linalg.svd(image - background)

    # separability = np.square(s[0]) / np.square(s).sum()

    # ymode, xmode = u[:,0], v[0]

    # x_roots = utils.estimate_root_indices(np.abs(xmode) - args['beam_prop'] * np.abs(xmode).max())
    # y_roots = utils.estimate_root_indices(np.abs(ymode) - args['beam_prop'] * np.abs(ymode).max())

    # x_min, x_max = x_roots.min(), x_roots.max()
    # y_min, y_max = y_roots.min(), y_roots.max()

    width_x = x_max - x_min
    width_y = y_max - y_min

    bad = x_min < (n_x + 1) * args["OOB_rel_buffer"]
    bad |= x_max > (n_x + 1) * (1 - args["OOB_rel_buffer"])
    bad |= y_min < (n_y + 1) * args["OOB_rel_buffer"]
    bad |= y_max > (n_y + 1) * (1 - args["OOB_rel_buffer"])
    bad |= separability < args["min_separability"]
    bad |= width_x < 2
    bad |= width_y < 2

    if bad:
        fitness = np.nan

    else:
        fitness = np.log(flux * separability / (width_x**2 + width_y**2))

    return ("fitness", "x_min", "x_max", "y_min", "y_max", "width_x", "width_y", "separability"), (
        fitness,
        x_min,
        x_max,
        y_min,
        y_max,
        width_x,
        width_y,
        separability,
    )
