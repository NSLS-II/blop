import bluesky.plan_stubs as bps
import numpy as np

from ... import utils

DEPENDENT_COMPONENTS = ["sample"]

IMAGE_NAME = "sample_image"

HORIZONTAL_EXTENT_NAME = "sample_horizontal_extent"
VERTICAL_EXTENT_NAME = "sample_vertical_extent"

PCA_BEAM_PROP = 0.5  # how much of the first principle component we want to have in our bounding box
MIN_SEPARABILITY = 0.1  # the minimal variance proportion of the first SVD mode of the beam image

MIN_SNR = 1e1


def initialize():
    yield from bps.null()  # do nothing


def parse_entry(entry):
    # get the ingredient from our dependent variables
    image = getattr(entry, IMAGE_NAME)
    horizontal_extent = getattr(entry, HORIZONTAL_EXTENT_NAME)
    vertical_extent = getattr(entry, VERTICAL_EXTENT_NAME)

    if not image.sum() > 0:
        image = np.random.uniform(size=image.shape)
        horizontal_extent = [np.nan, np.nan]
        vertical_extent = [np.nan, np.nan]

    pix_x_min, pix_x_max, pix_y_min, pix_y_max, separability = utils.get_principal_component_bounds(
        image, PCA_BEAM_PROP
    )
    n_y, n_x = image.shape
    x_min, x_max = np.interp([pix_x_min, pix_x_max], [0, n_x + 1], horizontal_extent)
    y_min, y_max = np.interp([pix_y_min, pix_y_max], [0, n_y + 1], vertical_extent)

    width_x = x_max - x_min
    width_y = y_max - y_min

    pos_x = (x_min + x_max) / 2
    pos_y = (y_min + y_max) / 2

    flux = image.sum()

    bad = False
    bad |= separability < MIN_SEPARABILITY

    if bad:
        fitness = np.nan

    else:
        fitness = np.log(flux * separability / (width_x**2 + width_y**2))

    return ("fitness", "flux", "pos_x", "pos_y", "width_x", "width_y", "separability"), (
        fitness,
        flux,
        pos_x,
        pos_y,
        width_x,
        width_y,
        separability,
    )
