import bluesky.plan_stubs as bps
import numpy as np

from .. import BaseTask

IMAGE_NAME = "w9_image"


def initialize():
    yield from bps.null()  # do nothing


class MinBeamWidth(BaseTask):
    name = "min_beam_width"

    def get_fitness(processed_entry):
        return -np.log(getattr(processed_entry, "x_width"))


class MinBeamHeight(BaseTask):
    name = "min_beam_height"

    def get_fitness(processed_entry):
        return -np.log(getattr(processed_entry, "y_width"))


class MaxBeamFlux(BaseTask):
    name = "max_beam_flux"

    def get_fitness(processed_entry):
        return np.log(getattr(processed_entry, "flux"))


def postprocess(entry):
    """
    This method eats the output of a Bluesky scan, and returns a dict with inputs for the
    """

    # get the ingredient from our dependent variables
    image = getattr(entry, "w9_image")
    horizontal_extent = getattr(entry, "w9_horizontal_extent")
    vertical_extent = getattr(entry, "w9_vertical_extent")

    flux = image.sum()
    n_y, n_x = image.shape

    if not flux > 0:
        image = np.random.uniform(size=image.shape)
        horizontal_extent = [np.nan, np.nan]
        vertical_extent = [np.nan, np.nan]

    X, Y = np.meshgrid(np.linspace(*horizontal_extent, n_x), np.linspace(*vertical_extent, n_y))

    mean_x = np.sum(X * image) / np.sum(image)
    mean_y = np.sum(Y * image) / np.sum(image)

    sigma_x = np.sqrt(np.sum((X - mean_x) ** 2 * image) / np.sum(image))
    sigma_y = np.sqrt(np.sum((Y - mean_y) ** 2 * image) / np.sum(image))

    bad = False
    bad |= ~(flux > 0)
    bad |= np.isnan([mean_x, mean_y, sigma_x, sigma_y]).any()

    if bad:
        return {"flux": np.nan, "x_pos": np.nan, "y_pos": np.nan, "x_width": np.nan, "y_width": np.nan}

    return {
        "flux": flux,
        "x_pos": mean_x,
        "y_pos": mean_y,
        "x_width": 2 * sigma_x,
        "y_width": 2 * sigma_y,
    }
