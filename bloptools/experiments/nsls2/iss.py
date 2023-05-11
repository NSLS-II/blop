import bluesky.plan_stubs as bps
import numpy as np

from .. import BaseTask

def initialize():
    yield from bps.null()  # do nothing


class MaxBeamFlux(BaseTask):
    name = "max_flux"

    def get_fitness(processed_entry):
        return getattr(processed_entry, "flux")


class MinJohannResolution(BaseTask):
    name = "min_johann_resolution"

    def get_fitness(processed_entry):
        return -np.log(getattr(processed_entry, "johann_fwhm"))


def postprocess(entry):
    """
    This method eats the output of a Bluesky scan, and returns a dict with inputs for the tasks
    """

    # get the ingredient from our dependent variables

    flux = -getattr(entry, "apb_ch4") # lower number is more flux :(

    if flux < 100:
        return {
        "flux": np.nan,
    }

    return {
        "flux": flux,
    }
