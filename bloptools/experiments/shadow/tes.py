import bluesky.plans as bp
import numpy as np

from .. import BaseTask


class MinBeamWidth(BaseTask):
    name = "min_beam_width"

    def get_fitness(entry):
        return -np.log(getattr(entry, "x_width"))


class MinBeamHeight(BaseTask):
    name = "min_beam_height"

    def get_fitness(entry):
        return -np.log(getattr(entry, "y_width"))


class MaxBeamFlux(BaseTask):
    name = "max_beam_flux"

    def get_fitness(processed_entry):
        return np.log(getattr(processed_entry, "flux"))


def acquisition(dofs, inputs, dets):
    uid = yield from bp.list_scan(dets, *[_ for items in zip(dofs, np.atleast_2d(inputs).T) for _ in items])
    return uid


def digestion(db, uid, image_name="w9"):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
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
        image = getattr(entry, f"{image_name}_image")
        horizontal_extent = getattr(entry, f"{image_name}_horizontal_extent")
        vertical_extent = getattr(entry, f"{image_name}_vertical_extent")

        products["image"].append(image)
        products["vertical_extent"].append(vertical_extent)
        products["horizontal_extent"].append(horizontal_extent)

        flux = image.sum()
        n_y, n_x = image.shape

        X, Y = np.meshgrid(np.linspace(*horizontal_extent, n_x), np.linspace(*vertical_extent, n_y))

        mean_x = np.sum(X * image) / np.sum(image)
        mean_y = np.sum(Y * image) / np.sum(image)

        sigma_x = np.sqrt(np.sum((X - mean_x) ** 2 * image) / np.sum(image))
        sigma_y = np.sqrt(np.sum((Y - mean_y) ** 2 * image) / np.sum(image))

        # reject if there is no flux, or we can't estimate the position and size of the beam
        bad = False
        bad |= ~(flux > 0)
        bad |= np.isnan([mean_x, mean_y, sigma_x, sigma_y]).any()

        if bad:
            for key in ["flux", "x_pos", "y_pos", "x_width", "y_width"]:
                products[key].append(np.nan)
        else:
            products["flux"].append(flux)
            products["x_pos"].append(mean_x)
            products["y_pos"].append(mean_y)
            products["x_width"].append(2 * sigma_x)
            products["y_width"].append(2 * sigma_y)

    return products
