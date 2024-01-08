import numpy as np


def w8_digestion(db, uid):
    return image_digestion(db, uid, image_name="w8")


def w9_digestion(db, uid):
    return image_digestion(db, uid, image_name="w9")


def image_digestion(db, uid, image_name):
    products = db[uid].table(fill=True)

    for index, entry in products.iterrows():
        image = getattr(entry, f"{image_name}_image")
        horizontal_extent = getattr(entry, f"{image_name}_horizontal_extent")
        vertical_extent = getattr(entry, f"{image_name}_vertical_extent")

        flux = image.sum()
        n_y, n_x = image.shape

        # reject if there is no flux, or we can't estimate the position and size of the beam for some reason
        bad = False
        bad |= not (flux > 0)
        if not bad:
            X, Y = np.meshgrid(np.linspace(*horizontal_extent, n_x), np.linspace(*vertical_extent, n_y))

            mean_x = np.sum(X * image) / np.sum(image)
            mean_y = np.sum(Y * image) / np.sum(image)

            sigma_x = np.sqrt(np.sum((X - mean_x) ** 2 * image) / np.sum(image))
            sigma_y = np.sqrt(np.sum((Y - mean_y) ** 2 * image) / np.sum(image))

            bad |= any(np.isnan([mean_x, mean_y, sigma_x, sigma_y]))

        if not bad:
            products.loc[index, ["flux", "x_pos", "y_pos", "x_width", "y_width"]] = (
                flux,
                mean_x,
                mean_y,
                2 * sigma_x,
                2 * sigma_y,
            )
        else:
            for col in ["flux", "x_pos", "y_pos", "x_width", "y_width"]:
                products.loc[index, col] = np.nan

    return products
