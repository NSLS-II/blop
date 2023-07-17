import numpy as np


def booth(x1, x2):
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def matyas(x1, x2):
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def himmelblau(x1, x2):
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def constrained_himmelblau(x1, x2):
    if x1**2 + x2**2 > 50:
        return np.nan
    return himmelblau(x1, x2)


def skewed_himmelblau(x1, x2):
    _x1 = 2 * x1 + x2
    _x2 = 0.5 * (x1 - 2 * x2)

    return constrained_himmelblau(_x1, _x2)


def rastrigin(*x):
    X = np.c_[x]
    return 10 * X.shape[-1] + (X**2 - 10 * np.cos(2 * np.pi * X)).sum(axis=1)


def gaussian_beam_waist(x1, x2):
    return np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)


def himmelblau_digestion(db, uid):
    products = db[uid].table()

    for index, entry in products.iterrows():
        products.loc[index, "himmelblau"] = himmelblau(entry.x1, entry.x2)

    return products


def mock_kbs_digestion(db, uid):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
    """

    products = db[uid].table()

    for index, entry in products.iterrows():
        sigma_x = gaussian_beam_waist(entry.x1, entry.x2)
        sigma_y = gaussian_beam_waist(entry.x3, entry.x4)

        products.loc[index, "x_width"] = 2 * sigma_x
        products.loc[index, "y_width"] = 2 * sigma_y

    return products
