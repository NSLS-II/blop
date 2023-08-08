import numpy as np


def booth(x1, x2):
    """
    The Booth function (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def matyas(x1, x2):
    """
    The Matyas function (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def himmelblau(x1, x2):
    """
    Himmelblau's function (https://en.wikipedia.org/wiki/Himmelblau%27s_function)
    """
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def constrained_himmelblau(x1, x2):
    """
    Himmelblau's function, returns NaN outside the constraint
    """
    return np.where(x1**2 + x2**2 < 50, himmelblau(x1, x2), np.nan)


def skewed_himmelblau(x1, x2):
    """
    Himmelblau's function, with skewed coordinates
    """
    _x1 = 2 * x1 + x2
    _x2 = 0.5 * (x1 - 2 * x2)

    return constrained_himmelblau(_x1, _x2)


def bukin(x1, x2):
    """
    Bukin function N.6 (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """
    return 100 * np.sqrt(np.abs(x2 - 1e-2 * x1**2)) + 0.01 * np.abs(x1)


def rastrigin(*x):
    """
    The Rastrigin function in arbitrary dimensions (https://en.wikipedia.org/wiki/Rastrigin_function)
    """
    X = np.c_[x]
    return 10 * X.shape[-1] + (X**2 - 10 * np.cos(2 * np.pi * X)).sum(axis=1)


def styblinski_tang(*x):
    """
    Styblinski-Tang function in arbitrary dimensions (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    """
    X = np.c_[x]
    return 0.5 * (X**4 - 16 * X**2 + 5 * X).sum(axis=1)


def ackley(*x):
    """
    The Ackley function in arbitrary dimensions (https://en.wikipedia.org/wiki/Ackley_function)
    """
    X = np.c_[x]
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2).sum(axis=1)))
        - np.exp(0.5 * np.cos(2 * np.pi * X).sum(axis=1))
        + np.e
        + 20
    )


def gaussian_beam_waist(x1, x2):
    """
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1, 1, 1)
    """
    return np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)


def himmelblau_digestion(db, uid):
    """
    Digests Himmelblau's function into the feedback.
    """
    products = db[uid].table()

    for index, entry in products.iterrows():
        products.loc[index, "himmelblau"] = himmelblau(entry.x1, entry.x2)

    return products


def mock_kbs_digestion(db, uid):
    """
    Digests a beam waist and height into the feedback.
    """

    products = db[uid].table()

    for index, entry in products.iterrows():
        sigma_x = gaussian_beam_waist(entry.x1, entry.x2)
        sigma_y = gaussian_beam_waist(entry.x3, entry.x4)

        products.loc[index, "x_width"] = 2 * sigma_x
        products.loc[index, "y_width"] = 2 * sigma_y

    return products
