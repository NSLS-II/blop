import numpy as np
import pandas as pd
import torch


def approximate_erf(x):
    """
    An approximation of erf(x), to compute the definite integral of the Gaussian PDF
    This is faster and better-conditioned near +/- infinity
    """
    return torch.tanh(1.20278247 * x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


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


def binh_korn(x1, x2):
    """
    Binh and Korn function
    """
    f1 = 4 * x1**2 + 4 * x2**2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    g1 = (x1 - 5) ** 2 + x2**2 <= 25
    g2 = (x1 - 8) ** 2 + (x2 + 3) ** 2 >= 7.7

    c = g1 & g2

    return np.where(c, f1, np.nan), np.where(c, f2, np.nan)


def binh_korn_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """

    for index, entry in df.iterrows():
        f1, f2 = binh_korn(entry.x1, entry.x2)
        df.loc[index, "f1"] = f1
        df.loc[index, "f2"] = f2

    return df


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
    Simulating a misaligned Gaussian beam. The optimum is at (1, 1)
    """
    return np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)


def hartmann4(*x):
    X = np.c_[x]

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([[10, 3, 17, 3.5], [0.05, 10, 17, 0.1], [3, 3.5, 1.7, 10], [17, 8, 0.05, 10]])

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124],
            [2329, 4135, 8307, 3736],
            [2348, 1451, 3522, 2883],
            [4047, 8828, 8732, 5743],
        ]
    )

    return -(alpha * np.exp(-(A * np.square(X - P)).sum(axis=1))).sum()


def hartmann6(*x):
    X = np.c_[x]

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array(
        [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
    )

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    return -(alpha * np.exp(-(A * np.square(X - P)).sum(axis=1))).sum()


def kb_tradeoff_2d(x1, x2):
    width = np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)
    d = np.sqrt(x1**2 + x2**2)
    flux = np.exp(-0.5 * np.where(d < 5, np.where(d > -5, 0, d + 5), d - 5) ** 2)

    return width, flux


def kb_tradeoff_4d(x1, x2, x3, x4):
    x_width = np.sqrt(1 + 0.25 * (x1 - x2) ** 2 + 16 * (x1 + x2 - 2) ** 2)
    y_width = np.sqrt(1 + 0.25 * (x3 - x4) ** 2 + 16 * (x3 + x4 - 2) ** 2)
    d = np.sqrt(x1**2 + x2**2 + x3**2 + x4**2)
    flux = np.exp(-0.5 * np.where(d < 5, np.where(d > -5, 0, d + 5), d - 5) ** 2)

    return x_width, y_width, flux


def himmelblau_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """
    for index, entry in df.iterrows():
        if not hasattr(entry, "x1"):
            df.loc[index, "x1"] = x1 = 0
        else:
            x1 = entry.x1
        if not hasattr(entry, "x2"):
            df.loc[index, "x2"] = x2 = 0
        else:
            x2 = entry.x2
        df.loc[index, "himmelblau"] = himmelblau(x1=x1, x2=x2)
        df.loc[index, "himmelblau_transpose"] = himmelblau(x1=x2, x2=x1)

    return df


def constrained_himmelblau_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """

    df = himmelblau_digestion(df)
    df.loc[:, "himmelblau"] = np.where(df.x1.values**2 + df.x1.values**2 < 36, df.himmelblau.values, np.nan)

    return df


"""
Chankong and Haimes function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def chankong_and_haimes_digestion(df):
    for index, entry in df.iterrows():
        df.loc[index, "f1"] = (entry.x1 - 2) ** 2 + (entry.x2 - 1) + 2
        df.loc[index, "f2"] = 9 * entry.x1 - (entry.x2 - 1) + 2
        df.loc[index, "c1"] = entry.x1**2 + entry.x2**2
        df.loc[index, "c2"] = entry.x1 - 3 * entry.x2 + 10

    return df


def mock_kbs_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests a beam waist and height into the feedback.
    """

    for index, entry in df.iterrows():
        sigma_x = gaussian_beam_waist(entry.x1, entry.x2)
        sigma_y = gaussian_beam_waist(entry.x3, entry.x4)

        df.loc[index, "x_width"] = 2 * sigma_x
        df.loc[index, "y_width"] = 2 * sigma_y

    return df
