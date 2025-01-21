import numpy as np
import pandas as pd

from ..utils import functions


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
        df.loc[index, "himmelblau"] = functions.himmelblau(x1=x1, x2=x2)
        df.loc[index, "himmelblau_transpose"] = functions.himmelblau(x1=x2, x2=x1)

    return df


def constrained_himmelblau_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback, constrained with NaN for a distance of more than 6 from the origin.
    """

    df = himmelblau_digestion(df)
    df.loc[:, "himmelblau"] = np.where(df.x1.values**2 + df.x1.values**2 < 36, df.himmelblau.values, np.nan)

    return df


def sketchy_himmelblau_digestion(df: pd.DataFrame, p=0.1) -> pd.DataFrame:
    """
    Evaluates the constrained Himmelblau, where every point is bad with probability p.
    """

    df = constrained_himmelblau_digestion(df)
    bad = np.random.choice(a=[True, False], size=len(df), p=[p, 1 - p])
    df.loc[:, "himmelblau"] = np.where(bad, np.nan, df.himmelblau.values)

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
        sigma_x = functions.gaussian_beam_waist(entry.x1, entry.x2)
        sigma_y = functions.gaussian_beam_waist(entry.x3, entry.x4)

        df.loc[index, "x_width"] = 2 * sigma_x
        df.loc[index, "y_width"] = 2 * sigma_y

    return df


def binh_korn_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """

    for index, entry in df.iterrows():
        f1, f2 = functions.binh_korn(entry.x1, entry.x2)
        df.loc[index, "f1"] = f1
        df.loc[index, "f2"] = f2

    return df
