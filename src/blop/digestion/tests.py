import numpy as np
import pandas as pd

from ..utils import functions


def himmelblau_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """
    df["x1"] = df["x1"].fillna(0) if "x1" in df.columns else 0
    df["x2"] = df["x2"].fillna(0) if "x2" in df.columns else 0
    df["himmelblau"] = functions.himmelblau(x1=df.x1, x2=df.x2)
    df["himmelblau_transpose"] = functions.himmelblau(x1=df.x2, x2=df.x1)
    return df


def constrained_himmelblau_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback, constrained with NaN for a distance of more than 6 from the origin.
    """

    df = himmelblau_digestion(df)
    df.loc[:, "himmelblau"] = np.where(
        np.array(df.x1.values) ** 2 + np.array(df.x2) ** 2 < 36, np.array(df.himmelblau), np.nan
    )

    return df


def sketchy_himmelblau_digestion(df: pd.DataFrame, p: float = 0.1) -> pd.DataFrame:
    """
    Evaluates the constrained Himmelblau, where every point is bad with probability p.
    """

    df = constrained_himmelblau_digestion(df)
    bad = np.random.choice(a=[True, False], size=len(df), p=[p, 1 - p])
    df.loc[:, "himmelblau"] = np.where(bad, np.nan, np.array(df.himmelblau))

    return df


"""
Chankong and Haimes function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def chankong_and_haimes_digestion(df: pd.DataFrame) -> pd.DataFrame:
    df["f1"] = (df.x1 - 2) ** 2 + (df.x2 - 1) + 2
    df["f2"] = 9 * df.x1 - (df.x2 - 1) + 2
    df["c1"] = df.x1**2 + df.x2**2
    df["c2"] = df.x1 - 3 * df.x2 + 10
    return df


def mock_kbs_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests a beam waist and height into the feedback.
    """
    sigma_x = functions.gaussian_beam_waist(df.x1.values, df.x2.values)
    sigma_y = functions.gaussian_beam_waist(df.x3.values, df.x4.values)
    df["x_width"] = 2 * sigma_x
    df["y_width"] = 2 * sigma_y
    return df


def binh_korn_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """

    f1, f2 = functions.binh_korn(df.x1.values, df.x2.values)
    df["f1"] = f1
    df["f2"] = f2
    return df
