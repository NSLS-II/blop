import numpy as np
import pandas as pd

from ..utils import functions


def himmelblau_digestion(df: dict) -> dict:
    """
    Digests Himmelblau's function into the feedback.
    """
    num_rows = len(next(iter(df.values())))

    df["x1"] = [0 if val is None else val for val in df.get("x1", [0] * num_rows)]
    df["x2"] = [0 if val is None else val for val in df.get("x2", [0] * num_rows)]
    df["himmelblau"] = [functions.himmelblau(x1=df["x1"][i], x2=df["x2"][i]) for i in range(num_rows)]
    df["himmelblau_transpose"] = [functions.himmelblau(x1=df["x2"][i], x2=df["x1"][i]) for i in range(num_rows)]

    return df


def constrained_himmelblau_digestion(df):
    """
    Digests Himmelblau's function into the feedback, constrained with NaN for a distance of more than 6 from the origin.
    """
    df = himmelblau_digestion(df)
    df["himmelblau"] = [
        np.nan if (x1_val**2 + x2_val**2 >= 36) else himmelblau_val
        for x1_val, x2_val, himmelblau_val in zip(df["x1"], df["x2"], df["himmelblau"])
    ]
    return df


def sketchy_himmelblau_digestion(df, p: float = 0.1): #-> pd.DataFrame:
    """
    Evaluates the constrained Himmelblau, where every point is bad with probability p.
    """

    df = constrained_himmelblau_digestion(df)
    bad = np.random.choice(a=[True, False], size=len(next(iter(df.values()))), p=[p, 1 - p])
    df['himmelblau'] = [
        np.nan if is_bad else val
        for val, is_bad in zip(df['himmelblau'], bad)
    ]
    return df

def chankong_and_haimes_digestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chankong and Haimes function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    
    df["f1"], df["f2"], df["c1"], df["c2"] = [], [], [], []
    for (val_x1, val_x2) in zip(df.get("x1"), df.get("x2")):
        df["f1"].append((val_x1 - 2) ** 2 + (val_x2 - 1) + 2)
        df["f2"].append(9 * val_x1 - (val_x2 - 1) + 2)
        df["c1"].append(val_x1**2 + val_x2**2)
        df["c2"].append(val_x1 - 3 * val_x2 + 10)
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
