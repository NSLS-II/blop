import numpy as np
import pandas as pd
import xarray

from ..utils import functions


# TODO
def himmelblau_digestion(xr: xarray.DataArray) -> xarray.DataArray:
    """
    Digests Himmelblau's function into the feedback.
    """
    xr = xr.assign(x1=(xr.x2 * 0 if "x1" not in xr else xr.x1.fillna(0)))
    xr = xr.assign(x2=(xr.x1 * 0 if "x2" not in xr else xr.x2.fillna(0)))
    xr = xr.assign(
        himmelblau=functions.himmelblau(xr.x1, xr.x2),
        himmelblau_transpose=functions.himmelblau(xr.x2, xr.x1),
    )
    return xr


def constrained_himmelblau_digestion(xr: xarray.DataArray) -> xarray.DataArray:
    """
    Digests Himmelblau's function into the feedback, constrained with NaN for a distance of more than 6 from the origin.
    """
    xr = himmelblau_digestion(xr)
    xr = xr.assign(himmelblau=xr.himmelblau.where((xr.x1**2 + xr.x2**2) < 36))
    return xr


def sketchy_himmelblau_digestion(xr: xarray.DataArray, p: float = 0.1) -> xarray.DataArray:
    """
    Evaluates the constrained Himmelblau, where every point is bad with probability p.
    """
    xr = constrained_himmelblau_digestion(xr)
    main_dim = list(xr.dims)[0]
    bad = np.random.choice(a=[True, False], size=xr.sizes[main_dim], p=[p, 1 - p])
    bad_xr = xarray.DataArray(bad, dims=(main_dim,), coords={main_dim: xr.coords[main_dim]})
    xr = xr.assign(himmelblau=xr.himmelblau.where(~bad_xr))
    return xr


"""
Chankong and Haimes function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def chankong_and_haimes_digestion(xr: xarray.DataArray) -> xarray.DataArray:
    xr = xr.assign(
        f1=(xr.x1 - 2) ** 2 + (xr.x2 - 1) + 2,
        f2=9 * xr.x1 - (xr.x2 - 1) + 2,
        c1=xr.x1**2 + xr.x2**2,
        c2=xr.x1 - 3 * xr.x2 + 10,
    )
    return xr


def mock_kbs_digestion(xr: xarray.DataArray) -> xarray.DataArray:
    """
    Digests a beam waist and height into the feedback.
    """
    xr = xr.assign(
        x_width=2 * functions.gaussian_beam_waist(xr.x1, xr.x2),
        y_width=2 * functions.gaussian_beam_waist(xr.x3, xr.x4),
    )
    return xr


def binh_korn_digestion(xr: xarray.DataArray) -> pd.DataFrame:
    """
    Digests Himmelblau's function into the feedback.
    """
    f1, f2 = functions.binh_korn(xr.x1, xr.x2)
    xr = xr.assign(
        f1=f1,
        f2=f2,
    )
    return xr
