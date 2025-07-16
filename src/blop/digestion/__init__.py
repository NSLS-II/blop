import pandas as pd
import xarray

from ..utils import get_beam_stats


def default_digestion_function(df: pd.DataFrame) -> xarray.DataArray:
    return df.to_xarray()


def beam_stats_digestion(xr: xarray.DataArray, image_key: str, **kwargs) -> xarray.DataArray:
    df = xarray.merge(
        [xr, xr[image_key].pipe(lambda arr: pd.DataFrame([get_beam_stats(img, **kwargs) for img in arr])).to_xarray()],
        compat="override",
    )
    return df
