import pandas as pd
import xarray

from ..utils import get_beam_stats


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


# TODO: make more general to include when there are multiple image keys
def beam_stats_digestion(xr: xarray.DataArray, image_key: str, **kwargs) -> pd.DataFrame:
    # converts the xarray dataset to a pandas dataframe
    abc = xr[image_key].values
    xr = xr.drop_vars(image_key)
    df1 = xr.to_dataframe()

    df2 = pd.DataFrame({image_key: [abc[i] for i in range(abc.shape[0])]}, index=range(abc.shape[0]))
    df = pd.concat([df1, df2], axis=1)

    # Get the beam stats for each image in the dataframe and add them as new columns
    df = pd.concat([df, df[image_key].apply(lambda img: pd.Series(get_beam_stats(img, **kwargs)))], axis=1)
    print(len(df))
    return df
