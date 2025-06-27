import pandas as pd
import xarray

from ..utils import get_beam_stats


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


# TODO: make more general to include when there are multiple image keys
def beam_stats_digestion(xr: xarray.DataArray, image_key: str, **kwargs) -> pd.DataFrame:
    # converts the xarray dataset to a pandas dataframe
    image_key_col = xr[image_key].values
    xr = xr.drop_vars(image_key)
    internal_data = xr.to_dataframe()

    external_data= pd.DataFrame({image_key: [image_key_col[i] for i in range(image_key_col.shape[0])]}, index=range(image_key_col.shape[0]))
    df = pd.concat([internal_data, external_data], axis=1)

    # Get the beam stats for each image in the dataframe and add them as new columns
    df = pd.concat([df, df[image_key].apply(lambda img: pd.Series(get_beam_stats(img, **kwargs)))], axis=1)
    print(len(df))
    return df
