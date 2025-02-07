import pandas as pd

from ..utils import get_beam_stats


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


def beam_stats_digestion(df: pd.DataFrame, image_key: str, **kwargs) -> pd.DataFrame:
    df = pd.concat([df, df[image_key].apply(lambda img: pd.Series(get_beam_stats(img, **kwargs)))], axis=1)
    return df
