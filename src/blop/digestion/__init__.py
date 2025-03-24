import pandas as pd
import numpy as np
from ..utils import get_beam_stats


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


def beam_stats_digestion(df: pd.DataFrame, image_key, **kwargs) -> pd.DataFrame:
    for index, entry in df.iterrows():
        stats = get_beam_stats(entry.loc[image_key].astype(dtype = np.float64), **kwargs)
    
        for attr, value in stats.items():
            if attr not in ["bbox"]:
                df.loc[index, attr] = value

    return df
