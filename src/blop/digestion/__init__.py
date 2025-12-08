import warnings

import pandas as pd

from ..utils import get_beam_stats

warnings.warn("The digestion module is deprecated and will be removed in Blop v1.0.0.", DeprecationWarning, stacklevel=2)


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


def beam_stats_digestion(df, image_key: str, **kwargs) -> dict:
    # Get the beam stats for each image in the dataframe and add them as new columns
    stats_list_of_dicts = [get_beam_stats(img, **kwargs) for img in df[image_key]]
    processed_stats = {}
    if stats_list_of_dicts:
        for key in stats_list_of_dicts[0].keys():
            processed_stats[key] = []

        for stat_dict in stats_list_of_dicts:
            for key, value in stat_dict.items():
                processed_stats[key].append(value)
        df.update(processed_stats)
    return df
