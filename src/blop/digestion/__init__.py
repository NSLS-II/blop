import numpy as np
import pandas as pd
import scipy as sp


def get_beam_stats(image, threshold=0.5):
    ny, nx = image.shape

    fim = image.copy()
    fim -= np.median(fim, axis=0)
    fim -= np.median(fim, axis=1)[:, None]

    fim = sp.ndimage.median_filter(fim, size=3)
    fim = sp.ndimage.gaussian_filter(fim, sigma=1)

    m = fim > threshold * fim.max()
    area = m.sum()

    cs_x = np.cumsum(m.sum(axis=0)) / area
    cs_y = np.cumsum(m.sum(axis=1)) / area

    q_min, q_max = [0.15865, 0.84135]  # one sigma
    q_min, q_max = [0.05, 0.95]  # 90%

    x_min, x_max = np.interp([q_min, q_max], cs_x, np.arange(nx))
    y_min, y_max = np.interp([q_min, q_max], cs_y, np.arange(ny))

    stats = {
        "max": fim.max(),
        "sum": fim.sum(),
        "area": area,
        "cen_x": (x_min + x_max) / 2,
        "cen_y": (y_min + y_max) / 2,
        "wid_x": x_max - x_min,
        "wid_y": y_max - y_min,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "bbox": [[x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min]],
    }

    return stats


def default_digestion_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


def beam_stats_digestion(df: pd.DataFrame, image_key, **kwargs) -> pd.DataFrame:
    for index, entry in df.iterrows():
        stats = get_beam_stats(entry.loc[image_key], **kwargs)

        for attr, value in stats.items():
            if attr not in ["bbox"]:
                df.loc[index, attr] = value

    return df
