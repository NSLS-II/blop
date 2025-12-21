import numpy as np
import scipy as sp


def get_beam_stats(image: np.ndarray, threshold: float = 0.5) -> dict[str, float | np.ndarray]:
    ny, nx = image.shape

    fim = image.copy()
    fim -= np.median(fim, axis=0)
    fim -= np.median(fim, axis=1)[:, None]

    fim = sp.ndimage.median_filter(fim, size=3)
    fim = sp.ndimage.gaussian_filter(fim, sigma=1)

    m = fim > (threshold * fim.max())
    area = m.sum()
    if area == 0.0:
        return {
            "max": 0.0,
            "sum": 0.0,
            "area": 0.0,
            "cen_x": 0.0,
            "cen_y": 0.0,
            "wid_x": 0.0,
            "wid_y": 0.0,
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "bbox": np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        }

    cs_x = np.cumsum(m.sum(axis=0)) / area
    cs_y = np.cumsum(m.sum(axis=1)) / area

    # q_min, q_max = [0.15865, 0.84135]  # one sigma
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


from .beamline import DatabrokerBeamline, DatabrokerDetector, TiledBeamline, TiledDetector  # noqa: E402, F401
from .handlers import HDF5Handler  # noqa: E402, F401
