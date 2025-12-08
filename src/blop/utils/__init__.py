import warnings

import botorch  # type: ignore[import-untyped]
import numpy as np
import scipy as sp  # type: ignore[import-untyped]
import torch
from python_tsp.heuristics import solve_tsp_simulated_annealing  # type: ignore[import-untyped]

from . import functions  # noqa

warnings.warn("The utils module is deprecated and will be removed in Blop v1.0.0.", DeprecationWarning, stacklevel=2)


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


def cummax(x: np.ndarray) -> list[float]:
    return [np.nanmax(x[: i + 1]) for i in range(len(np.atleast_1d(x)))]


def sobol_sampler(bounds: torch.Tensor, n: int, q: int = 1) -> torch.Tensor:
    """
    Returns $n$ quasi-randomly sampled points within the bounds (a 2 by d tensor)
    and batch size $q$
    """
    return botorch.utils.sampling.draw_sobol_samples(bounds, n=n, q=q)


def normalized_sobol_sampler(n: int, d: int) -> torch.Tensor:
    """
    Returns $n$ quasi-randomly sampled points in the [0,1]^d hypercube
    """
    normalized_bounds = torch.outer(torch.tensor([0, 1]), torch.ones(d))
    return sobol_sampler(normalized_bounds, n=n, q=1)


def estimate_root_indices(x: np.ndarray) -> np.ndarray:
    # or, indices_before_sign_changes
    i_whole = np.where(np.sign(x[1:]) != np.sign(x[:-1]))[0]
    i_part = 1 - x[i_whole + 1] / (x[i_whole + 1] - x[i_whole])
    return i_whole + i_part


def _fast_psd_inverse(M: np.ndarray) -> np.ndarray:
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """
    cholesky, _ = sp.linalg.lapack.dpotrf(M)
    invM, _ = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)


def mprod(*M: np.ndarray) -> np.ndarray:
    res = M[0]
    for m in M[1:]:
        res = np.matmul(res, m)
    return res


def route(start_point: np.ndarray, points: np.ndarray, dim_weights: float | np.ndarray = 1) -> np.ndarray:
    """
    Returns the indices of the most efficient way to visit `points`, starting from `start_point`.
    """

    total_points = np.concatenate(
        [start_point[None], points], axis=0
    )  # add the starting point, even though we won't go there
    points_dim_range = np.ptp(total_points, axis=0)
    dim_mask = points_dim_range > 0
    scaled_points = (total_points - total_points.min(axis=0)) * (
        dim_weights / np.where(points_dim_range > 0, points_dim_range, 1)
    )
    D = np.sqrt(np.square(scaled_points[:, None, :] - scaled_points[None, :, :]).sum(axis=-1))
    D = (D / np.where(D > 0, D, np.inf).min()).astype(int)
    D[:, 0] = 0  # zero cost to return, since we don't care where we end up

    if dim_mask.sum() == 0:
        return np.arange(len(points))

    permutation, _ = solve_tsp_simulated_annealing(D / np.where(D > 0, D, np.inf).min())
    return np.array(permutation[1:]) - 1  # ignore the starting point since we're there already


def get_movement_time(x: float | np.ndarray, v_max: float, a: float) -> float | np.ndarray:
    """
    How long does it take an object to go distance $x$ with acceleration $a$ and maximum velocity $v_max$?
    That's what this function answers.
    """
    return 2 * np.sqrt(np.abs(x) / a) * (np.abs(x) < v_max**2 / a) + (np.abs(x) / v_max + v_max / a) * (
        np.abs(x) > v_max**2 / a
    )
