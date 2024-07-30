import botorch
import numpy as np
import scipy as sp
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from . import functions  # noqa


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


def cummax(x):
    return [np.nanmax(x[: i + 1]) for i in range(len(np.atleast_1d(x)))]


def sobol_sampler(bounds, n, q=1):
    """
    Returns $n$ quasi-randomly sampled points within the bounds (a 2 by d tensor)
    and batch size $q$
    """
    return botorch.utils.sampling.draw_sobol_samples(bounds, n=n, q=q)


def normalized_sobol_sampler(n, d):
    """
    Returns $n$ quasi-randomly sampled points in the [0,1]^d hypercube
    """
    normalized_bounds = torch.outer(torch.tensor([0, 1]), torch.ones(d))
    return sobol_sampler(normalized_bounds, n=n, q=1)


def estimate_root_indices(x):
    # or, indices_before_sign_changes
    i_whole = np.where(np.sign(x[1:]) != np.sign(x[:-1]))[0]
    i_part = 1 - x[i_whole + 1] / (x[i_whole + 1] - x[i_whole])
    return i_whole + i_part


def _fast_psd_inverse(M):
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """
    cholesky, dpotrf_info = sp.linalg.lapack.dpotrf(M)
    invM, dpotri_info = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)


def mprod(*M):
    res = M[0]
    for m in M[1:]:
        res = np.matmul(res, m)
    return res


def route(start_point, points, dim_weights=1):
    """
    Returns the indices of the most efficient way to visit `points`, starting from `start_point`.
    """

    total_points = np.r_[np.atleast_2d(start_point), points]
    points_scale = np.ptp(total_points, axis=0)
    dim_mask = points_scale > 0

    if dim_mask.sum() == 0:
        return np.arange(len(points))

    scaled_points = (total_points - total_points.min(axis=0)) * (dim_weights / np.where(points_scale > 0, points_scale, 1))

    delay_matrix = np.sqrt(np.square(scaled_points[:, None, :] - scaled_points[None, :, :]).sum(axis=-1))
    delay_matrix = (1e4 * delay_matrix).astype(int)  # it likes integers idk

    manager = pywrapcp.RoutingIndexManager(len(total_points), 1, 0)  # number of depots, number of salesmen, starting index
    routing = pywrapcp.RoutingModel(manager)

    def delay_callback(from_index, to_index):
        to_node = manager.IndexToNode(to_index)
        if to_node == 0:
            return 0  # it is free to return to the depot from anywhere; we just won't do it
        from_node = manager.IndexToNode(from_index)
        return delay_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(delay_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    route_indices, route_delays = [0], []
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_delays.append(routing.GetArcCostForVehicle(previous_index, index, 0))
        route_indices.append(index)

    # omit the first and last indices, which correspond to the start
    return np.array(route_indices)[1:-1] - 1


def get_movement_time(x, v_max, a):
    """
    How long does it take an object to go distance $x$ with acceleration $a$ and maximum velocity $v_max$?
    That's what this function answers.
    """
    return 2 * np.sqrt(np.abs(x) / a) * (np.abs(x) < v_max**2 / a) + (np.abs(x) / v_max + v_max / a) * (
        np.abs(x) > v_max**2 / a
    )
