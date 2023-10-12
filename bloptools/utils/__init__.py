import botorch
import numpy as np
import scipy as sp
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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


def route(start_point, points):
    """
    Returns the indices of the most efficient way to visit `points`, starting from `start_point`.
    """

    total_points = np.r_[np.atleast_2d(start_point), points]
    points_scale = total_points.ptp(axis=0)
    dim_mask = points_scale > 0

    if dim_mask.sum() == 0:
        return np.arange(len(points))

    normalized_points = (total_points - total_points.min(axis=0))[:, dim_mask] / points_scale[dim_mask]

    delay_matrix = np.sqrt(np.square(normalized_points[:, None, :] - normalized_points[None, :, :]).sum(axis=-1))
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


def get_principal_component_bounds(image, beam_prop=0.5):
    """
    Returns the bounding box in pixel units of an image, along with a goodness of fit parameter.
    This should go off without a hitch as long as beam_prop is less than 1.
    """

    if image.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # print(f'{extent = }')

    u, s, v = np.linalg.svd(image - image.mean())
    separability = np.square(s[0]) / np.square(s).sum()

    # q refers to "quantile"
    q_min, q_max = 0.5 * (1 - beam_prop), 0.5 * (1 + beam_prop)

    # these represent the cumulative proportion of the beam, as captured by the SVD.
    cs_beam_x = np.cumsum(v[0]) / np.sum(v[0])
    cs_beam_y = np.cumsum(u[:, 0]) / np.sum(u[:, 0])
    cs_beam_x[0], cs_beam_y[0] = 0, 0  # change this lol

    # the first coordinate where the cumulative beam is greater than the minimum
    i_q_min_x = np.where((cs_beam_x[1:] > q_min) & (cs_beam_x[:-1] < q_min))[0][0]
    i_q_min_y = np.where((cs_beam_y[1:] > q_min) & (cs_beam_y[:-1] < q_min))[0][0]

    # the last coordinate where the cumulative beam is less than the maximum
    i_q_max_x = np.where((cs_beam_x[1:] > q_max) & (cs_beam_x[:-1] < q_max))[0][-1]
    i_q_max_y = np.where((cs_beam_y[1:] > q_max) & (cs_beam_y[:-1] < q_max))[0][-1]

    # interpolate, so that we can go finer than one pixel. this quartet is the "bounding box", from 0 to 1.
    # (let's make this more efficient later)
    x_min = np.interp(q_min, cs_beam_x[[i_q_min_x, i_q_min_x + 1]], [i_q_min_x, i_q_min_x + 1])
    x_max = np.interp(q_max, cs_beam_x[[i_q_max_x, i_q_max_x + 1]], [i_q_max_x, i_q_max_x + 1])
    y_min = np.interp(q_min, cs_beam_y[[i_q_min_y, i_q_min_y + 1]], [i_q_min_y, i_q_min_y + 1])
    y_max = np.interp(q_max, cs_beam_y[[i_q_max_y, i_q_max_y + 1]], [i_q_max_y, i_q_max_y + 1])

    return (
        x_min,
        x_max,
        y_min,
        y_max,
        separability,
    )


def get_beam_bounding_box(image, thresh=0.5):
    """
    Returns the bounding box in pixel units of an image, along with a goodness of fit parameter.
    This should go off without a hitch as long as beam_prop is less than 1.
    """

    n_y, n_x = image.shape

    if image.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # filter the image
    zim = sp.ndimage.median_filter(image.astype(float), size=3)
    zim -= np.median(zim, axis=0)
    zim -= np.median(zim, axis=1)[:, None]

    x_sum = zim.sum(axis=0)
    y_sum = zim.sum(axis=1)

    x_sum_min_val = thresh * x_sum.max()
    y_sum_min_val = thresh * y_sum.max()

    gtt_x = x_sum > x_sum_min_val
    gtt_y = y_sum > y_sum_min_val

    i_x_min_start = np.where(~gtt_x[:-1] & gtt_x[1:])[0][0]
    i_x_max_start = np.where(gtt_x[:-1] & ~gtt_x[1:])[0][-1]
    i_y_min_start = np.where(~gtt_y[:-1] & gtt_y[1:])[0][0]
    i_y_max_start = np.where(gtt_y[:-1] & ~gtt_y[1:])[0][-1]

    x_min = (
        0
        if gtt_x[0]
        else np.interp(x_sum_min_val, x_sum[[i_x_min_start, i_x_min_start + 1]], [i_x_min_start, i_x_min_start + 1])
    )
    y_min = (
        0
        if gtt_y[0]
        else np.interp(y_sum_min_val, y_sum[[i_y_min_start, i_y_min_start + 1]], [i_y_min_start, i_y_min_start + 1])
    )
    x_max = (
        n_x - 2
        if gtt_x[-1]
        else np.interp(x_sum_min_val, x_sum[[i_x_max_start + 1, i_x_max_start]], [i_x_max_start + 1, i_x_max_start])
    )
    y_max = (
        n_y - 2
        if gtt_y[-1]
        else np.interp(y_sum_min_val, y_sum[[i_y_max_start + 1, i_y_max_start]], [i_y_max_start + 1, i_y_max_start])
    )

    return (
        x_min,
        x_max,
        y_min,
        y_max,
    )


def best_image_feedback(image):
    n_y, n_x = image.shape

    fim = sp.ndimage.median_filter(image, size=3)

    masked_image = fim * (fim - fim.mean() > 0.5 * fim.ptp())

    x_weight = masked_image.sum(axis=0)
    y_weight = masked_image.sum(axis=1)

    x = np.arange(n_x)
    y = np.arange(n_y)

    x0 = np.sum(x_weight * x) / np.sum(x_weight)
    y0 = np.sum(y_weight * y) / np.sum(y_weight)

    xw = 2 * np.sqrt((np.sum(x_weight * x**2) / np.sum(x_weight) - x0**2))
    yw = 2 * np.sqrt((np.sum(y_weight * y**2) / np.sum(y_weight) - y0**2))

    return x0, xw, y0, yw
