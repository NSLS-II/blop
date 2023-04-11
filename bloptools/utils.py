import numpy as np
import scipy as sp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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


def get_routing(origin, points):
    """
    Finds an efficient routing between $n$ points, after normalizing each dimension.
    Returns $n-1$ indices, ignoring the zeroeth index (the origin).
    """

    _points = np.r_[np.atleast_2d(origin), points]

    rel_points = _points / _points.std(axis=0)

    # delay_matrix = gpo.delay_estimate(rel_points[:,None,:] - rel_points[None,:,:])
    delay_matrix = np.sqrt(np.square(rel_points[:, None, :] - rel_points[None, :, :]).sum(axis=-1))
    delay_matrix = (1e6 * delay_matrix).astype(int)  # it likes integers idk

    manager = pywrapcp.RoutingIndexManager(
        len(_points), 1, 0
    )  # number of depots, number of salesmen, starting index
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
    dir(solution)

    index = routing.Start(0)
    route_indices, route_delays = [], []
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_delays.append(routing.GetArcCostForVehicle(previous_index, index, 0))
        route_indices.append(index)

    return np.array(route_indices)[:-1] - 1, route_delays[:-1]


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
