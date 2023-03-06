import numpy as np
import scipy as sp
import pandas as pd

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def get_routing(origin, points):
    '''
    Finds an efficient routing between $n$ points, after normalizing each dimension.
    Returns $n-1$ indices, ignoring the zeroeth index (the origin).
    '''
    
    _points = np.r_[np.atleast_2d(origin), points]
    
    rel_points = _points / _points.std(axis=0)

    #delay_matrix = gpo.delay_estimate(rel_points[:,None,:] - rel_points[None,:,:])
    delay_matrix  = np.sqrt(np.square(rel_points[:,None,:] - rel_points[None,:,:]).sum(axis=-1))
    delay_matrix *= 1000
    
    manager = pywrapcp.RoutingIndexManager(len(_points), 1, 0) # number of depots, number of salesmen, starting index
    routing = pywrapcp.RoutingModel(manager)

    def delay_callback(from_index, to_index):
        to_node = manager.IndexToNode(to_index)
        if to_node == 0: return 0 # it is free to return to the depot from anywhere; we just won't do it
        from_node = manager.IndexToNode(from_index)
        return delay_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(delay_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

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


def get_movement_time(x,v_max,a):
    '''
    How long does it take an object to go distance $x$ with acceleration $a$ and maximum velocity $v_max$?
    That's what this function answers. 
    '''
    return 2*np.sqrt(np.abs(x)/a)*(np.abs(x)<v_max**2/a)+(np.abs(x)/v_max+v_max/a)*(np.abs(x)>v_max**2/a) # it works lol

def parse_images(images, extents=None, index_to_parse=None, n_max_median=1024, remove_background=False, verbose=False):
    '''
    Parse a stack of images of shape (n_f, n_y, n_x)
    '''
    n_f, n_y, n_x = images.shape

    if extents is None: extents = [None for i in range(n_f)]
    if index_to_parse is None: index_to_parse = np.arange(n_f)

    # sample at most n_max_median points in estimating the background
    background = np.median(images[np.unique(np.linspace(0,len(images)-1,n_max_median).astype(int))], axis=0) if remove_background else 0
    beam_stats = pd.DataFrame(columns=['x_min', 'x_max', 'y_min', 'y_max', 'rel_x_min', 'rel_x_max', 'rel_y_min', 'rel_y_max', 'flux', 'maximum', 'separability'], dtype=float)

    for i, (image, extent) in enumerate(zip(images, extents)):

        if not i in index_to_parse:
            continue
        if extent is None: 
            extent = np.array([0, n_y, 0, n_x])

        beam_stats.loc[i, ['rel_x_min', 'rel_x_max', 'rel_y_min', 'rel_y_max', 'flux', 'maximum', 'separability']] = _get_beam_stats(image - background, beam_prop=0.95)
        beam_stats.loc[i, ['x_min', 'x_max']] = np.interp(beam_stats.loc[i, ['rel_x_min', 'rel_x_max']].values, [0, 1], extent[2:])
        beam_stats.loc[i, ['y_min', 'y_max']] = np.interp(beam_stats.loc[i, ['rel_y_min', 'rel_y_max']].values, [0, 1], extent[:-2])
        #if verbose: print(i); ip.display.clear_output(wait=True)

    beam_stats['w_x'] = beam_stats.x_max - beam_stats.x_min
    beam_stats['w_y'] = beam_stats.y_max - beam_stats.y_min

    beam_stats['pixel_area'] = (images > 0.05 * images.max(axis=(1,2))[:,None,None]).sum(axis=(1,2))

<<<<<<< HEAD
    beam_stats['fitness'] = beam_stats['fitness'] = beam_stats.flux / ((1 - beam_stats.separability) ** 2 * (beam_stats.w_x**2 + beam_stats.w_y**2))
=======
    beam_stats['fitness'] = beam_stats.separability / (beam_stats.w_x**2 + beam_stats.w_y**2)
>>>>>>> 7a50be43b77600402ee182af6535314c7f05ace0

    #beam_stats['fitness'] = beam_stats['flux'] / beam_stats['pixel_area']

    OOB_rel_buffer = 1/32 # out-of-bounds relative buffer

    bad  = beam_stats.rel_x_min.values < OOB_rel_buffer
    bad |= beam_stats.rel_x_max.values > 1 - OOB_rel_buffer
    bad |= beam_stats.rel_y_min.values < OOB_rel_buffer
    bad |= beam_stats.rel_y_max.values > 1 - OOB_rel_buffer
    bad |= beam_stats.separability.values < 0.5 # at least half the variance must be explained by a beam

    beam_stats.loc[bad, 'fitness'] = np.nan # set the fitness of questionable beams to nan

    return beam_stats

def _get_beam_stats(image, beam_prop=0.5):
    '''
    Parse the beam from an image. Returns the normalized bounding box, along with a flux
    estimate and a goodness of fit parameter. This should go off without a hitch
    as long as beam_prop is less than 1.
    '''

    n_y, n_x = image.shape

    if image.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    

    # print(f'{extent = }')

    u, s, v = np.linalg.svd(image - image.mean())
    separability = np.square(s[0])/np.square(s).sum()

    # q refers to "quantile"
    q_min, q_max = 0.5 * (1 - beam_prop), 0.5 * (1 + beam_prop)

    # these represent the cumulative proportion of the beam, as captured by the SVD.
    cs_beam_x = np.cumsum(v[0]) / np.sum(v[0])
    cs_beam_y = np.cumsum(u[:,0]) / np.sum(u[:,0])
    cs_beam_x[0], cs_beam_y[0] = 0, 0 # change this lol

    # the first coordinate where the cumulative beam is greater than the minimum
    i_q_min_x = np.where((cs_beam_x[1:] > q_min) & (cs_beam_x[:-1] < q_min))[0][0]
    i_q_min_y = np.where((cs_beam_y[1:] > q_min) & (cs_beam_y[:-1] < q_min))[0][0]
    
    # the last coordinate where the cumulative beam is less than the maximum
    i_q_max_x = np.where((cs_beam_x[1:] > q_max) & (cs_beam_x[:-1] < q_max))[0][-1]
    i_q_max_y = np.where((cs_beam_y[1:] > q_max) & (cs_beam_y[:-1] < q_max))[0][-1]

    # interpolate, so that we can go finer than one pixel. this quartet is the "bounding box", and each value is between 0 and 1.
    # (let's make this more efficient later)
    x_min = np.interp(q_min, cs_beam_x[[i_q_min_x,i_q_min_x+1]], [i_q_min_x,i_q_min_x+1]) / n_x
    x_max = np.interp(q_max, cs_beam_x[[i_q_max_x,i_q_max_x+1]], [i_q_max_x,i_q_max_x+1]) / n_x
    y_min = np.interp(q_min, cs_beam_y[[i_q_min_y,i_q_min_y+1]], [i_q_min_y,i_q_min_y+1]) / n_y
    y_max = np.interp(q_max, cs_beam_y[[i_q_max_y,i_q_max_y+1]], [i_q_max_y,i_q_max_y+1]) / n_y

    return x_min, x_max, y_min, y_max, s[0] * np.outer(u[:,0], v[0]).sum(), image.max() - image.mean(), separability

