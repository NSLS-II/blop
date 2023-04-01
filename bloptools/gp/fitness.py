import numpy as np
import pandas as pd

from . import utils

def tes(entry, args):

    image = getattr(entry, args['image'])
    #if (args['flux'] not in entry.index) or (args['flux'] is not None):
    #    flux = 1e0
    #else:
    flux = getattr(entry, args['flux'])

    background = args['background']

    x_min, x_max, y_min, y_max, separability = utils.get_beam_stats(image - background, beam_prop=args['beam_prop'])

    n_y, n_x = image.shape

    # u, s, v = np.linalg.svd(image - background)

    # separability = np.square(s[0]) / np.square(s).sum()

    # ymode, xmode = u[:,0], v[0]

    # x_roots = utils.estimate_root_indices(np.abs(xmode) - args['beam_prop'] * np.abs(xmode).max())
    # y_roots = utils.estimate_root_indices(np.abs(ymode) - args['beam_prop'] * np.abs(ymode).max())

    # x_min, x_max = x_roots.min(), x_roots.max()
    # y_min, y_max = y_roots.min(), y_roots.max()

    width_x = x_max - x_min
    width_y = y_max - y_min

    bad  = x_min < (n_x + 1) * args['OOB_rel_buffer']
    bad |= x_max > (n_x + 1) * (1 - args['OOB_rel_buffer'])
    bad |= y_min < (n_y + 1) * args['OOB_rel_buffer']
    bad |= y_max > (n_y + 1) * (1 - args['OOB_rel_buffer'])
    bad |= separability < args['min_separability']
    bad |= width_x < 2 
    bad |= width_y < 2

    if bad:
        fitness = np.nan
    
    else:
        fitness = np.log(flux * separability / (width_x ** 2 + width_y ** 2))

    return ('fitness', 'x_min', 'x_max', 'y_min', 'y_max', 'width_x', 'width_y', 'separability'), (fitness, x_min, x_max, y_min, y_max, width_x, width_y, separability)

# def parse_images(
#     images, extents=None, index_to_parse=None, n_max_median=1024, remove_background=False, verbose=False
# ):
#     """
#     Parse a stack of images of shape (n_f, n_y, n_x)
#     """
#     n_f, n_y, n_x = images.shape

#     if extents is None:
#         extents = [None for i in range(n_f)]
#     if index_to_parse is None:
#         index_to_parse = np.arange(n_f)

#     # sample at most n_max_median points in estimating the background
#     background = (
#         np.median(images[np.unique(np.linspace(0, len(images) - 1, n_max_median).astype(int))], axis=0)
#         if remove_background
#         else 0
#     )
#     beam_stats = pd.DataFrame(
#         columns=[
#             "x_min",
#             "x_max",
#             "y_min",
#             "y_max",
#             "rel_x_min",
#             "rel_x_max",
#             "rel_y_min",
#             "rel_y_max",
#             "flux",
#             "maximum",
#             "separability",
#         ],
#         dtype=float,
#     )

#     for i, (image, extent) in enumerate(zip(images, extents)):
#         if i not in index_to_parse:
#             continue
#         if extent is None:
#             extent = np.array([0, n_y, 0, n_x])

#         beam_stats.loc[
#             i, ["rel_x_min", "rel_x_max", "rel_y_min", "rel_y_max", "flux", "maximum", "separability"]
#         ] = _get_beam_stats(image - background, beam_prop=0.95)
#         beam_stats.loc[i, ["x_min", "x_max"]] = np.interp(
#             beam_stats.loc[i, ["rel_x_min", "rel_x_max"]].values, [0, 1], extent[2:]
#         )
#         beam_stats.loc[i, ["y_min", "y_max"]] = np.interp(
#             beam_stats.loc[i, ["rel_y_min", "rel_y_max"]].values, [0, 1], extent[:-2]
#         )
#         # if verbose: print(i); ip.display.clear_output(wait=True)

#     beam_stats["w_x"] = beam_stats.x_max - beam_stats.x_min
#     beam_stats["w_y"] = beam_stats.y_max - beam_stats.y_min

#     beam_stats["pixel_area"] = (images > 0.05 * images.max(axis=(1, 2))[:, None, None]).sum(axis=(1, 2))

#     beam_stats["fitness"] = np.log(
#         beam_stats.separability / (beam_stats.w_x**2 + beam_stats.w_y**2)
#     )

#     # beam_stats['fitness'] = beam_stats['flux'] / beam_stats['pixel_area']

#     OOB_rel_buffer = 1 / 32  # out-of-bounds relative buffer

#     bad = beam_stats.rel_x_min.values < OOB_rel_buffer
#     bad |= beam_stats.rel_x_max.values > 1 - OOB_rel_buffer
#     bad |= beam_stats.rel_y_min.values < OOB_rel_buffer
#     bad |= beam_stats.rel_y_max.values > 1 - OOB_rel_buffer
#     bad |= beam_stats.separability.values < 0.1  # at least half the variance must be explained by a beam

#     beam_stats.loc[bad, "fitness"] = np.nan  # set the fitness of questionable beams to nan

#     return beam_stats
