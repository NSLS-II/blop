import numpy as np
import scipy as sp
import pandas as pd

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 2a8fbcf15463292b877801d3a77e6e8a22c944d3
def parse_images(images, index_to_parse, n_max_median=1024, verbose=False):
    '''
    Parse a stack of images of shape (n_f, n_y, n_x)
    '''
    
    # sample at most n_max_median points in estimating the background
    background = np.median(images[np.unique(np.linspace(0,len(images)-1,n_max_median).astype(int))], axis=0)
    
    beam_stats = pd.DataFrame(columns=['x_min', 'x_max', 'y_min', 'y_max', 'flux', 'separability'])

    for i, image in enumerate(images):
        
        if not i in index_to_parse:
            continue
        
        beam_stats.loc[i] = _get_beam_stats(image)# - background)
        if verbose: print(i); ip.display.clear_output(wait=True)
        
    return beam_stats

def _get_beam_stats(image, beam_prop=0.8):
    '''
    Parse the beam from an image. Returns the bounding box, along with a flux 
    estimate and a goodness of fit parameter. This should go off without a hitch
    as long as beam_prop is less than 1. 
    '''
    
    n_y, n_x = image.shape
    u, s, v = np.linalg.svd(image - image.mean())
    
    separability = np.square(s[0])/np.square(s).sum()
    
    # q refers to "quantile"
    q_min, q_max = 0.5 * (1 - beam_prop), 0.5 * (1 + beam_prop)
    
    # these represent the cumulative proportion of the beam, as captured by the SVD.
    cs_beam_x = np.cumsum(v[0]) / np.sum(v[0])
    cs_beam_y = np.cumsum(u[:,0]) / np.sum(u[:,0])
<<<<<<< HEAD

    # the first coordinate where the cumulative beam is greater than the minimum
    i_q_min_x = np.where((cs_beam_x[1:] > q_min) & (cs_beam_x[:-1] < q_min))[0][0]
    i_q_min_y = np.where((cs_beam_y[1:] > q_min) & (cs_beam_y[:-1] < q_min))[0][0]
    
    # the last coordinate where the cumulative beam is less than the maximum
    i_q_max_x = np.where((cs_beam_x[1:] > q_max) & (cs_beam_x[:-1] < q_max))[0][-1]
    i_q_max_y = np.where((cs_beam_y[1:] > q_max) & (cs_beam_y[:-1] < q_max))[0][-1]

    # interpolate, so that we can go finer than one pixel. this quartet is the "bounding box"
    x_min = np.interp(q_min, cs_beam_x[[i_q_min_x,i_q_min_x+1]], [i_q_min_x,i_q_min_x+1])
    x_max = np.interp(q_max, cs_beam_x[[i_q_max_x,i_q_max_x+1]], [i_q_max_x,i_q_max_x+1])
    y_min = np.interp(q_min, cs_beam_y[[i_q_min_y,i_q_min_y+1]], [i_q_min_y,i_q_min_y+1])
    y_max = np.interp(q_max, cs_beam_y[[i_q_max_y,i_q_max_y+1]], [i_q_max_y,i_q_max_y+1])
    
    # the center and width of the bounding box
    c_x, w_x = 0.5 * (x_min + x_max), x_max - x_min
    c_y, w_y = 0.5 * (y_min + y_max), y_max - y_min
    
    flux = image[int(np.max([0,c_y-w_y])):int(np.min([c_y+w_y,n_y-1]))][:,int(np.max([0,c_x-w_x])):int(np.min([c_x+w_x,n_x-1]))].sum()
    
    return x_min, x_max, y_min, y_max, flux, separability
=======
=======
def parse_images(images, extents=None, index_to_parse=None, n_max_median=1024, remove_background=False, verbose=False):
    '''
    Parse a stack of images of shape (n_f, n_y, n_x)
    '''
    if extents is None: extents = [None for i in range(len(images))]
    if index_to_parse is None: index_to_parse = np.arange(len(images))

    # sample at most n_max_median points in estimating the background
    background = np.median(images[np.unique(np.linspace(0,len(images)-1,n_max_median).astype(int))], axis=0) if remove_background else 0
    beam_stats = pd.DataFrame(columns=['x_min', 'x_max', 'y_min', 'y_max', 'flux', 'maximum', 'separability'])

    for i, (image, extent) in enumerate(zip(images, extents)):
        if not i in index_to_parse:
            continue

        beam_stats.loc[i] = _get_beam_stats(image - background, extent)
        #if verbose: print(i); ip.display.clear_output(wait=True)

    beam_stats['w_x'] = beam_stats.x_max - beam_stats.x_min
    beam_stats['w_y'] = beam_stats.y_max - beam_stats.y_min

    beam_stats['pixel_area'] = (images > 0.05 * images.max(axis=(1,2))[:,None,None]).sum(axis=(1,2))

    beam_stats['fitness'] = beam_stats.separability / (beam_stats.w_x**2 + beam_stats.w_y**2)

    beam_stats['fitness'] = beam_stats['flux'] / beam_stats['pixel_area']

    OOB_buffer = 16

    n_f, n_y, n_x = images.shape

    bad  = beam_stats.x_min.values < OOB_buffer
    bad |= beam_stats.x_max.values > n_x - 1 - OOB_buffer
    bad |= beam_stats.y_min.values < OOB_buffer
    bad |= beam_stats.y_max.values > n_y - 1 - OOB_buffer
    bad |= beam_stats.separability.values < 0.8
    #bad |= images.sum(axis=(1,2)) == 0

    beam_stats.loc[bad, 'fitness'] = np.nan

    return beam_stats

def _get_beam_stats(image, extent, beam_prop=0.5):
    '''
    Parse the beam from an image. Returns the bounding box, along with a flux
    estimate and a goodness of fit parameter. This should go off without a hitch
    as long as beam_prop is less than 1.
    '''

    n_y, n_x = image.shape

    if image.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if extent is None: extent = np.array([0, n_y, 0, n_x])

    # print(f'{extent = }')

    u, s, v = np.linalg.svd(image - image.mean())

    separability = np.square(s[0])/np.square(s).sum()

    # q refers to "quantile"
    q_min, q_max = 0.5 * (1 - beam_prop), 0.5 * (1 + beam_prop)

    # these represent the cumulative proportion of the beam, as captured by the SVD.
    cs_beam_x = np.cumsum(v[0]) / np.sum(v[0])
    cs_beam_y = np.cumsum(u[:,0]) / np.sum(u[:,0])
    cs_beam_x[0], cs_beam_y[0] = 0, 0 # change this lol
>>>>>>> abd3ed6a693f3b1be896bd88f9ba9aa1a1147487

    # the first coordinate where the cumulative beam is greater than the minimum
    i_q_min_x = np.where((cs_beam_x[1:] > q_min) & (cs_beam_x[:-1] < q_min))[0][0]
    i_q_min_y = np.where((cs_beam_y[1:] > q_min) & (cs_beam_y[:-1] < q_min))[0][0]
<<<<<<< HEAD
    
=======

>>>>>>> abd3ed6a693f3b1be896bd88f9ba9aa1a1147487
    # the last coordinate where the cumulative beam is less than the maximum
    i_q_max_x = np.where((cs_beam_x[1:] > q_max) & (cs_beam_x[:-1] < q_max))[0][-1]
    i_q_max_y = np.where((cs_beam_y[1:] > q_max) & (cs_beam_y[:-1] < q_max))[0][-1]

    # interpolate, so that we can go finer than one pixel. this quartet is the "bounding box"
<<<<<<< HEAD
    x_min = np.interp(q_min, cs_beam_x[[i_q_min_x,i_q_min_x+1]], [i_q_min_x,i_q_min_x+1])
    x_max = np.interp(q_max, cs_beam_x[[i_q_max_x,i_q_max_x+1]], [i_q_max_x,i_q_max_x+1])
    y_min = np.interp(q_min, cs_beam_y[[i_q_min_y,i_q_min_y+1]], [i_q_min_y,i_q_min_y+1])
    y_max = np.interp(q_max, cs_beam_y[[i_q_max_y,i_q_max_y+1]], [i_q_max_y,i_q_max_y+1])
    
    # the center and width of the bounding box
    c_x, w_x = 0.5 * (x_min + x_max), x_max - x_min
    c_y, w_y = 0.5 * (y_min + y_max), y_max - y_min
    
    flux = image[int(np.max([0,c_y-w_y])):int(np.min([c_y+w_y,n_y-1]))][:,int(np.max([0,c_x-w_x])):int(np.min([c_x+w_x,n_x-1]))].sum()
    
    return x_min, x_max, y_min, y_max, flux, separability
=======
    # (let's make this more efficient later)
    i_x_min = np.interp(q_min, cs_beam_x[[i_q_min_x,i_q_min_x+1]], [i_q_min_x,i_q_min_x+1])
    i_x_max = np.interp(q_max, cs_beam_x[[i_q_max_x,i_q_max_x+1]], [i_q_max_x,i_q_max_x+1])
    i_y_min = np.interp(q_min, cs_beam_y[[i_q_min_y,i_q_min_y+1]], [i_q_min_y,i_q_min_y+1])
    i_y_max = np.interp(q_max, cs_beam_y[[i_q_max_y,i_q_max_y+1]], [i_q_max_y,i_q_max_y+1])

    x_min = np.interp(i_x_min, [0, n_x], extent[2:])
    x_max = np.interp(i_x_max, [0, n_x], extent[2:])
    y_min = np.interp(i_y_min, [0, n_y], extent[:2])
    y_max = np.interp(i_y_max, [0, n_y], extent[:2])

    # the center and width of the bounding box
    c_x, w_x = 0.5 * (x_min + x_max), x_max - x_min
    c_y, w_y = 0.5 * (y_min + y_max), y_max - y_min

    i_c_x, i_w_x = 0.5 * (i_x_min + i_x_max), i_x_max - i_x_min
    i_c_y, i_w_y = 0.5 * (i_y_min + i_y_max), i_y_max - i_y_min

    flux = image[int(np.max([0,i_c_y-i_w_y])):int(np.min([i_c_y+i_w_y,n_y-1]))][:,int(np.max([0,i_c_x-i_w_x])):int(np.min([i_c_x+i_w_x,n_x-1]))].sum()

    return x_min, x_max, y_min, y_max, flux, image.max() - image.mean(), separability
>>>>>>> abd3ed6a693f3b1be896bd88f9ba9aa1a1147487
>>>>>>> 2a8fbcf15463292b877801d3a77e6e8a22c944d3

def process_beam(image, separable_threshold=0.1):

    gaussian = lambda t, a, c, s, o, const : a * np.exp(-np.abs((t-c)/s)**o) + const

    def get_profile_pars(profile):

        ydata = profile.copy()
        ndata = len(ydata)
        xdata = np.arange(ndata).astype(float)

        p0 = [ydata.ptp(), np.argmax(ydata), 1e-1 * ndata, 2, ydata.min()]

        bounds = [[0,0,0,0,-np.inf],[np.inf,ndata-1,ndata,16,np.inf]]

        fit_pars, fit_cpars = sp.optimize.curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds)

        RSS = np.square(ydata - gaussian(xdata, *fit_pars)).sum()
        TSS = np.square(ydata).sum()

        co_r_2 = RSS / TSS

        return fit_pars, np.sqrt(np.diag(fit_cpars)), co_r_2

    ny, nx = image.shape

    u,s,v = np.linalg.svd(image - np.median(image))
    (_, cx, sx, ox, _), (_, cx_err, sx_err, _, _), x_cr2 = get_profile_pars(v[0])
    (_, cy, sy, oy, _), (_, cy_err, sy_err, _, _), y_cr2 = get_profile_pars(u[:,0])

    print(f'cx = {cx:.03f} ± {cx_err:.03f} | sx = {sx:.03f} ± {sx_err:.03f} | {x_cr2:.03f}')
    print(f'cy = {cy:.03f} ± {cy_err:.03f} | sy = {sy:.03f} ± {sy_err:.03f} | {y_cr2:.03f}')

    if (x_cr2 > separable_threshold) or (y_cr2 > separable_threshold): return np.nan * image

    crop_sigma = 3

    x_crop_min = np.maximum(int(cx - crop_sigma * sx), 0)
    x_crop_max = np.minimum(int(cx + crop_sigma * sx), nx - 1)
    y_crop_min = np.maximum(int(cy - crop_sigma * sy), 0)
    y_crop_max = np.minimum(int(cy + crop_sigma * sy), ny - 1)

    cropped_beam = image[y_crop_min:y_crop_max, x_crop_min:x_crop_max]

    cropped_ny, cropped_nx = cropped_beam.shape
    on_edge = np.ones((cropped_ny, cropped_nx)).astype(bool)
    on_edge[1:-1, 1:-1] = False
    background_median = np.median(cropped_beam[on_edge])

    return cropped_beam - background_median
