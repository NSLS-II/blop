import numpy as np
import scipy as sp
import pandas as pd

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