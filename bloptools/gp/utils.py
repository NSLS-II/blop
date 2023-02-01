import numpy as np
import scipy as sp

def get_density(image, extent=None):
    
    if not image.sum() > 0: return np.nan 
        
    (xmin, xmax), (ymin, ymax) = extent if extent is not None else ((0, 1), (0, 1))
    
    # print((xmin, xmax), (ymin, ymax))
    
    x_peak, y_peak, x_width, y_width, x_bounds, y_bounds, flux = get_beam_stats(image, ((xmin, xmax), (ymin, ymax)))
    
    return flux / (x_width * y_width)

def get_beam_stats(image, extents, q_beam=0.8):
    
    _image = image + 1e-6 * image.std() * np.random.standard_normal(image.shape)
    _imsum = _image.sum()

    q  = np.linspace(0,1,1024)
    dq = np.gradient(q).mean()
    nq = int(q_beam / dq)

    nx, ny = _image.shape
    xe, ye = extents 
    
    us_x, us_y = np.linspace(*xe, 1024), np.linspace(*ye, 1024)
    
    # find the smallest bounds such that they have at least q_beam times the total flux between them

    # normalized cumulative sum
    ncs0 = sp.interpolate.interp1d(np.cumsum(_image.sum(axis=0))/_imsum, np.arange(nx), kind='linear', fill_value='extrapolate')(q)
    ncs1 = sp.interpolate.interp1d(np.cumsum(_image.sum(axis=1))/_imsum, np.arange(ny), kind='linear', fill_value='extrapolate')(q)
    
    is0 = (ncs0[nq:] - ncs0[:-nq]).argmin()
    is1 = (ncs1[nq:] - ncs1[:-nq]).argmin()
    
    x_bounds = np.interp([ncs0[is0], ncs0[is0+nq]], np.arange(nx), np.linspace(*xe, nx))
    y_bounds = np.interp([ncs1[is1], ncs1[is1+nq]], np.arange(ny), np.linspace(*ye, ny))
    
    # find the coordinates of the maximum with interpolation
    x_peak = us_x[sp.interpolate.interp1d(np.linspace(*xe, nx), _image.sum(axis=0), kind='quadratic')(us_x).argmax()]
    y_peak = us_y[sp.interpolate.interp1d(np.linspace(*ye, ny), _image.sum(axis=1), kind='quadratic')(us_y).argmax()]
    
    x_width  = np.diff(x_bounds)[0]
    y_width  = np.diff(y_bounds)[0]

    return x_peak, y_peak, x_width, y_width, x_bounds, y_bounds, _imsum

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