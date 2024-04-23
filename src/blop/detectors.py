import bluesky.plans as bp  # noqa F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from ophyd import Component as Cpt
from ophyd import Device, Signal

DOF_FIELD_TYPES = {
    "description": "str",
    "readback": "object",
    "type": "str",
    "units": "str",
    "tags": "object",
    "transform": "str",
    "search_domain": "object",
    "trust_domain": "object",
    "domain": "object",
    "active": "bool",
    "read_only": "bool",
}

DOF_TYPES = ["continuous", "binary", "ordinal", "categorical"]
TRANSFORM_DOMAINS = {"log": (0.0, np.inf), "logit": (0.0, 1.0), "arctanh": (-1.0, 1.0)}


def get_beam_stats(im, roi=None, threshold=0.2, method="rms", median_filter_size=3, gaussian_filter_sigma=2, downsample=1):
    if roi:
        imin, imax, jmin, jmax = roi
        # cropped image
        cim = im[jmin:jmax, imin:imax]
    else:
        cim = im

    if downsample > 1:
        cim = cim[::downsample, ::downsample]

    cfim = sp.ndimage.median_filter(cim, size=median_filter_size)
    cfim = sp.ndimage.gaussian_filter(cfim, sigma=gaussian_filter_sigma)

    # filtered cropped image
    cfim = np.where(cfim > (1 - threshold) * cfim.min() + threshold * cfim.max(), cfim, 0)

    stats = {}
    stats["raw_image"] = im
    stats["image"] = cfim
    stats["max"] = cfim.max()
    stats["sum"] = cfim.sum()

    for iax, axis in enumerate(["x", "y"]):
        index = np.arange(cfim.shape[1 - iax])
        profile = cfim.sum(axis=iax)
        profile -= profile.min()
        profile /= profile.max()

        if method == "rms":
            center = np.sum(profile * index) / np.sum(profile)
            width = 4 * np.sqrt(np.sum(profile * np.square(index - center)) / np.sum(profile))

        elif method == "fwhm":
            beam_index = index[profile > 0.5 * profile.max()]
            center = beam_index.mean()
            width = beam_index.ptp()

        elif method == "quantile":
            normed_cumsum = np.cumsum(profile) / np.sum(profile)
            lower, center, upper = np.interp([0.05, 0.5, 0.95], normed_cumsum, index)
            width = upper - lower
        else:
            raise ValueError(f"Invalid method '{method}'.")

        stats[f"center_{axis}"] = center
        stats[f"width_{axis}"] = width

    stats["area"] = (cfim > cfim.max() * threshold).sum()

    stats["eff_area"] = 0.5 * (stats["width_x"] ** 2 + stats["width_y"] ** 2)

    # bs = BeamStats(**stats)

    return stats


beam_stats_fields = ["center_x", "width_x", "center_y", "width_y", "eff_area", "area", "sum", "max"]


class AreaDetectorWithStats(Device):
    center_x = Cpt(Signal, kind="hinted")
    width_x = Cpt(Signal, kind="hinted")
    center_y = Cpt(Signal, kind="hinted")
    width_y = Cpt(Signal, kind="hinted")
    eff_area = Cpt(Signal, kind="hinted")
    area = Cpt(Signal, kind="hinted")
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="hinted")
    xmin = Cpt(Signal, kind="config")
    xmax = Cpt(Signal, kind="config")
    ymin = Cpt(Signal, kind="config")
    ymax = Cpt(Signal, kind="config")

    def __init__(self, device, beam_stats_kwargs={}, roi=None, *args, **kwargs):
        super().__init__(name=device.name, *args, **kwargs)

        self.device = device
        self.beam_stats_kwargs = beam_stats_kwargs

        if roi:
            for i, attr in enumerate(["xmin", "xmax", "ymin", "ymax"]):
                getattr(self, attr).put(roi[i])

    def stage(self):
        super().stage()
        return self.device.stage()

    def trigger(self):
        super().trigger()
        st = self.device.trigger()

        for im in self.device._dataset:
            beam_stats = get_beam_stats(im, **self.beam_stats_kwargs)

            print(beam_stats)

            for attr in beam_stats_fields:
                getattr(self, attr).put(beam_stats[attr])

        return st

    def unstage(self):
        super().unstage()
        self.device.unstage()

    def __post_init__(self):
        self.name = self.device.name

    def test(self, RE, db, num=1):
        (uid,) = RE(bp.count([self.device], num=num))

        im_list = []

        table = db[uid].table(fill=True)

        stats = pd.DataFrame(columns=beam_stats_fields)
        for index, entry in table.iterrows():
            image = getattr(entry, f"{self.device.name}_image")

            beam_stats = get_beam_stats(image, **self.beam_stats_kwargs)

            for attr in beam_stats_fields:
                stats.loc[index, attr] = beam_stats[attr]

            image = image if image.ndim == 3 else image[None]
            im_list.append(image)

        ims = np.concatenate(im_list, axis=0)

        n_ims, n_y, n_x = ims.shape

        fig, axes = plt.subplots(n_ims, 2)
        axes = np.atleast_2d(axes)

        for i in range(n_ims):
            axes[i, 0].imshow(ims[i])

            if self.roi:
                x0, x1, y0, y1 = self.roi

                axes[i, 0].plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], c="r")

                cropped_image = ims[i, int(y0) : int(y1), int(x0) : int(x1)]

                axes[i, 1].imshow(cropped_image)

        print(stats)
