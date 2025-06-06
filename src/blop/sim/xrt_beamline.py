import itertools
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import scipy as sp
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import Device, EpicsSignal, Signal
from ophyd.sim import NullStatus, new_uid
from ophyd.utils import make_dir_tree

from blop.sim.handlers import ExternalFileReference
from blop.sim.xrt_kb_model import build_beamline, build_histRGB, run_process
from blop.utils import get_beam_stats

TEST = False


class xrtEpicsScreen(Device):
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="normal")
    area = Cpt(Signal, kind="normal")
    cen_x = Cpt(Signal, kind="hinted")
    cen_y = Cpt(Signal, kind="hinted")
    wid_x = Cpt(Signal, kind="hinted")
    wid_y = Cpt(Signal, kind="hinted")
    image = Cpt(EpicsSignal, "BL:Screen1:Array", kind="normal")
    acquire = Cpt(EpicsSignal, "BL:Screen1:Acquire", kind="normal")
    image_shape = Cpt(Signal, value=(300, 400), kind="normal")
    noise = Cpt(Signal, kind="normal")

    def __init__(self, root_dir: str = "/tmp/blop/sim", verbose: bool = True, noise: bool = True, *args, **kwargs):
        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None
        super().__init__(*args, **kwargs)

    def trigger(self):
        super().trigger()
        self.acquire.put(1)
        while self.acquire.get() > 0:
            time.sleep(0.01)
        raw_image = self.image.get()
        image = raw_image.reshape(*self.image_shape.get())

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))

        self._dataset[current_frame, :, :] = image

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        stats = get_beam_stats(image)

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        return NullStatus()

    def stage(self):
        super().stage()
        date = datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        self._data_file = str(Path(self._resource_document["root"]) / Path(self._resource_document["resource_path"]))

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *self.image_shape.get())),
            maxshape=(None, *self.image_shape.get()),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()

    def unstage(self):
        super().unstage()
        del self._dataset
        self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None


class Detector(Device):
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="normal")
    area = Cpt(Signal, kind="normal")
    cen_x = Cpt(Signal, kind="hinted")
    cen_y = Cpt(Signal, kind="hinted")
    wid_x = Cpt(Signal, kind="hinted")
    wid_y = Cpt(Signal, kind="hinted")
    image = Cpt(ExternalFileReference, kind="normal")
    image_shape = Cpt(Signal, value=(300, 400), kind="normal")
    noise = Cpt(Signal, kind="normal")

    def __init__(self, root_dir: str = "/tmp/blop/sim", verbose: bool = True, noise: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None
        self.noise.put(noise)
        self.limits = [[-0.6, 0.6], [-0.45, 0.45]]
        if TEST:
            self.mplFig = mpl.figure.Figure()
            self.mplFig.subplots_adjust(left=0.15, bottom=0.15, top=0.92)
            self.mplAx = self.mplFig.add_subplot(111)

            xv = np.random.rand(400, 300)
            self.im = self.mplAx.imshow(
                xv.T,
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1e3,
                cmap="jet",
                extent=(self.limits[0][0], self.limits[0][1], self.limits[1][0], self.limits[1][1]),
            )
        self.counter = 0
        self.beamLine = build_beamline()

    def trigger(self):
        super().trigger()
        raw_image = self.generate_beam(noise=self.noise.get())

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))

        self._dataset[current_frame, :, :] = raw_image

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        stats = get_beam_stats(raw_image)
        self.image.put(datum_document["datum_id"])

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        return NullStatus()

    def stage(self):
        super().stage()
        date = datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        self._data_file = str(Path(self._resource_document["root"]) / Path(self._resource_document["resource_path"]))

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *self.image_shape.get())),
            maxshape=(None, *self.image_shape.get()),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()

    def unstage(self):
        super().unstage()
        del self._dataset
        self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items

    def generate_beam_func(self, noise: bool = True):
        nx, ny = self.image_shape.get()

        x = np.linspace(-10, 10, ny)
        y = np.linspace(-10, 10, nx)
        X, Y = np.meshgrid(x, y)

        x0 = self.parent.kbh_ush.get() - self.parent.kbh_dsh.get()
        y0 = self.parent.kbv_usv.get() - self.parent.kbv_dsv.get()
        x_width = np.sqrt(0.2 + 5e-1 * (self.parent.kbh_ush.get() + self.parent.kbh_dsh.get() - 1) ** 2)
        y_width = np.sqrt(0.1 + 5e-1 * (self.parent.kbv_usv.get() + self.parent.kbv_dsv.get() - 2) ** 2)

        beam = np.exp(-0.5 * (((X - x0) / x_width) ** 4 + ((Y - y0) / y_width) ** 4)) / (
            np.sqrt(2 * np.pi) * x_width * y_width
        )

        mask = X > self.parent.ssa_inboard.get()
        mask &= X < self.parent.ssa_outboard.get()
        mask &= Y > self.parent.ssa_lower.get()
        mask &= Y < self.parent.ssa_upper.get()
        mask = sp.ndimage.gaussian_filter(mask.astype(float), sigma=1)

        image = beam * mask

        if noise:
            kx = np.fft.fftfreq(n=len(x), d=0.1)
            ky = np.fft.fftfreq(n=len(y), d=0.1)
            KX, KY = np.meshgrid(kx, ky)

            power_spectrum = 1 / (1e-2 + KX**2 + KY**2)

            white_noise = 1e-3 * np.random.standard_normal(size=X.shape)
            pink_noise = 1e-3 * np.real(np.fft.ifft2(power_spectrum * np.fft.fft2(np.random.standard_normal(size=X.shape))))
            # background = 5e-3 * (X - Y) / X.max()

            image += white_noise + pink_noise

        return image

    def generate_beam_xrt(self, noise: bool = True):
        R2 = self.parent.kbh_dsh.get()
        R1 = self.parent.kbv_dsv.get()

        self.beamLine.toroidMirror01.R = R1
        self.beamLine.toroidMirror02.R = R2
        outDict = run_process(self.beamLine)
        lb = outDict["screen01beamLocal01"]

        hist2d, _, _ = build_histRGB(lb, lb, limits=self.limits, isScreen=True, shape=[400, 300])
        image = hist2d
        _ = np.max(image)
        image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))
        self.counter += 1

        return image

    def generate_beam(self, *args, **kwargs):
        return self.generate_beam_xrt(*args, **kwargs)


class BeamlineEpics(Device):
    det = Cpt(xrtEpicsScreen, name="DetectorScreen")

    kbh_ush = Cpt(Signal, kind="hinted")
    kbh_dsh = Cpt(EpicsSignal, ":TM_HOR:R", kind="hinted")
    kbv_usv = Cpt(Signal, kind="hinted")
    kbv_dsv = Cpt(EpicsSignal, ":TM_VERT:R", kind="hinted")

    ssa_inboard = Cpt(Signal, value=-5.0, kind="hinted")
    ssa_outboard = Cpt(Signal, value=5.0, kind="hinted")
    ssa_lower = Cpt(Signal, value=-5.0, kind="hinted")
    ssa_upper = Cpt(Signal, value=5.0, kind="hinted")

    def __init__(self, *args, **kwargs):
        self.beamline = build_beamline()
        super().__init__(*args, **kwargs)


class Beamline(Device):
    det = Cpt(Detector)

    kbh_ush = Cpt(Signal, kind="hinted")
    kbh_dsh = Cpt(Signal, kind="hinted")
    kbv_usv = Cpt(Signal, kind="hinted")
    kbv_dsv = Cpt(Signal, kind="hinted")

    ssa_inboard = Cpt(Signal, value=-5.0, kind="hinted")
    ssa_outboard = Cpt(Signal, value=5.0, kind="hinted")
    ssa_lower = Cpt(Signal, value=-5.0, kind="hinted")
    ssa_upper = Cpt(Signal, value=5.0, kind="hinted")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
