import h5py  # type: ignore[import-untyped]
import numpy as np
from area_detector_handlers.handlers import HandlerBase  # type: ignore[import-untyped]
from ophyd import Signal  # type: ignore[import-untyped]


class HDF5Handler(HandlerBase):
    specs = {"HDF5"}

    def __init__(self, filename):
        self._name = filename

    def __call__(self, frame):
        with h5py.File(self._name, "r") as f:
            entry = f["/entry/image"]
            return entry[frame]


class ExternalFileReference(Signal):
    """
    A pure software Signal that describe()s an image in an external file.
    """

    def describe(self):
        descriptor_data = super().describe()
        descriptor_data[self.name].update(
            {"shape": (300, 400), "external": "STREAM:", "dtype": "array", "dtype_numpy": np.dtype(np.float64).str}
        )
        return descriptor_data
