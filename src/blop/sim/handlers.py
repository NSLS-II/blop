import h5py  # type: ignore[import-untyped]
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
        resource_document_data = super().describe()
        resource_document_data[self.name].update(
            {
                "external": "FILESTORE:",
                "dtype": "array",
            }
        )
        return resource_document_data
