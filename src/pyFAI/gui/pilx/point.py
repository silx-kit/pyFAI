import h5py
from .models import ImageIndices
from .utils import (
    get_dataset,
    get_dataset_name,
    get_radial_dataset,
    get_signal_dataset,
    get_axes_index
)


class Point:

    def __init__(self,
                 indices: ImageIndices,
                 url_nxdata_path: str):
        self.indices = indices
        row = indices.row
        col = indices.col
        file_name, nxdata_path = url_nxdata_path.split("?")

        with h5py.File(file_name, "r") as h5file:
            intensity_dset = get_signal_dataset(h5file, nxdata_path, default="intensity")
            axes_index = get_axes_index(intensity_dset)
            if axes_index.radial == 0:
                self._intensity_curve = intensity_dset[:, row, col]
            else:
                self._intensity_curve = intensity_dset[row, col,:]
            self._y_name = intensity_dset.attrs.get("long_name", "Intensity")
            radial_dset = get_radial_dataset(h5file,
                                             nxdata_path=nxdata_path,
                                             size=self._intensity_curve.size)
            self._radial_curve = radial_dset[()]
            self._x_name = get_dataset_name(radial_dset)

    def __repr__(self) -> str:
        return str(self.indices)

    def get_curve(self):
        return self._intensity_curve
