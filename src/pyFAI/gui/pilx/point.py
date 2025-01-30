import h5py
from .models import ImageIndices
from .utils import (
    get_dataset,
    get_dataset_name,
    get_radial_dataset,
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
            radial_dset = get_radial_dataset(h5file,
                                             nxdata_path=nxdata_path)
            self._radial_curve = radial_dset[()]
            self._x_name = get_dataset_name(radial_dset)
            intensity_dset = get_dataset(h5file, f"{nxdata_path}/intensity")
            self._intensity_curve = intensity_dset[row, col,:]
            self._y_name = intensity_dset.attrs.get("long_name", "Intensity")

    def __repr__(self) -> str:
        return str(self.indices)

    def get_curve(self):
        return self._intensity_curve
