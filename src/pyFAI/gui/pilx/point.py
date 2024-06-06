import h5py

from .models import ImageIndices
from .utils import (
    get_dataset,
    get_dataset_name,
    get_radial_dataset,
)


class Point:
    def __init__(
        self,
        indices: ImageIndices,
        file_name: str,
    ):
        self.indices = indices
        row = indices.row
        col = indices.col

        with h5py.File(file_name, "r") as h5file:
            radial_dset = get_radial_dataset(
                h5file, nxdata_path="/entry_0000/pyFAI/result"
            )
            self._radial_curve = radial_dset[()]
            self._x_name = get_dataset_name(radial_dset)
            intensity_dset = get_dataset(h5file, "/entry_0000/pyFAI/result/intensity")
            self._intensity_curve = intensity_dset[row, col, :]
            self._y_name = intensity_dset.attrs.get("long_name", "Intensity")

    def __repr__(self) -> str:
        return str(self.indices)

    def get_curve(self):
        return self._intensity_curve
