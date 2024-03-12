from __future__ import annotations
import numpy
from silx.gui.plot.items import ImageData


from .ImagePlotWidget import ImagePlotWidget
from ..models import ROI_COLOR, ImageIndices

_LEGEND = "IMAGE"


class DiffractionImagePlotWidget(ImagePlotWidget):
    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        image_item = self.addImage([[]], legend=_LEGEND)
        assert isinstance(image_item, ImageData)
        self._image_item = image_item
        self._first_plot = True

    def _dataConverter(self, x, y):
        image = self._image_item.getData(copy=False)
        indices = self.getImageIndices(x, y)
        if indices is None:
            return

        return image[indices.row, indices.col]

    def setImageData(self, image: numpy.ndarray):
        self._image_item.setData(image)
        if self._first_plot:
            self.resetZoom()
            self._first_plot = False

    def getImageIndices(self, x_data: float, y_data: float) -> ImageIndices | None:
        pixel_x, pixel_y = self.dataToPixel(x_data, y_data)
        picking_result = self._image_item.pick(pixel_x, pixel_y)
        if picking_result is None:
            return
        # Image dims are first rows then cols
        row_indices_array, col_indices_array = picking_result.getIndices(copy=False)
        return ImageIndices(row=row_indices_array[0], col=col_indices_array[0])

    def addContour(
        self, contour: numpy.ndarray, legend: str, linestyle: str | None = None
    ):
        self.addCurve(
            contour[:, 1],
            contour[:, 0],
            legend=legend,
            linestyle=linestyle,
            color=ROI_COLOR,
            resetzoom=False,
            selectable=False,
        )
