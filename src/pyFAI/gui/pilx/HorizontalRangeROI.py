from silx.gui.plot.items.roi import HorizontalRangeROI as SilxHorizontalRangeROI
from silx.gui import qt


class HorizontalRangeROI(SilxHorizontalRangeROI):
    """A HorizontalRangeROI that calls a custom signal each time the range changes"""

    # https://gitlab.esrf.fr/silx/pilx/-/merge_requests/19#note_290530
    sigRangeChanged = qt.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigEditingFinished.connect(self.sigRangeChanged)

    def setRange(self, vmin: float, vmax: float):
        super().setRange(vmin, vmax)
        self.sigRangeChanged.emit()
