from silx.gui.plot import PlotWidget
from silx.gui.plot.actions import PlotAction


from ..HorizontalRangeROI import HorizontalRangeROI
from ..models import ROI_COLOR


class RoiModeAction(PlotAction):
    def __init__(self, plot: PlotWidget, parent=None):
        super(RoiModeAction, self).__init__(
            plot,
            icon=HorizontalRangeROI.ICON,
            text="ROI mode",
            tooltip="Draw a ROI",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        # Listen to mode change
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        # Init the state
        self._modeChanged(None)

    def _modeChanged(self, source):
        modeDict = self.plot.getInteractiveMode()
        old = self.blockSignals(True)
        self.setChecked(modeDict["mode"] == "select-draw")
        self.blockSignals(old)

    def _actionTriggered(self, checked=False):
        plot = self.plot
        if plot is not None:
            plot.setInteractiveMode("select-draw", shape="rectangle", color=ROI_COLOR)
