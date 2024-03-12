from silx.gui import qt
import silx.gui.icons
from silx.gui.dialog.DataFileDialog import DataFileDialog
from silx.io.url import DataUrl


class OpenAxisDatasetAction(qt.QAction):
    datasetOpened = qt.Signal(DataUrl)

    def __init__(self, parent=None):
        super().__init__(
            icon=silx.gui.icons.getQIcon("axis"),
            text="Open axis dataset",
            parent=parent,
        )
        self._file_directory = None
        self.setToolTip("Change axis values to motor positions")
        self.triggered[bool].connect(self._onTrigger)

    def _onTrigger(self):
        dialog = DataFileDialog(self.parentWidget())
        dialog.setWindowTitle("Open the dataset containing X values")
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingDataset)
        if self._file_directory is not None:
            dialog.setDirectory(self._file_directory)

        result = dialog.exec()
        if not result:
            return

        self.datasetOpened.emit(dialog.selectedDataUrl())

    def setFileDirectory(self, file_directory: str):
        self._file_directory = file_directory
