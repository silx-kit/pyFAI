from silx.gui import qt


class TitleWidget(qt.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("font: 16px;")
        self.setAlignment(qt.Qt.AlignmentFlag.AlignCenter)
