# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["V. Valls", "J. Kieffer"]
__license__ = "MIT"
__date__ = "24/02/2023"

from silx.gui import qt
from ..model.GeometryModel import GeometryModel
from ..widgets.GeometryTabs import GeometryTabs


class GeometryDialog(qt.QDialog):
    """Dialog to display a selected geometry
    """

    def __init__(self, parent=None):
        super(GeometryDialog, self).__init__(parent)
        self.setWindowTitle("Sample stage geometry")
        self._geometryTabs = GeometryTabs(self)
        self._buttonBox = qt.QDialogButtonBox()
        layout = qt.QVBoxLayout()
        layout.addWidget(self._geometryTabs)
        layout.addWidget(self._buttonBox, 1)
        self.setLayout(layout)

        # Connect buttons
        self._buttonBox.rejected.connect(self.reject)
        self._buttonBox.accepted.connect(self.accept)

        # NOTE: All the buttons have to be create here.
        # Changing available buttons on the focus event create a segfault
        types = (qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel |
                 qt.QDialogButtonBox.Reset | qt.QDialogButtonBox.Close)
        self._buttonBox.setStandardButtons(types)
        resetButton = self._buttonBox.button(qt.QDialogButtonBox.Reset)
        resetButton.clicked.connect(self._geometryTabs.resetToOriginalGeometry)
        self._geometryTabs._geometry.changed.connect(self.__updateButtons)
        self.__updateButtons()

    def accept(self):
        self.__originalGeometry = None
        return qt.QDialog.accept(self)

    def reject(self):
        self.__originalGeometry = None
        return qt.QDialog.reject(self)

    def isReadOnly(self):
        """
        Returns True if the dialog is in read only.

        In read-only mode, the geometry is displayed, but the user can't edited
        it.

        By default, this returns false.

        :rtype: bool
        """
        return self._geometryTabs.isReadOnly()

    def setReadOnly(self, readOnly):
        """
        Enable or disable the read-only mode.

        :param bool readOnly: True to enable the read-only mode.
        """
        self._geometryTabs.setReadOnly(readOnly)

    def __updateButtons(self):
        """Update the state of the dialog's buttons"""
        haveChanges = self._geometryTabs.isDirty()
        existing = [qt.QDialogButtonBox.Ok, qt.QDialogButtonBox.Cancel, qt.QDialogButtonBox.Reset, qt.QDialogButtonBox.Close]
        if haveChanges:
            available = set([qt.QDialogButtonBox.Ok, qt.QDialogButtonBox.Cancel, qt.QDialogButtonBox.Reset])
        else:
            available = set([qt.QDialogButtonBox.Close])
        for buttonType in existing:
            button = self._buttonBox.button(buttonType)
            isVisible = buttonType in available
            button.setVisible(isVisible)

    def setDetector(self, detector):
        """Set the used detector.

        This information is needed to display the Fit2D geometry.
        """
        self._geometryTabs.setDetector(detector)

    def setGeometryModel(self, geometryModel):
        """Set the geometry to display.

        :param ~pyFAI.gui.model.GeometryModel geometryModel: A geometry.
        """
        assert(isinstance(geometryModel, GeometryModel))
        if self._geometryTabs._geometry is geometryModel:
            return
        self._geometryTabs._geometry.changed.disconnect(self.__updateButtons)
        self._geometryTabs.setGeometryModel(geometryModel)
        self._geometryTabs._geometry.changed.connect(self.__updateButtons)
        self.__updateButtons()

    def geometryModel(self):
        """Returns the geometry model

        :rtype: ~pyFAI.gui.model.GeometryModel
        """
        return self._geometryTabs._geometry
