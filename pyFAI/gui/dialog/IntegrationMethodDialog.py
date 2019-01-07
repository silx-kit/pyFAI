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

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "07/01/2019"


from silx.gui import qt
from silx.gui import icons

from ... import method_registry
import pyFAI.utils


class IntegrationMethodWidget(qt.QWidget):
    """Label displaying a specific OpenCL device.
    """

    _HUMAN_READABLE = {
        "no": "No splitting",
        "bbox": "Bounding box",
        "pseudo": "Pseudo split",
        "full": "Full splitting",
        "histogram": "Histogram",
        "lut": "LUT",
        "csr": "CSR",
        "python": "Python",
        "cython": "Cython",
        "opencl": "OpenCL",
    }

    _IMAGE_DOC = {
        "no": "pyfai:gui/images/pixelsplitting-no",
        "bbox": "pyfai:gui/images/pixelsplitting-bbox",
        "pseudo": "pyfai:gui/images/pixelsplitting-pseudo",
        "full": "pyfai:gui/images/pixelsplitting-full",
    }

    _DESCRIPTION_DOC = {
        "no": "No pixel splitting. Each pixel is used in a single box of the result.",
        "bbox": "Split the bounding box corresponding to the pixel in the integrated geometry.",
        "pseudo": "Split an approximative bounding box corresponding to the pixel in the integrated geometry.",
        "full": "Split the pixel using a klinear approximation.",
        "histogram": "Preprocess the data using an histogram.",
        "lut": "Structure the data using a LUT (look-up table). Usually consuming less memory.",
        "csr": "Structure the data using a CSR (compressed sparse row). Usually faster for processing.",
        "python": "Use a pure Python/numpy implementation. Slow but portable.",
        "cython": "Use a Cython/C/C++ implementation. Fast but platform dependent.",
        "opencl": "Use an OpenCL implementation based on hardware acceleration using parallelization. Fastest but hardware/driver dependeant.",
    }

    CodeRole = qt.Qt.UserRole + 1

    sigMethodChanged = qt.Signal()

    def __init__(self, parent=None):
        super(IntegrationMethodWidget, self).__init__(parent)
        qt.loadUi(pyFAI.utils.get_ui_file("integration-method.ui"), self)

        self._implementationModel = self._createImplementationModel()
        self._implView.setModel(self._implementationModel)
        selection = self._implView.selectionModel()
        selection.selectionChanged.connect(self.__implementationChanged)
        self._implView.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self._algoModel = self._createAlgorithmModel()
        self._algoView.setModel(self._algoModel)
        selection = self._algoView.selectionModel()
        selection.selectionChanged.connect(self.__algorithmChanged)
        self._algoView.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self._splittingModel = self._createSplittingModel()
        self._splitView.setModel(self._splittingModel)
        selection = self._splitView.selectionModel()
        selection.selectionChanged.connect(self.__splittingChanged)
        self._splitView.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self._implView.entered.connect(self.__mouseEntered)
        self._implView.setMouseTracking(True)
        self._algoView.entered.connect(self.__mouseEntered)
        self._algoView.setMouseTracking(True)
        self._splitView.entered.connect(self.__mouseEntered)
        self._splitView.setMouseTracking(True)

        self.setTupleMethod(("*", "*", "*"))
        self.__updateFeedback()
        self.sigMethodChanged.connect(self.__updateFeedback)

    def __mouseEntered(self, index):
        code = index.data(self.CodeRole)
        if code == "*":
            qt.QToolTip.hideText()
        else:
            qt.QToolTip.showText(qt.QCursor.pos(),
                                 self.__createToolTip(code),
                                 self)

    def __createToolTip(self, code):
        image = self._IMAGE_DOC.get(code, None)
        if image is not None:
            try:
                image = icons.getQFile(image).fileName()
            except ValueError:
                image = "foo"
            template = """<html><table><tr><td valign="middle"><img src="{image}" /></td><td valign="middle">{description}</td></tr></html>"""
            print(image)
        else:
            template = "<html><table><tr><td>{description}</td></tr></html>"
        description = self._DESCRIPTION_DOC.get(code, "No description.")
        toolTip = template.format(image=image, description=description)
        return toolTip

    def _createAlgorithmModel(self):
        model = qt.QStandardItemModel(self)
        item = qt.QStandardItem("Any")
        item.setData("*", self.CodeRole)
        model.appendRow(item)
        for name in method_registry.IntegrationMethod.AVAILABLE_ALGOS:
            label = self._HUMAN_READABLE.get(name, name)
            item = qt.QStandardItem(label)
            item.setData(name, self.CodeRole)
            model.appendRow(item)
        return model

    def _createImplementationModel(self):
        model = qt.QStandardItemModel(self)
        item = qt.QStandardItem("Any")
        item.setData("*", self.CodeRole)
        model.appendRow(item)
        for name in method_registry.IntegrationMethod.AVAILABLE_IMPLS:
            label = self._HUMAN_READABLE.get(name, name)
            item = qt.QStandardItem(label)
            item.setData(name, self.CodeRole)
            model.appendRow(item)
        return model

    def _createSplittingModel(self):
        model = qt.QStandardItemModel(self)
        item = qt.QStandardItem("Any")
        item.setData("*", self.CodeRole)
        model.appendRow(item)
        for name in method_registry.IntegrationMethod.AVAILABLE_SLITS:
            label = self._HUMAN_READABLE.get(name, name)
            item = qt.QStandardItem(label)
            item.setData(name, self.CodeRole)
            model.appendRow(item)
        return model

    def __implementationChanged(self):
        self.__updateFeedback()

    def __algorithmChanged(self):
        self.__updateFeedback()

    def __splittingChanged(self):
        self.__updateFeedback()

    def __indexFromCode(self, model, code):
        for i in range(model.rowCount()):
            index = model.index(i, 0)
            if index.data(self.CodeRole) == code:
                return index
        return qt.QModelIndex()

    def __selectCode(self, code, model, view):
        selection = view.selectionModel()
        index = self.__indexFromCode(model, code)
        if not index.isValid():
            # Create this code, as it looks to be provided somehow
            item = qt.QStandardItem(code)
            item.setData(code, self.CodeRole)
            model.appendRow(item)
            index = self.__indexFromCode(model, code)
        old = selection.blockSignals(True)
        selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
        selection.blockSignals(old)

    def setMethod(self, method):
        if isinstance(method, tuple):
            self.setTupleMethod(method)
        else:
            self.setStringMethod(method)

    def setStringMethod(self, method):
        methods1 = method_registry.IntegrationMethod.select_old_method(dim=1, old_method=method)
        methods2 = method_registry.IntegrationMethod.select_old_method(dim=2, old_method=method)
        methods = methods1 + methods2
        if len(methods) != 0:
            method = methods[0].method
            split = method.split
            algo = method.algo
            impl = method.impl
            self.setTupleMethod((split, algo, impl))

    def setTupleMethod(self, method):
        split, algo, impl = method
        self.__selectCode(split, self._splittingModel, self._splitView)
        self.__selectCode(impl, self._implementationModel, self._implView)
        self.__selectCode(algo, self._algoModel, self._algoView)
        self.__updateFeedback()

    def method(self):
        """
        Returns method as tuple of slit, algo and impl

        :rtype: Tuple[str,str,str]
        """
        index = self._implView.selectedIndexes()[0]
        impl = index.data(self.CodeRole)
        index = self._algoView.selectedIndexes()[0]
        algo = index.data(self.CodeRole)
        index = self._splitView.selectedIndexes()[0]
        split = index.data(self.CodeRole)
        return split, algo, impl

    def __updateFeedback(self):
        self.__updateItems()
        self.__updateMessage()

    def __updateItemsFromModel(self, codeList, model, method):
        for name in codeList:
            index = self.__indexFromCode(model, name)
            if not index.isValid():
                continue
            item = model.itemFromIndex(index)

            localMethod = (name if m is None else m for m in method)
            methods = method_registry.IntegrationMethod.select_method(1, *localMethod, degradable=False)
            available1d = len(methods) != 0
            methods = method_registry.IntegrationMethod.select_method(2, *localMethod, degradable=False)
            available2d = len(methods) != 0

            if available1d and available2d:
                color = qt.Qt.black
                label = self._HUMAN_READABLE.get(name, name)
            elif not available1d and not available2d:
                color = qt.Qt.grey
                label = self._HUMAN_READABLE.get(name, name)
            else:
                color = qt.Qt.red
                label = self._HUMAN_READABLE.get(name, name)
                if available1d:
                    label = "%s (only 1D)" % label
                elif available2d:
                    label = "%s (only 2D)" % label

            item.setForeground(qt.QBrush(color))
            item.setText(label)

    def __updateItems(self):
        method = self.method()
        split, algo, impl = method

        localMethod = None, algo, impl
        self.__updateItemsFromModel(method_registry.IntegrationMethod.AVAILABLE_SLITS,
                                    self._splittingModel,
                                    localMethod)
        localMethod = split, None, impl
        self.__updateItemsFromModel(method_registry.IntegrationMethod.AVAILABLE_ALGOS,
                                    self._algoModel,
                                    localMethod)
        localMethod = split, algo, None
        self.__updateItemsFromModel(method_registry.IntegrationMethod.AVAILABLE_IMPLS,
                                    self._implementationModel,
                                    localMethod)

    def __updateMessage(self):
        method = self.method()

        methods = method_registry.IntegrationMethod.select_method(1, *method, degradable=False)
        available1d = len(methods) != 0
        methods = method_registry.IntegrationMethod.select_method(2, *method, degradable=False)
        available2d = len(methods) != 0

        if available1d and available2d:
            message = ""
        elif not available1d:
            message = "Not available for 1D integration."
        elif not available2d:
            message = "Not available for 2D integration."
        else:
            message = "Not available for botrh 1D/2D integrations."

        self._error.setText(message)
        self._error.setVisible(message != "")


class IntegrationMethodDialog(qt.QDialog):

    def __init__(self, parent=None):
        super(IntegrationMethodDialog, self).__init__(parent=parent)
        self.setWindowTitle("Method selection")

        self.__content = IntegrationMethodWidget(self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__content)

        buttonBox = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok |
                                        qt.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def selectMethod(self, method):
        """
        Select a detector.

        :param pyFAI.detectors.Detector detector: Detector to select in this
            dialog
        """
        self.__content.setMethod(method)

    def selectedMethod(self):
        """
        Returns the selected detector.

        :rtype: tuple
        """
        return self.__content.method()
