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
__date__ = "14/12/2018"

from silx.gui import qt

try:
    import pyopencl
except ImportError:
    pyopencl = None

from pyFAI.utils import get_ui_file


class OpenClDeviceDialog(qt.QDialog):
    """Dialog to select an OpenCl device. It could be both select an available
    device on this machine or a custom one using indexes, or some types of
    requested devices.

    This dialog do not expect PyOpenCL to installed.
    """

    def __init__(self, parent=None):
        super(OpenClDeviceDialog, self).__init__(parent)
        filename = get_ui_file("opencl-device-dialog.ui")
        qt.loadUi(filename, self)

        self.__availableIds = {}
        model = self.__createModel()
        self._deviceView.setModel(model)
        self._deviceView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self._deviceView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)

        header = self._deviceView.horizontalHeader()
        if qt.qVersion() < "5.0":
            header.setClickable(True)
            header.setMovable(True)
            header.setSectionResizeMode = self.setResizeMode
        else:
            header.setSectionsClickable(True)
            header.setSectionsMovable(True)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        header.setSectionResizeMode(1, qt.QHeaderView.Interactive)
        header.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)

        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)

        self._group = qt.QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.addButton(self._anyDeviceButton)
        self._group.addButton(self._anyGpuButton)
        self._group.addButton(self._anyCpuButton)
        self._group.addButton(self._availableButton)
        self._group.addButton(self._customButton)
        self._group.buttonClicked.connect(self.__modeChanged)
        self._anyDeviceButton.setChecked(True)
        self.__modeChanged()

        if pyopencl is None:
            self._availableButton.setEnabled(False)
            self._availableButton.setToolTip("PyOpenCL have to be installed to display available devices.")

    def showEvent(self, event):
        result = qt.QDialog.showEvent(self, event)
        # TODO: It would be nice to resize in a smooth way when we resize the dialog
        # It was in Stretch mode until the redisplay to have a good first layout
        header = self._deviceView.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.Interactive)
        return result

    def __modeChanged(self):
        enabled = self._availableButton.isChecked()
        self._deviceView.setEnabled(enabled)
        enabled = self._customButton.isChecked()
        self._platformId.setEnabled(enabled)
        self._deviceId.setEnabled(enabled)

    def __createModel(self):
        model = qt.QStandardItemModel(self)
        model.setHorizontalHeaderLabels(["Device", "Platform", "Type", "PID", "DID"])
        if pyopencl is None:
            model.setColumnCount(5)
            return model

        for platformId, platform in enumerate(pyopencl.get_platforms()):
            for deviceId, device in enumerate(platform.get_devices(pyopencl.device_type.ALL)):
                self.__availableIds[(platformId, deviceId)] = model.rowCount()
                typeName = pyopencl.device_type.to_string(device.type)
                deviceName = qt.QStandardItem(device.name)
                platformName = qt.QStandardItem(platform.name)
                deviceType = qt.QStandardItem(typeName)
                deviceItem = qt.QStandardItem(str(deviceId))
                platformItem = qt.QStandardItem(str(platformId))
                model.appendRow([deviceName, platformName, deviceType, platformItem, deviceItem])

        return model

    def selectDevice(self, device):
        """Select an OpenCL device displayed on this dialog.

        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
        tuple containing the platform index and the device index.

        If this device is available on this platform is is selected in the
        list. Else it is selected as a custom indexes.

        :param Union[str,Tuple[int,int]] device: A device.
        """
        if device is None:
            self._anyDeviceButton.setChecked(True)
        elif device == "any":
            self._anyDeviceButton.setChecked(True)
        elif device == "gpu":
            self._anyGpuButton.setChecked(True)
        elif device == "cpu":
            self._anyCpuButton.setChecked(True)
        else:
            platformId, deviceId = device
            self._deviceId.setValue(deviceId)
            self._platformId.setValue(platformId)
            if device in self.__availableIds:
                self._availableButton.setChecked(True)
                index = self.__availableIds[device]
                model = self._deviceView.model()
                indexStart = model.index(index, 0)
                indexEnd = model.index(index, model.columnCount() - 1)
                selection = qt.QItemSelection(indexStart, indexEnd)
                selectionModel = self._deviceView.selectionModel()
                selectionModel.select(selection, qt.QItemSelectionModel.ClearAndSelect)
            else:
                self._customButton.setChecked(True)
            self.__modeChanged()

    def device(self):
        """Returns the selected OpenCL device.

        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
        tuple containing the platform index and the device index.

        :rtype: Union[str,Tuple[int,int]]
        :raises ValueError: If no devices are selected
        """
        if self._anyDeviceButton.isChecked():
            return "any"
        if self._anyGpuButton.isChecked():
            return "gpu"
        if self._anyCpuButton.isChecked():
            return "cpu"
        if self._availableButton.isChecked():
            selectionModel = self._deviceView.selectionModel()
            index = selectionModel.currentIndex()
            index = index.row()
            for key, value in self.__availableIds.items():
                if value == index:
                    return key
            raise ValueError("No device selected")
        if self._customButton.isChecked():
            return self._platformId.value(), self._deviceId.value()

        assert(False)
