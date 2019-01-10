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
__date__ = "04/01/2019"

from silx.gui import qt

try:
    import pyopencl
except ImportError:
    pyopencl = None


class OpenClDeviceLabel(qt.QLabel):
    """Label displaying a specific OpenCL device.
    """

    def __init__(self, parent=None):
        super(OpenClDeviceLabel, self).__init__(parent)
        self.__device = None
        self.__updateDisplay()

    def __getOpenClDevice(self, platformId, deviceId):
        if pyopencl is None:
            return None
        if not (0 <= platformId < len(pyopencl.get_platforms())):
            return None
        platform = pyopencl.get_platforms()[platformId]
        if not (0 <= deviceId < len(platform.get_devices())):
            return None
        return platform.get_devices()[deviceId]

    def __updateDisplay(self):
        toolTip = ""
        if self.__device is None:
            label = "No OpenCL device selected"
        elif self.__device == "any":
            label = "Any available device"
        elif self.__device == "cpu":
            label = "Any available CPU"
        elif self.__device == "gpu":
            label = "Any available GPU"
        else:
            platformId, deviceId = self.__device
            device = self.__getOpenClDevice(platformId, deviceId)
            if device is not None:
                deviceName, platformName = device.name, device.platform.name
            else:
                deviceName, platformName = None, None

            args = {
                "deviceName": deviceName,
                "deviceId": deviceId,
                "platformName": platformName,
                "platformId": platformId,
            }

            if deviceName is None:
                labelTemplate = "{deviceId}, {platformId}"
                tipTemplate = """
                    <ul>
                    <li><b>Platform index</b>: {platformId}</li>
                    <li><b>Device index</b>: {deviceId}</li>
                    </ul>
                """
            else:
                labelTemplate = "{deviceName} ({platformId}, {deviceId})"
                tipTemplate = """
                    <ul>
                    <li><b>Platform name</b>: {platformName}</li>
                    <li><b>Platform index</b>: {platformId}</li>
                    <li><b>Device name</b>: {deviceName}</li>
                    <li><b>Device index</b>: {deviceId}</li>
                    </ul>
                """

            label = labelTemplate.format(**args)
            toolTip = tipTemplate.format(**args)

        self.setText(label)
        self.setToolTip(toolTip)

    def setDevice(self, device):
        """Select an OpenCL device displayed on this dialog.

        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
        tuple containing the platform index and the device index.

        If this device is available on this platform is is selected in the
        list. Else it is selected as a custom indexes.

        :param Union[None,str,Tuple[int,int]] device: A device.
        """
        self.__device = device
        self.__updateDisplay()

    def device(self):
        """Returns the selected OpenCL device.

        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
        tuple containing the platform index and the device index.

        :rtype: Union[None,str,Tuple[int,int]]
        :raises ValueError: If no devices are selected
        """
        return self.__device
