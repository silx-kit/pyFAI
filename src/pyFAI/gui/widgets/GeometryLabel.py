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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/11/2022"

from typing import Optional
from ..model.GeometryModel import GeometryModel
from silx.gui.widgets.ElidedLabel import ElidedLabel


class GeometryLabel(ElidedLabel):
    """Read-only widget to display a :class:`GeometryModel`.

    - a detector holder (see :meth:`setDetector`, :meth:`detector`)
    - a view on top of a model (see :meth:`setDetectorModel`, :meth:`detectorModel`)
    """

    def __init__(self, parent=None):
        super(GeometryLabel, self).__init__(parent)
        self.__geometry: Optional[GeometryModel] = None
        self.__updateDisplay()
        self.setTextAsToolTip(False)

    def __updateDisplay(self):
        geometry = self.__geometry
        if geometry is None:
            self.setText("No geometry")
            self.setToolTip("")
            return

        args = {
            "distance": geometry.distance().value(),
            "poni1": geometry.poni1().value(),
            "poni2": geometry.poni2().value(),
            "rotation1": geometry.rotation1().value(),
            "rotation2": geometry.rotation2().value(),
            "rotation3": geometry.rotation3().value(),
        }

        if set(args.values()) == set([None]):
            self.setText("No geometry")
            self.setToolTip("")
            return

        tipTemplate = """<html>
            <ul style="margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 0">
            <li style="white-space:pre"><b>Distance</b>: {distance} m</li>
            <li style="white-space:pre"><b>PONI1</b>: {poni1} m</li>
            <li style="white-space:pre"><b>PONI2</b>: {poni2} m</li>
            <li style="white-space:pre"><b>Rotation 1</b>: {rotation1} rad</li>
            <li style="white-space:pre"><b>Rotation 2</b>: {rotation2} rad</li>
            <li style="white-space:pre"><b>Rotation 3</b>: {rotation3} rad</li>
            </ul>
        </html>"""

        labelTemplate = "Distance: {distance}; PONIs: {poni1},{poni2}; Rotations: {rotation1},{rotation2},{rotation3}"

        self.setText(labelTemplate.format(**args))
        self.setToolTip(tipTemplate.format(**args))

    def setGeometryModel(self, geometryModel: Optional[GeometryModel]):
        """Set the geometry to display.

        :param ~pyFAI.gui.model.GeometryModel geometryModel: A geometry.
        """
        if not isinstance(geometryModel, GeometryModel):
            raise RuntimeError("geometryModel is not a GeometryModel")
        if self.__geometry is geometryModel:
            return
        if self.__geometry is not None:
            self.__geometry.changed.disconnect(self.__updateDisplay)
        self.__geometry = geometryModel
        if self.__geometry is not None:
            self.__geometry.changed.connect(self.__updateDisplay)

    def geometryModel(self) -> Optional[GeometryModel]:
        """Returns the geometry model

        :rtype: Union[None,~pyFAI.gui.model.GeometryModel]
        """
        return self.__geometry
