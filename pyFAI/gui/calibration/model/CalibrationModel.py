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
__date__ = "22/08/2018"

from .AbstractModel import AbstractModel
from .PlotViewModel import PlotViewModel
from .ExperimentSettingsModel import ExperimentSettingsModel
from .PeakSelectionModel import PeakSelectionModel
from .GeometryModel import GeometryModel
from .GeometryConstraintsModel import GeometryConstraintsModel
from .IntegrationSettingsModel import IntegrationSettingsModel
from .MarkerModel import MarkerModel


class CalibrationModel(AbstractModel):

    def __init__(self, parent=None):
        super(CalibrationModel, self).__init__(parent)
        self.__experimentSettingsModel = ExperimentSettingsModel(self)
        self.__peakSelectionModel = PeakSelectionModel(self)
        self.__fittedGeometry = GeometryModel(self)
        self.__peakGeometry = GeometryModel(self)
        self.__geometryConstraintsModel = GeometryConstraintsModel(self)
        self.__integrationSettingsModel = IntegrationSettingsModel(self)
        self.__markerModel = MarkerModel(self)
        self.__rawPlotView = PlotViewModel(self)

    def isValid(self):
        return True

    def experimentSettingsModel(self):
        return self.__experimentSettingsModel

    def peakSelectionModel(self):
        return self.__peakSelectionModel

    def fittedGeometry(self):
        return self.__fittedGeometry

    def peakGeometry(self):
        return self.__peakGeometry

    def geometryConstraintsModel(self):
        return self.__geometryConstraintsModel

    def integrationSettingsModel(self):
        return self.__integrationSettingsModel

    def markerModel(self):
        return self.__markerModel

    def rawPlotView(self):
        """Store definition of the RAW data view.

        This view is shared by some plots
        """
        return self.__rawPlotView
