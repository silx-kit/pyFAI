# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "02/03/2017"

import logging
from pyFAI.gui import qt
import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

import silx.gui.plot

_logger = logging.getLogger(__name__)


class IntegrationTask(AbstractCalibrationTask):

    def __init__(self):
        super(IntegrationTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-result.ui"), self)

        self.__plot1d = silx.gui.plot.Plot1D(self)
        self.__plot2d = silx.gui.plot.Plot2D(self)
        colormap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        self.__plot2d.setDefaultColormap(colormap)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.__plot2d)
        layout.addWidget(self.__plot1d)
        self._integrateButton.clicked.connect(self.__invalidateIntegration)

    def __invalidateIntegration(self):
        self.__integrate()

    def __integrate(self):
        image = self.model().experimentSettingsModel().image().value()
        if image is None:
            return
        mask = self.model().experimentSettingsModel().mask().value()
        detector = self.model().experimentSettingsModel().detectorModel().detector()
        if detector is None:
            return
        geometry = self.model().fittedGeometry()
        if not geometry.isValid():
            return
        wavelength = geometry.wavelength().value()
        wavelength = wavelength / 1e10
        distance = geometry.distance().value()
        poni1 = geometry.poni1().value()
        poni2 = geometry.poni2().value()
        rotation1 = geometry.rotation1().value()
        rotation2 = geometry.rotation2().value()
        rotation3 = geometry.rotation3().value()

        ai = AzimuthalIntegrator(
            dist=distance,
            poni1=poni1,
            poni2=poni2,
            rot1=rotation1,
            rot2=rotation2,
            rot3=rotation3,
            detector=detector,
            wavelength=wavelength)

        unit = pyFAI.units.to_unit("2th_deg")
        numberPoint1D = 1024
        numberPointRadial = 400
        numberPointAzimuthal = 360

        # FIXME polarization factor, error model, method

        result1d = ai.integrate1d(
            data=image,
            npt=numberPoint1D,
            unit=unit,
            mask=mask)

        result2d = ai.integrate2d(
            data=image,
            npt_rad=numberPointRadial,
            npt_azim=numberPointAzimuthal,
            unit=unit)

        # FIXME set axes
        self.__plot1d.addCurve(
            legend="result1d",
            x=result1d.radial,
            y=result1d.intensity)

        # FIXME Add vertical line for each used calibration ring
        # Assume that axes are linear
        origin = (result2d.radial[0], result2d.azimuthal[0])
        scaleX = (result2d.radial[-1] - result2d.radial[0]) / result2d.intensity.shape[0]
        scaleY = (result2d.azimuthal[-1] - result2d.azimuthal[0]) / result2d.intensity.shape[1]
        self.__plot2d.addImage(
            legend="result2d",
            data=result2d.intensity,
            origin=origin,
            scale=(scaleX, scaleY))

    def _updateModel(self, model):
        settings = model.experimentSettingsModel()
        settings.mask().changed.connect(self.__invalidateIntegration)
        settings.polarizationFactor().changed.connect(self.__invalidateIntegration)
        model.fittedGeometry().changed.connect(self.__invalidateIntegration)
        self._polarizationFactor.setModel(settings.polarizationFactor())
