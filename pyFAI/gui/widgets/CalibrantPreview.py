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
__date__ = "16/05/2019"

import logging
import numpy
import os.path

from silx.gui import qt
from silx.gui.colors import Colormap

from ..utils import imageutils


_logger = logging.getLogger(__name__)


class CalibrantPreview(qt.QFrame):
    """
    CalibrantPreview show the rays of a calibrat at a wayvelength between 0
    and Pi.
    """

    _PIXMAP_OFFSET = 2

    def __init__(self, parent=None):
        super(CalibrantPreview, self).__init__(parent)
        self.__calibrant = None
        self.__waveLength = None
        self.__pixmap = None
        self.__cachedSize = None
        self.setMinimumSize(qt.QSize(50, 20))

    def setCalibrant(self, calibrant):
        if self.__calibrant is calibrant:
            return
        self.__calibrant = calibrant
        self.__pixmap = None
        self.__updateToolTip()
        self.repaint()

    def setWaveLength(self, waveLength):
        if self.__waveLength == waveLength:
            return
        self.__waveLength = waveLength
        self.__pixmap = None
        self.__updateToolTip()
        self.repaint()

    def getCalibrant(self):
        return self.__pixmap

    def __getConfiguredCalibrant(self):
        calibrant = self.__calibrant
        if calibrant is None:
            return None
        waveLenght = self.__waveLength
        if waveLenght is None:
            return None

        calibrant.setWavelength_change2th(waveLenght)
        return calibrant

    def __updateToolTip(self):
        calibrant = self.__getConfiguredCalibrant()
        if calibrant is None:
            return

        name = calibrant.filename
        if name is not None:
            name = os.path.basename(name)
            if name.endswith(".D"):
                name = name[0:-2]

        fileds = []
        if name is not None:
            fileds.append((u"Name", name, None))
        fileds.append((u"Nb registered rays", calibrant.count_registered_dSpacing(), None))
        dSpacing = calibrant.get_dSpacing()
        fileds.append((u"Nb visible rays", len(dSpacing), u"between 0 and 180°"))
        if len(dSpacing) > 0:
            ray = calibrant.get_dSpacing()[0]
            angle = calibrant.get_2th()[0]
            fileds.append((u"First visible ray", u"%f Å (%f°)" % (ray, numpy.rad2deg(angle)), None))
            ray = calibrant.get_dSpacing()[-1]
            angle = calibrant.get_2th()[-1]
            fileds.append((u"Last visible ray", u"%f Å (%f°)" % (ray, numpy.rad2deg(angle)), None))

        toolTip = []
        for f in fileds:
            field_name, field_value, suffix = f
            field = u"<li><b>%s</b>: %s</li>" % (field_name, field_value)
            if suffix is not None:
                field = u"%s (%s)" % (field, suffix)
            toolTip.append(field)

        toolTip = u"\n".join(toolTip)
        toolTip = u"<html><ul>%s</ul></html>" % toolTip
        self.setToolTip(toolTip)

    def __getPixmap(self, size=360):
        if self.__pixmap is not None and self.__cachedSize == size:
            return self.__pixmap
        calibrant = self.__getConfiguredCalibrant()
        if calibrant is None:
            return None
        tths = numpy.array(calibrant.get_2th())

        tth_min, tth_max = 0, numpy.pi
        histo = numpy.histogram(tths, bins=size, range=(tth_min, tth_max))
        agregation = histo[0].reshape(1, -1)
        colormap = Colormap(name="reversed gray", vmin=agregation.min(), vmax=agregation.max())
        rgbImage = colormap.applyToData(agregation)[:, :, :3]
        qimage = imageutils.convertArrayToQImage(rgbImage)
        qpixmap = qt.QPixmap.fromImage(qimage)
        self.__pixmap = qpixmap
        self.__cachedSize = size
        return self.__pixmap

    def paintEvent(self, event):
        super(CalibrantPreview, self).paintEvent(event)
        painter = qt.QPainter(self)

        # border
        option = qt.QStyleOptionProgressBar()
        option.initFrom(self)
        option.rect = self.rect()
        option.state = qt.QStyle.State_Enabled if self.isEnabled() else qt.QStyle.State_None
        style = qt.QApplication.style()
        style.drawControl(qt.QStyle.CE_ProgressBarGroove,
                          option,
                          painter,
                          self)

        # content
        pixmapRect = self.rect().adjusted(self._PIXMAP_OFFSET, self._PIXMAP_OFFSET,
                                          -self._PIXMAP_OFFSET, -self._PIXMAP_OFFSET)
        pixmap = self.__getPixmap(size=pixmapRect.width())
        if pixmap is not None:
            painter.drawPixmap(pixmapRect,
                               pixmap,
                               pixmap.rect())

    def sizeHint(self):
        return qt.QSize(200, self.minimumHeight())
