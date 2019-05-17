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

"""This module inits matplotlib and setups the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/05/2019"


import sys
import logging

_logger = logging.getLogger(__name__)

_check_matplotlib = 'matplotlib' in sys.modules

from silx.gui import qt

import matplotlib


def _configure(backend, backend_qt4=None, check=False):
    if check:
        valid = matplotlib.rcParams['backend'] == backend
        if backend_qt4 is not None:
            valid = valid and matplotlib.rcParams['backend.qt4'] == backend_qt4

        if not valid:
            _logger.warning('matplotlib already loaded, setting its backend may not work')
        return
    matplotlib.rcParams['backend'] = backend
    if backend_qt4 is not None:
        matplotlib.rcParams['backend.qt4'] = backend_qt4


if qt.BINDING == 'PySide':
    _configure('Qt4Agg', 'PySide', check=_check_matplotlib)
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg  # noqa

elif qt.BINDING == 'PyQt4':
    _configure('Qt4Agg', check=_check_matplotlib)
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg  # noqa

elif qt.BINDING == 'PyQt5':
    _configure('Qt5Agg', check=_check_matplotlib)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa

from matplotlib import pyplot  # noqa
from matplotlib import pylab  # noqa
