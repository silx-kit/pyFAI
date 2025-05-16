# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2025 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/05/2025"

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
            msg = f'Matplotlib already loaded with backend `{matplotlib.rcParams["backend"]}`, setting its backend to `{backend}` may not work!'
            _logger.warning(msg)
        return
    matplotlib.rcParams['backend'] = backend
    if backend_qt4 is not None:
        matplotlib.rcParams['backend.qt4'] = backend_qt4

if qt.BINDING in ('PySide', 'PyQt4'):
    _configure('Qt4Agg', qt.BINDING, check=_check_matplotlib)
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg  # noqa

elif qt.BINDING in ('PyQt6', 'PySide6', 'PyQt5', 'PySide2'):
    _configure('QtAgg', check=_check_matplotlib)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa

from matplotlib import pyplot  # noqa
from matplotlib import pylab  # noqa
from matplotlib import colors

#differs from the silx one (no normalization)
DEFAULT_MPL_COLORMAP = colors.Colormap(name="inferno")
