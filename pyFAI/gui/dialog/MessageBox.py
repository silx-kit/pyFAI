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
__date__ = "07/05/2019"

import logging
import sys

from silx.gui import qt


_logger = logging.getLogger(__name__)


def exception(parent, title, exc_info, logger=None):
    """
    Display an exception as a MessageBox

    :param str title: A context message (displayed a s a title)
    :param qt.QWidget parent: The parent widget
    :param Union[tuple,Exception] exc_info: An exception or the output of
        exc_info.
    :param object logger: Logger to record the error inside. If `None` a
        default logger is provided.
    """
    if logger is None:
        logger = _logger

    logger.error(title, exc_info=True)

    if isinstance(exc_info, BaseException):
        exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
    elif not isinstance(exc_info, tuple):
        exc_info = sys.exc_info()

    if exc_info[2] is not None:
        # Mimic the syntax of the default Python exception
        import traceback
        detailed = (''.join(traceback.format_tb(exc_info[2])))
        detailed = '{1}\nTraceback (most recent call last):\n{2}{0}: {1}'.format(exc_info[0].__name__, exc_info[1], detailed)
    else:
        # There is no backtrace
        detailed = '{0}: {1}'.format(exc_info[0].__name__, exc_info[1])

    msg = qt.QMessageBox(parent=parent)
    msg.setWindowTitle(title)
    msg.setIcon(qt.QMessageBox.Critical)
    msg.setInformativeText("%s" % exc_info[1])
    msg.setDetailedText(detailed)

    msg.raise_()
    msg.exec_()
