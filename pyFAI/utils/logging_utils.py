# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2019 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""This modules contains helper function relative to logging system.
"""

from __future__ import division, print_function

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/02/2019"


import logging
import contextlib


class PrePostEmitStreamHandler(logging.Handler):
    """Handler to allow to hook a function before and after the emit function.

    The main logging feature is delegated to a sub handler.
    """

    def __init__(self, handler):
        self._handler = handler

    def emit(self, record):
        """
        Call pre_emit function then delegate the emit to the sub handler.

        :type record: logging.LogRecord
        """
        self.pre_emit()
        self._handler.emit(record)
        self.post_emit()

    def __getattr__(self, attr):
        """Reach the attribute from the sub handler and cache it to the current
        object"""
        value = getattr(self._handler, attr)
        setattr(self, attr, value)
        return value

    def pre_emit(self):
        pass

    def post_emit(self):
        pass


def set_prepost_emit_callback(logger, pre_callback, post_callback):
    """Patch the logging system to have a working progress bar without glitch.
    pyFAI define a default handler then we have to rework it

    :return: The new handler
    """
    # assume there is a logger
    assert(len(logger.handlers) == 1)
    previous_handler = logger.handlers[0]
    logger.removeHandler(previous_handler)
    # use our custom handler
    handler = PrePostEmitStreamHandler(previous_handler)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if pre_callback:
        handler.pre_emit = pre_callback
    if post_callback:
        handler.post_emit = post_callback
    return handler


@contextlib.contextmanager
def prepost_emit_callback(logger, pre_callback, post_callback):
    """Context manager to add pre/post emit callback to a logger"""
    patched_handler = set_prepost_emit_callback(logger, pre_callback, post_callback)
    yield
    previous_handler = patched_handler._handler
    logger.removeHandler(patched_handler)
    # use the previous handler
    logger.addHandler(previous_handler)
    logger.setLevel(logging.INFO)
