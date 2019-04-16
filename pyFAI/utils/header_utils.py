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

"""This modules contains helper function relative to image header.
"""

from __future__ import division, print_function

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/03/2019"


import logging
import fabio


_logger = logging.getLogger(__name__)


class MonitorNotFound(Exception):
    """Raised when monitor information in not found or is not valid."""
    pass


def _get_monitor_value_from_edf(image, monitor_key):
    """Return the monitor value from an EDF image using an header key.

    Take care of the counter and motor syntax using for example 'counter/bmon'
    which reach 'bmon' value from 'counter_pos' key using index from
    'counter_mne' key.

    :param fabio.fabioimage.FabioImage image: Image containing the header
    :param str monitor_key: Key containing the monitor
    :return: returns the monitor else raise a MonitorNotFound
    :rtype: float
    :raise MonitorNotFound: when the expected monitor is not found on the
        header
    """
    keys = image.header

    if "/" in monitor_key:
        base_key, mnemonic = monitor_key.split('/', 1)

        mnemonic_values_key = base_key + "_mne"
        mnemonic_values = keys.get(mnemonic_values_key, None)
        if mnemonic_values is None:
            raise MonitorNotFound("Monitor mnemonic key '%s' not found in the header" % (mnemonic_values_key))

        mnemonic_values = mnemonic_values.split()
        pos_values_key = base_key + "_pos"
        pos_values = keys.get(pos_values_key)
        if pos_values is None:
            raise MonitorNotFound("Monitor pos key '%s' not found in the header" % (pos_values_key))

        pos_values = pos_values.split()

        try:
            index = mnemonic_values.index(mnemonic)
        except ValueError:
            _logger.debug("Exception", exc_info=1)
            raise MonitorNotFound("Monitor mnemonic '%s' not found in the header key '%s'" % (mnemonic, mnemonic_values_key))

        if index >= len(pos_values):
            raise MonitorNotFound("Monitor value '%s' not found in '%s'. Not enougth values." % (pos_values_key))

        monitor = pos_values[index]

    else:
        if monitor_key not in keys:
            raise MonitorNotFound("Monitor key '%s' not found in the header" % (monitor_key))
        monitor = keys[monitor_key]

    try:
        monitor = float(monitor)
    except ValueError as _e:
        _logger.debug("Exception", exc_info=1)
        raise MonitorNotFound("Monitor value '%s' is not valid" % (monitor))
    return monitor


def _get_monitor_value_from_hdf5(image, monitor_key):
    """Return the monitor value from an HDF5 image using an header key.

    The monotor_key is a path from the image path containing:

    - A dataset containing a scalar (a constant monitor)
    - A dataset containing a vector of values (it must containes enougth values
        than the amount of frames)

    :param fabio.fabioimage.FabioImage image: Image containing the header
    :param str monitor_key: Key identify the path of the monitor
    :return: returns the monitor else raise a MonitorNotFound
    :rtype: float
    :raise MonitorNotFound: when the expected monitor is not found on the
        header
    """
    if monitor_key not in image.hdf5:
        raise MonitorNotFound("Monitor path '%s' not found" % (monitor_key))

    monitor_dataset = image.hdf5[monitor_key]
    if not hasattr(monitor_dataset, "dtype"):
        raise MonitorNotFound("Monitor path '%s' is not a datatset" % (monitor_key))

    if monitor_dataset.dtype.kind not in "fiu":
        raise MonitorNotFound("Monitor path '%s' does not contain a numerical value" % (monitor_key))

    if monitor_dataset.shape == tuple():
        # A constant monitor
        return monitor_dataset[()]

    if len(monitor_dataset.shape) != 1:
        raise MonitorNotFound("Monitor path '%s' expect a vector of values" % (monitor_key))

    if image.currentframe >= monitor_dataset.size:
        raise MonitorNotFound("Monitor path '%s' does not provide enougth values" % (monitor_key))

    return monitor_dataset[image.currentframe]


def get_monitor_value(image, monitor_key):
    """Return the monitor value from an image using an header key.

    :param fabio.fabioimage.FabioImage image: Image containing the header
    :param str monitor_key: Key containing the monitor
    :return: returns the monitor else raise an exception
    :rtype: float
    :raise MonitorNotFound: when the expected monitor is not found on the
        header
    """
    if monitor_key is None:
        return Exception("No monitor defined")

    if fabio.version_info[0:2] < (0, 9):
        # FIXME: Remove this dead code by upgrading the dependancy to fabio>=0.9
        if isinstance(image, fabio.edfimage.EdfImage):
            return _get_monitor_value_from_edf(image, monitor_key)
        elif isinstance(image, fabio.numpyimage.NumpyImage):
            return _get_monitor_value_from_edf(image, monitor_key)
        elif isinstance(image, fabio.hdf5image.Hdf5Image):
            return _get_monitor_value_from_hdf5(image, monitor_key)
    else:
        if isinstance(image, (fabio.edfimage.EdfImage, fabio.edfimage.EdfFrame)):
            return _get_monitor_value_from_edf(image, monitor_key)
        elif isinstance(image, fabio.numpyimage.NumpyImage):
            return _get_monitor_value_from_edf(image, monitor_key)
        elif isinstance(image, (fabio.hdf5image.Hdf5Image, fabio.hdf5image.Hdf5Frame)):
            return _get_monitor_value_from_hdf5(image, monitor_key)

    raise Exception("File format '%s' unsupported" % type(image))
