#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
Module with miscelaneous tools
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/06/2025"
__status__ = "production"

import logging
import sys
import os
import glob
import threading
sem = threading.Semaphore()  # global lock for image processing initialization
import fabio

from ..version import calc_hexversion
if ("hexversion" in dir(fabio)) and (fabio.hexversion >= calc_hexversion(0, 2, 2)):
    from fabio.nexus import exists
else:
    from os.path import exists

logger = logging.getLogger(__name__)
from .. import resources
try:
    from ..directories import data_dir
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    data_dir = None

if sys.platform != "win32":
    WindowsError = RuntimeError

win32 = (os.name == "nt") and (tuple.__itemsize__ == 4)

StringTypes = (bytes, str)

try:
    from ..ext.fastcrc import crc32
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from zlib import crc32

# TODO: Added on 2018-01-01 for pyFAI 0.15
# Can be deprecated for the next release, and then removed
# Functions should be used directly from the mathutil module
from .mathutil import *


def calc_checksum(ary, safe=True):
    """
    Calculate the checksum by default (or returns its buffer location if unsafe)
    """
    if safe:
        return crc32(ary)
    else:
        return ary.__array_interface__['data'][0]


def float_(val):
    """
    Convert anything to a float ... or None if not applicable
    """
    try:
        f = float(str(val).strip())
    except ValueError:
        f = None
    return f


def int_(val):
    """
    Convert anything to an int ... or None if not applicable
    """
    try:
        f = int(str(val).strip())
    except ValueError:
        f = None
    return f


def str_(val):
    """
    Convert anything to a string ... but None -> ""
    """
    s = ""
    if val is not None:
        try:
            s = str(val)
        except UnicodeError:
            # Python2 specific...
            s = unicode(val)
    return s


def expand_args(args):
    """
    Takes an argv and expand it (under Windows, cmd does not convert ``*.tif``
    into a list of files.
    Keeps only valid files (thanks to glob)

    :param args: list of files or wilcards
    :return: list of actual args
    """
    new = []
    for afile in args:
        if exists(afile):
            new.append(afile)
        else:
            new += glob.glob(afile)
    return new


def _get_data_path(filename):
    """
    :param filename: the name of the requested data file.
    :type filename: str

    Can search root of data directory in:
    - Environment variable PYFAI_DATA
    - path hard coded into pyFAI.directories.data_dir
    - where this file is installed.

    In the future ....
    This method try to find the requested ui-name following the
    xfreedesktop recommendations. First the source directory then
    the system locations

    For now, just perform a recursive search
    """
    resources = [
        os.environ.get("PYFAI_DATA"),
        data_dir,
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    try:
        import xdg.BaseDirectory
        resources += xdg.BaseDirectory.load_data_paths("pyFAI")
    except ImportError:
        pass

    for resource in resources:
        if not resource:
            continue
        real_filename = os.path.join(resource, "resources", filename)
        if os.path.exists(real_filename):
            return real_filename
    else:
        raise RuntimeError("Can not find the [%s] resource, "
                           "something went wrong !!!" % (real_filename,))


def get_calibration_dir():
    """get the full path of a calibration directory

    :return: the full path of the calibrant file
    """
    return _get_data_path("calibration")


def get_cl_file(resource):
    """get the full path of a openCL resource file

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".
    See also :func:`silx.resources.register_resource_directory`.

    :param str resource: Resource name. File name contained if the `opencl`
        directory of the resources.
    :return: the full path of the openCL source file
    """
    if not resource.endswith(".cl"):
        resource += ".cl"
    s = resource.split(":")
    if (len(s) == 1):
        resource = "pyfai:" + resource
    return resources._resource_filename(resource,
                                        default_directory="opencl")


def get_ui_file(filename):
    """get the full path of a user-interface file

    :return: the full path of the ui
    """
    return _get_data_path(os.path.join("gui", filename))


class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.func_name if sys.version_info[0] < 3 else fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


def convert_CamelCase(name):
    """
    convert a function name in CamelCase into camel_case
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def readFloatFromKeyboard(text, dictVar):
    """
    Read float from the keyboard ....

    :param text: string to be displayed
    :param dictVar: dict of this type: {1: [set_dist_min],3: [set_dist_min, set_dist_guess, set_dist_max]}
    """
    fromkb = input(text).strip()
    try:
        vals = [float(i) for i in fromkb.split()]
    except ValueError:
        logging.error("Error in parsing values")
    else:
        found = False
        for i in dictVar:
            if len(vals) == i:
                found = True
                for j in range(i):
                    dictVar[i][j](vals[j])
        if not found:
            logger.error("You should provide the good number of floats")


class FixedParameters(set):
    """
    Like a set, made for FixedParameters in geometry refinement
    """

    def add_or_discard(self, key, value=True):
        """
        Add a value to a set if value, else discard it
        :param key: element to added or discared from set
        :type value: boolean. If None do nothing !
        :return: None
        """
        if value is None:
            return
        if value:
            self.add(key)
        else:
            self.discard(key)
    def __repr__(self):
        return f"Fixed parameters: {', '.join(self)}."

    def __iadd__(self, other):
        for i in other:
            self.add(i)
        return self

    def __add__(self, other):
        """enables the addition of a list"""
        new = self.__class__(self)
        new.__iadd__(other)
        return new


def fully_qualified_name(obj):
    "Return the fully qualified name of an object"
    actual_class = obj.__class__.__mro__[0]
    return actual_class.__module__ + "." + actual_class.__name__
