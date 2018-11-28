# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Helper to read file and compute file
"""

from __future__ import division, print_function

__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__status__ = "production"


import numpy
try:
    import fabio
except ImportError:
    fabio = None
from pyFAI import io
from pyFAI import average


def average_files(files, method="mean"):
    """Average multiple files using the specified method.

    Returns a tuple containing the averaged data and a list of path used to
    compute this data.

    :param Union[str,List[str],None] files: file(s) used to compute the dark.
    :param str method: method used to compute the dark, "mean" or "median"
    :rtype: Tuple[numpy.array,List[str]]
    """
    if type(files) in io.StringTypes:
        files = [i.strip() for i in files.split(",")]
    elif not files:
        files = []
    if len(files) == 0:
        return None, []
    elif len(files) == 1:
        if fabio is None:
            raise RuntimeError("FabIO is missing")
        data = fabio.open(files[0]).data.astype(numpy.float32)
        source = files[0]
    else:
        data = average.average_images(files, filter_=method, fformat=None, threshold=0)
        source = "%s(%s)" % (method, ",".join(files))

    return data, source
