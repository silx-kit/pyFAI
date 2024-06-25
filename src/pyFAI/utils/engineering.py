# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for engineering notation formation"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2024"
__status__ = "development"
__docformat__ = 'restructuredtext'

import math

prefixes = {-7: "y",
            -6: "z",
            -5: "a",
            -4: "f",
            -3: "n",
            -2: "µ",
            -1:  "m",
             0: "",
             1: "k",
             2: "M",
             3: "G",
             4: "T",
             5: "P",
             6: "E",
             7: "Z"}

def eng_fmt(value, fmt=None, space=""):
    """Return an engineering notation for the numerical value

    :param value: the actual value
    :param fmt: the formating, for example "5.3f"
    :param space: can be used to insert a "_"  im 1_k
    :return: string
    """
    key = int(math.log(value, 10)//3)
    pfix = prefixes.get(key)
    if pfix is None:
        return str(value)
    else:
        value *= 10**(-3*key)
        if fmt:
            ffmt = "{value:%s}{space}{pfix}"%fmt
            return ffmt.format(value=value, space=space, pfix=pfix, fmt=fmt)
        else:
            return f"{value:f}".rstrip("0")+space+pfix
