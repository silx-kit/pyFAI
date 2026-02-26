# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2022-2022 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for exporting pyFAI object in JSON"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/02/2026"
__status__ = "production"
__docformat__ = 'restructuredtext'

import numpy
from .. import units
from json import JSONEncoder, dump, dumps


class PyFAIEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, units.Unit):
            return obj.name
        elif isinstance(obj, numpy.generic):
            return obj.item()
        JSONEncoder.default(self, obj)

UnitEncoder = PyFAIEncoder


def json_dump(*args, **kwargs):
    """Tailored `json.dump` function.
    See doc of json.dump
    """
    return dump(*args, cls=PyFAIEncoder, **kwargs)

def json_dumps(*args, **kwargs):
    """Tailored `json.dumps` function.
    See doc of json.dumps
    """
    return dumps(*args, cls=PyFAIEncoder, **kwargs)