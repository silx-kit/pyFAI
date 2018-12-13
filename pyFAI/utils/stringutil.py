# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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


from __future__ import absolute_import, print_function, division

"""Module containing utilitary around string"""

__author__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/12/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import string


class SafeFormatter(string.Formatter):
    """Like default formater but unmatched keys are still present
    into the result string"""

    def get_field(self, field_name, args, kwargs):
        try:
            return super(SafeFormatter, self).get_field(field_name, args, kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            return "{%s}" % field_name, field_name


_safe_formater = SafeFormatter()


def safe_format(format_string, arguments):
    """Like default str.format but unmatching patterns will be
    still present into the result string.

    :param format_string str: Format string as defined in the default
        formatter.
    :param arguments dict or tuple: Arbitrary set of positional and keyword
        arguments.
    :rtype: str
    """
    if isinstance(arguments, dict):
        args = []
        kwargs = arguments
    else:
        args = arguments
        kwargs = {}
    return _safe_formater.vformat(format_string, args, kwargs)


def latex_to_unicode(string):
    """Returns a unicode representation from latex strings used by pyFAI.

    .. note:: The latex string could be removed from the pyFAI core.

    :param str string: A latex string to convert
    :rtype: str
    """
    string = string.replace("$", u"")
    string = string.replace("^{-2}", u"⁻²")
    string = string.replace("^{-1}", u"⁻¹")
    string = string.replace("^.", u"⋅")
    string = string.replace("2\\theta", u"2θ")
    string = string.replace("^{o}", u"°")
    string = string.replace("\\AA", u"Å")
    string = string.replace("log10", u"log₁₀")
    string = string.replace("^{*2}", u"d*²")
    return string
