# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module containing enhanced string formatters."""

__author__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/10/2025"
__status__ = "development"
__docformat__ = 'restructuredtext'

import string
import math

PREFIXES = {-7: "y",
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

class SafeFormatter(string.Formatter):
    """Like default formatter but unmatched keys are still present
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
    """Like default str.format but unmatched patterns will be
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
    string = string.replace("^{*2}", u"*²")
    return string


def to_scientific_unicode(value, digits=3):
    """Convert a float value into a string using scientific notation and
    superscript unicode character.

    This avoid using HTML in some case, when Qt widget does not support it.

    :param float value: Value to convert to displayable string
    :param int digits: Number of digits expected (`3` means `1.000`).
    """
    if math.isfinite(value):
        value = ("%%0.%de" % digits) % value
        value, power10 = value.split("e")
        power = ""
        for p in power10:
            if p == "-":
                power += u"\u207B"
            elif p == "+":
                power += u"\u207A"
            elif p == "1":
                power += u"\u00B9"
            elif p == "2":
                power += u"\u00B2"
            elif p == "3":
                power += u"\u00B3"
            else:
                v = ord(p) - ord("0")
                power += chr(0x2070 + v)
        value = value + u"\u00B710" + power
    else:
        value = str(value)
    return value


_TRUE_STRINGS = set(["yes", "true", "1"])
_FALSE_STRINGS = set(["no", "false", "0"])


def to_bool(string):
    """Returns a safe boolean from a string.

    :raise ValueError: If the string do not contains a boolean information.
    """
    lower = string.lower()
    if lower in _TRUE_STRINGS:
        return True
    if lower in _FALSE_STRINGS:
        return False
    raise ValueError("'%s' is not a valid boolean" % string)


_ordinal_suffix = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"]
"""Suffix for ordinal numbers from 0 to 9"""


def to_ordinal(number):
    """
    Returns a string from an ordinal value with it's suffix.

    :param int number: A number referring to a position
    :rtype: str
    """
    string = "%d" % number
    if len(string) >= 2 and string[-2] == "1":
        return string + "th"
    digit = ord(string[-1]) - ord("0")
    return string + _ordinal_suffix[digit]


def to_eng(value:float, fmt:str|None=None, space=""):
    """Return an engineering notation for the numerical value

    :param value: the actual value
    :param fmt: the formatting, for example "5.3f"
    :param space: can be used to insert a "_"  im 1_k
    :return: string
    """
    key = int(math.log10(value)//3)
    pfix = PREFIXES.get(key)
    if pfix is None:
        return str(value)
    else:
        value *= 10**(-3*key)
        if fmt:
            ffmt = "{value:%s}{space}{pfix}"%fmt
            return ffmt.format(value=value, space=space, pfix=pfix, fmt=fmt)
        else:
            string = f"{value:f}"
            if "." in string:
                string = string.rstrip("0")
                string = string.rstrip(".")
            return string + space + pfix


def from_eng(txt:str)-> float:
    """Convert a engineering notation to a floating point value"""
    value = ""
    for i in txt:
        if i.isdigit():
            value+=i
        elif i in " _":
            continue
        elif i in PREFIXES.values():
            imap = {v:k*3 for k,v in PREFIXES.items()}
            return float(value)*(10.0**imap[i])
        else:
            break
    return float(value)
