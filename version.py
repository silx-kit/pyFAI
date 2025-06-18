#!/usr/bin/env python3
# coding: utf-8
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
"""Unique place where the version number is defined.

provides:
* version = "1.2.3" or "1.2.3-beta4"
* version_info = named tuple (1,2,3,"beta",4)
* hexversion: 0x010203B4
* strictversion = "1.2.3b4
* debianversion = "1.2.3~beta4"
* calc_hexversion: the function to transform a version_tuple into an integer

This is called hexversion since it only really looks meaningful when viewed as the
result of passing it to the built-in hex() function.
The version_info value may be used for a more human-friendly encoding of the same information.

The hexversion is a 32-bit number with the following layout:
Bits (big endian order)     Meaning
1-8     PY_MAJOR_VERSION (the 2 in 2.1.0a3)
9-16     PY_MINOR_VERSION (the 1 in 2.1.0a3)
17-24     PY_MICRO_VERSION (the 0 in 2.1.0a3)
25-28     PY_RELEASE_LEVEL (0xA for alpha, 0xB for beta, 0xC for release candidate and 0xF for final)
29-32     PY_RELEASE_SERIAL (the 3 in 2.1.0a3, zero for final releases)

Thus 2.1.0a3 is hexversion 0x020100a3.

"""

__authors__ = ["Jérôme Kieffer", "V. Valls"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/06/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'
__all__ = ["date", "version_info", "strictversion", "hexversion", "debianversion",
           "calc_hexversion", "citation"]

RELEASE_LEVEL_VALUE = {"dev": 0,
                       "alpha": 10,
                       "beta": 11,
                       "gamma": 12,
                       "rc": 13,
                       "final": 15}

MAJOR = 2025
MINOR = 6
MICRO = 0
RELEV = "dev"  # <16
SERIAL = 0  # <16

date = __date__

from collections import namedtuple

_version_info = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])

version_info = _version_info(MAJOR, MINOR, MICRO, RELEV, SERIAL)

strictversion = version = debianversion = "%d.%d.%d" % version_info[:3]
if version_info.releaselevel != "final":
    version += "-%s%s" % version_info[-2:]
    debianversion += "~adev%i" % version_info[-1] if RELEV == "dev" else "~%s%i" % version_info[-2:]
    prerel = "a" if RELEASE_LEVEL_VALUE.get(version_info[3], 0) < 10 else "b"
    if prerel not in "ab":
        prerel = "a"
    strictversion += prerel + str(version_info[-1])

_PATTERN = None


def calc_hexversion(major=0, minor=0, micro=0, releaselevel="dev", serial=0, string=None):
    """Calculate the hexadecimal version number from the tuple version_info:

    :param major: integer
    :param minor: integer
    :param micro: integer
    :param relev: integer or string
    :param serial: integer
    :param string: version number as a string
    :return: integer always increasing with revision numbers
    """
    if string is not None:
        global _PATTERN
        if _PATTERN is None:
            import re
            _PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)(\w+)?$")
        result = _PATTERN.match(string)
        if result is None:
            raise ValueError("'%s' is not a valid version" % string)
        result = result.groups()
        major, minor, micro = int(result[0]), int(result[1]), int(result[2])
        releaselevel = result[3]
        if releaselevel is None:
            releaselevel = 0

    try:
        releaselevel = int(releaselevel)
    except ValueError:
        releaselevel = RELEASE_LEVEL_VALUE.get(releaselevel, 0)

    hex_version = int(serial)
    hex_version |= releaselevel * 1 << 4
    hex_version |= int(micro) * 1 << 8
    hex_version |= int(minor) * 1 << 16
    hex_version |= int(major) * 1 << 24
    return hex_version


hexversion = calc_hexversion(*version_info)

citation = "doi:10.1107/S1600576715004306"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(usage="print the version of the software")
    parser.add_argument("--wheel", action="store_true", dest="wheel", default=None,
                        help="print version formated for wheel")
    parser.add_argument("--debian", action="store_true", dest="debian", default=None,
                        help="print version formated for debian")
    options = parser.parse_args()
    if options.debian:
        print(debianversion)
    elif options.wheel:
        print(strictversion)
    else:
        print(version)
