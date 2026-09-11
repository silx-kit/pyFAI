#!/usr/bin/env python
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2013-2020 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
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

"""Common for all applications within pyFAI"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "2020, European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/09/2026"
__status__ = "production"

import logging
import sys

logger = logging.getLogger(__name__)


def configure_console_output(encoding="utf-8", errors="backslashreplace"):
    r"""Make stdout and stderr able to print the unicode used by pyFAI.

    The representation of a geometry contains characters such as
    ``\N{GREEK SMALL LETTER LAMDA}``, ``\N{INFINITY}`` or
    ``\N{SUPERSCRIPT MINUS}``, which no Windows code page can encode. An
    interactive console is not affected, since Python writes to it through
    ``WriteConsoleW`` (PEP 528), but as soon as the output is redirected to a
    pipe or a file, Python falls back on the locale codec (cp1252) and
    ``print`` raises a `UnicodeEncodeError`.

    This is a no-op where the locale is already UTF-8, and Python 3.15 makes the
    UTF-8 mode the default (PEP 686), which will make this call useless.

    Only applications are expected to call this: re-encoding the streams of a
    process is not the business of a library.

    Note: this docstring is raw on purpose. The ``\N{...}`` above then stay
    literal instead of being expanded at compile time, which keeps `help()`
    printable on the very consoles this function works around.

    :param encoding: codec to write with, None to keep the current one
    :param errors: how to handle characters missing from the codec. The default
        degrades them to ``\uXXXX`` escapes instead of raising
    """
    for stream in (sys.stdout, sys.stderr):
        # `None` under pythonw.exe, StringIO when captured by a test runner
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding=encoding, errors=errors)
        except (OSError, ValueError):
            # The codec may be refused: at least stop raising on such characters
            try:
                stream.reconfigure(errors=errors)
            except (OSError, ValueError):
                logger.debug("Unable to reconfigure %s", stream, exc_info=True)


# Applied on import: every `pyFAI.app.*` module is a command line application
# and some of them are used as a library entry point, without going through main
configure_console_output()
