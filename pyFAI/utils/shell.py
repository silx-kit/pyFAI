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

"""Module containing utilities around shell"""

from __future__ import absolute_import, print_function, division

__author__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/12/2017"
__status__ = "development"
__docformat__ = 'restructuredtext'

import sys
import codecs


class ProgressBar:
    """
    Progress bar in shell mode
    """
    def __init__(self, title, max_value, bar_width):
        """
        Create a progress bar using a title, a maximum value and a graphical size.

        The display is done with stdout using carriage return to to hide the
        previous progress. It is not possible to use stdout for something else
        whill a progress bar is in use.

        The result looks like:

        .. code-block:: none

            Title [■■■■■■      ]  50%  Message

        :param title: Title displayed before the progress bar
        :type title: str
        :param max_value: The maximum value of the progress bar
        :type max_value: float
        :param bar_width: Size of the progressbar in the screen
        :type bar_width: int
        """
        self.title = title
        self.max_value = max_value
        self.bar_width = bar_width
        self.last_size = 0

        encoding = None
        if hasattr(sys.stdout, "encoding"):
            # sys.stdout.encoding can't be used in unittest context with some
            # configurations of TestRunner. It does not exists in Python2
            # StringIO and is None in Python3 StringIO.
            encoding = sys.stdout.encoding
        if encoding is None:
            # We uses the safer aproch: a valid ASCII character.
            self.progress_char = '#'
        else:
            try:
                self.progress_char = u'\u25A0'
                _byte = codecs.encode(self.progress_char, encoding)
            except (ValueError, TypeError, LookupError):
                # In case the char is not supported by the encoding,
                # or if the encoding does not exists
                self.progress_char = '#'

    def clear(self):
        """
        Remove the progress bar from the display and move the cursor
        at the beginning of the line using carriage return.
        """
        sys.stdout.write('\r' + " " * self.last_size + "\r")
        sys.stdout.flush()

    def update(self, value, message=""):
        """
        Update the progrss bar with the progress bar's current value.

        Set the progress bar's current value, compute the percentage
        of progress and update the screen with. Carriage return is used
        first and then the content of the progress bar. The cursor is
        at the begining of the line.

        :param value: progress bar's current value
        :type value: float
        :param message: message displayed after the progress bar
        :type message: str
        """
        coef = (1.0 * value) / self.max_value
        percent = round(coef * 100)
        bar_position = int(coef * self.bar_width)
        if bar_position > self.bar_width:
            bar_position = self.bar_width

        # line to display
        line = '\r%15s [%s%s] % 3d%%  %s' % (self.title, self.progress_char * bar_position, ' ' * (self.bar_width - bar_position), percent, message)

        # trailing to mask the previous message
        line_size = len(line)
        clean_size = self.last_size - line_size
        if clean_size < 0:
            clean_size = 0
        self.last_size = line_size

        sys.stdout.write(line + " " * clean_size + "\r")
        sys.stdout.flush()
