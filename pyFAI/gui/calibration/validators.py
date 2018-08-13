# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "10/08/2018"

import logging
from silx.gui import qt

_logger = logging.getLogger(__name__)


class DoubleValidator(qt.QDoubleValidator):
    """
    Double validator with extra feature.

    The default locale used is not the default one. It uses locale C with
    RejectGroupSeparator option. This allows to have consistant rendering of
    double using dot separator without any comma.

    QLocale provides an API to support or not groups on numbers. Unfortunatly
    the default Qt QDoubleValidator do not filter out the group character in
    case the locale rejected it. This implementation reject the group character
    from the validation, and remove it from the fixup. Only if the locale is
    defined to reject it.
    """
    def __init__(self, parent):
        qt.QDoubleValidator.__init__(self, parent)
        locale = qt.QLocale(qt.QLocale.C)
        locale.setNumberOptions(qt.QLocale.RejectGroupSeparator)
        self.setLocale(locale)

    def validate(self, inputText, pos):
        """
        Reimplemented from `QDoubleValidator.validate`.

        :param str inputText: Text to validate
        :param int pos: Position of the cursor
        """
        locale = self.locale()
        if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
            if pos > 0:
                if inputText[pos - 1] == locale.groupSeparator():
                    # filter the group separator
                    inputText = inputText[pos - 1:] + inputText[pos:]
                    pos = pos - 1

        return super(DoubleValidator, self).validate(inputText, pos)

    def fixup(self, inputText):
        """
        Remove group characters from the input text if the locale is defined to
        do so.

        :param str inputText: Text to validate
        """
        locale = self.locale()
        if locale.numberOptions() == qt.QLocale.RejectGroupSeparator:
            inputText = input.replace(locale.groupSeparator(), "")
        return inputText

    def toValue(self, text):
        """Convert the input string into an interpreted value

        :param str text: Input string
        :rtype: Tuple[object,bool]
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        value, validated = self.locale().toDouble(text)
        return value, validated

    def toText(self, value):
        """Convert the input string into an interpreted value

        :param object value: Input object
        :rtype: str
        """
        return str(value)


class DoubleAndEmptyValidator(DoubleValidator):
    """
    Validate double values or empty string.
    """

    def validate(self, inputText, pos):
        """
        Reimplemented from `QDoubleValidator.validate`.

        Allow to provide an empty value.

        :param str inputText: Text to validate
        :param int pos: Position of the cursor
        """
        if inputText.strip() == "":
            # python API is not the same as C++ one
            return qt.QValidator.Acceptable, inputText, pos

        return super(DoubleAndEmptyValidator, self).validate(inputText, pos)

    def toValue(self, text):
        """Convert the input string into an interpreted value

        :param str text: Input string
        :rtype: Tuple[object,bool]
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        if text.strip() == "":
            return None, True
        return super(DoubleAndEmptyValidator, self).toValue(text)

    def toText(self, value):
        """Convert the input string into an interpreted value

        :param object value: Input object
        :rtype: str
        """
        if value is None:
            return ""
        return super(DoubleAndEmptyValidator, self).toText(value)


class IntegerAndEmptyValidator(qt.QIntValidator):
    """
    Validate double values or empty string.
    """

    def validate(self, inputText, pos):
        """
        Reimplemented from `QIntValidator.validate`.

        Allow to provide an empty value.

        :param str inputText: Text to validate
        :param int pos: Position of the cursor
        """
        if inputText.strip() == "":
            # python API is not the same as C++ one
            return qt.QValidator.Acceptable, inputText, pos

        return super(IntegerAndEmptyValidator, self).validate(inputText, pos)

    def toValue(self, text):
        """Convert the input string into an interpreted value

        :param str text: Input string
        :rtype: Tuple[object,bool]
        :returns: A tuple containing the resulting object and True if the
            string is valid
        """
        value, validated = self.locale().toInt(text)
        return value, validated

    def toText(self, value):
        """Convert the input string into an interpreted value

        :param object value: Input object
        :rtype: str
        """
        if value is None:
            return ""
        return str(value)
