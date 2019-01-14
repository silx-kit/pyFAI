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
__date__ = "14/01/2019"

import logging
_logger = logging.getLogger(__name__)

from silx.gui import qt

from ... import method_registry


class MethodLabel(qt.QLabel):
    """Readonly line display"""

    _HUMAN_READABLE = {
        "*": "Any",
        "any": "Any",
        "all": "Any",
        "no": "No splitting",
        "bbox": "Bounding box",
        "pseudo": "Pseudo split",
        "full": "Full splitting",
        "histogram": "Histogram",
        "lut": "LUT",
        "csr": "CSR",
        "python": "Python",
        "cython": "Cython",
        "opencl": "OpenCL",
    }

    _TOOLTIP_TEMPLATE = """<ul>
    <li><b>Pixel splitting:</b> {split}</li>
    <li><b>Implementation:</b> {impl}</li>
    <li><b>Algorithm:</b> {algo}</li>
    <li><b>Availability:</b> {availability}</li>
    </ul>"""

    def __init__(self, parent=None):
        super(MethodLabel, self).__init__(parent)
        self.__method = None
        self.__labelTemplate = "{split} / {impl} / {algo}"
        self.__availability = False
        self.__updateFeedback()

    def method(self):
        """
        :rtype: Union[None,method_registry.Method]
        """
        return self.__method

    def setMethod(self, method):
        if self.__method == method:
            return
        self.__method = method
        self.__updateFeedback()

    def setLabelTemplate(self, template):
        """Set the template used to format the label

        :param str template: The template used to format the label
        """
        self.__labelTemplate = template
        self.__updateFeedback()

    def labelTemplate(self):
        return self.__labelTemplate

    def __compare(self, method, methodReference):
        if method == methodReference:
            return "same"
        any_set = set(["*", "any", "all", None])
        if method.split != methodReference.split and methodReference.split not in any_set:
            return "degraded"
        if method.impl != methodReference.impl and methodReference.impl not in any_set:
            return "degraded"
        if method.algo != methodReference.algo and methodReference.algo not in any_set:
            return "degraded"
        return "specialized"

    def setMethodAvailability(self, availability):
        """Display or not in the widget if the method is available or not.

        :param bool availability: Display in the widget if the method is
            available or not.
        """
        self.__availability = availability
        self.__updateFeedback()

    def methodAvailability(self):
        return self.__availability

    def __updateFeedback(self):
        method = self.__method
        if method is None:
            label = "No method"
            toolTip = "No method"
        else:
            if not self.__availability:
                    label = self.__methodToString(method, self.__labelTemplate)
                    toolTip = "<html>%s</html>" % self.__methodToString(method, self._TOOLTIP_TEMPLATE)
            else:
                usedMethods = method_registry.IntegrationMethod.select_method(method=method)
                if len(usedMethods) == 0:
                    label = "No method fit"
                    toolTip = self.__methodToString(method, self._TOOLTIP_TEMPLATE)
                    toolTip = ("No method fit. Integration could be compromized. "
                               "The following configuration is defined:"
                               "%s</html>" % toolTip)
                else:
                    usedMethod = usedMethods[0]
                    usedMethod = usedMethod.method
                    compare = self.__compare(usedMethod, method)

                    if compare == "same":
                        label = self.__methodToString(method, self.__labelTemplate)
                        toolTip = "<html>%s</html>" % self.__methodToString(method, self._TOOLTIP_TEMPLATE)
                    else:
                        original = self.__methodToString(method, self.__labelTemplate)
                        label = self.__methodToString(usedMethod, self.__labelTemplate)
                        toolTip = self.__methodToString(usedMethod, self._TOOLTIP_TEMPLATE)

                        if compare == "degraded":
                            label = "Degraded to: " + label
                            toolTip = ("<html>The method %s is not available, at least, in this computer. "
                                       "The following method will be used:"
                                       "%s</html>" % (original, toolTip))
                        elif compare == "specialized":
                            label = "Specialized with: " + label
                            toolTip = ("<html>The generic selection %s will use the following method in this computer:"
                                       "%s</html>" % (original, toolTip))
                        else:
                            assert(False)

        self.setText(label)
        self.setToolTip(toolTip)

    def __methodToString(self, method, template):
        _dim, split, algo, impl, _target = method

        methods = method_registry.IntegrationMethod.select_method(dim="*",
                                                                  split=split,
                                                                  algo=algo,
                                                                  impl=impl,
                                                                  degradable=False)
        dimensions = set([m.dimension for m in methods])

        if dimensions == set([1, 2]):
            availability = "1D and 2D"
        elif dimensions == set([1]):
            availability = "Only 1D"
        elif dimensions == set([2]):
            availability = "Only 2D"
        elif dimensions == set([]):
            availability = "Not available"
        else:
            _logger.error("Unexpected dimensions %s", dimensions)
            availability = "Unsupported"

        split = self._HUMAN_READABLE.get(split, split)
        algo = self._HUMAN_READABLE.get(algo, algo)
        impl = self._HUMAN_READABLE.get(impl, impl)
        return template.format(split=split,
                               algo=algo,
                               impl=impl,
                               availability=availability)
