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
__date__ = "08/01/2019"

from silx.gui import qt

from ... import method_registry


class MethodLabel(qt.QLabel):
    """Readonly line display"""

    _HUMAN_READABLE = {
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

    _LABEL_TEMPLATE = """{split} / {impl} / {algo}"""

    _TOOLTIP_TEMPLATE = """<ul>
    <li><b>Pixel splitting:</b> {split}</li>
    <li><b>Implementation:</b> {impl}</li>
    <li><b>Algorithm:</b> {algo}</li>
    </ul>"""

    def __init__(self, parent=None):
        super(MethodLabel, self).__init__(parent)
        self.__method = None
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

    def __updateFeedback(self):
        method = self.__method
        if method is None:
            label = "No method"
            toolTip = "No method"
        else:
            usedMethods = method_registry.IntegrationMethod.select_method(method=method)
            if len(usedMethods) == 0:
                label = "No method"
                toolTip = "No method fit. Integration could be compromized."
            else:
                usedMethod = usedMethods[0]
                usedMethod = usedMethod.method
                compare = self.__compare(usedMethod, method)

                if compare == "same":
                    label = self.__methodToString(method, self._LABEL_TEMPLATE)
                    toolTip = "<html>%s</html>" % self.__methodToString(method, self._TOOLTIP_TEMPLATE)
                else:
                    original = self.__methodToString(method, self._LABEL_TEMPLATE)
                    label = self.__methodToString(usedMethod, self._LABEL_TEMPLATE)
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
        split = self._HUMAN_READABLE.get(split, split)
        algo = self._HUMAN_READABLE.get(algo, algo)
        impl = self._HUMAN_READABLE.get(impl, impl)
        return template.format(split=split, algo=algo, impl=impl)
