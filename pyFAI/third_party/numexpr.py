# coding: utf-8
# /*##########################################################################
# Copyright (C) 2023-2023 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "14/09/2023"

import logging
logger = logging.getLogger(__name__)
local = locals()
try:
    import numexpr as _numexpr
except ImportError as error:
    logger.errror("unable to import `numexp`")
    raise error
else:
    _version = tuple(int(i) for i in _numexpr.__version__.split(".")[:3] if i.isdigit())
    if _version < (2,8,6):
        for key in dir(_numexpr):
            local[key] = getattr(_numexpr, key)
    else:
        for key in dir(_numexpr):
            if key not in ("evaluate", "NumExpr"):
                local[key] = getattr(_numexpr, key)
        # patch NumExpr and evaluate with  `sanitize=False`
        def NumExpr(*args, **kwargs):
            kwargs["sanitize"] = False
            return _numexpr.NumExpr(*args, **kwargs)

        def evaluate(*args, **kwargs):
            kwargs["sanitize"] = False
            return _numexpr.evaluate(*args, **kwargs)
