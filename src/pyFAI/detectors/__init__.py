# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2025 European Synchrotron Radiation Facility, Grenoble, France
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
#

"""Description of all detectors with a factory to instantiate them."""


__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/10/2025"
__status__ = "stable"

import inspect
from ._common import Detector, NexusDetector
from ._adsc import *      # noqa: F403
from ._dectris import *   # noqa: F403
from ._imxpad import *    # noqa: F403
from ._rayonix import *   # noqa: F403
from ._esrf import *      # noqa: F403
from ._xspectrum import * # noqa: F403 
from ._psi import *       # noqa: F403  
from ._non_flat import *  # noqa: F403
from ._others import *    # noqa: F403
from ._hexagonal import * # noqa: F403


ALL_DETECTORS = Detector.registry
detector_factory = Detector.factory
load = NexusDetector.sload

# Expose all the classes, else it is not part of the documentation
_detector_class_names = [i[0] for i in locals().items() if inspect.isclass(i[1]) and issubclass(i[1], Detector)]
__all__ = _detector_class_names + ["ALL_DETECTORS", "detector_factory", "load"]
