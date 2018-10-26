#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "22/10/2018"

import sys
import logging
if "ps1" in dir(sys):
    logging.basicConfig()

import os

try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion
    from ._version import citation, calc_hexversion
except ImportError:
    project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    raise RuntimeError("Do NOT use %s from its sources: build it and use the built version" % project)

if sys.version_info < (2, 6):
    logger = logging.getLogger(__name__)
    logger.error("pyFAI required a python version >= 2.6")
    raise RuntimeError("pyFAI required a python version >= 2.6, now we are running: %s" % sys.version)

from .utils import decorators


use_opencl = True
"""Global configuration which allow to disable OpenCL programatically.
It must be set before requesting any OpenCL modules.

.. code-block:: python

    import pyFAI
    pyFAI.use_opencl = False
"""


@decorators.deprecated(replacement="pyFAI.azimuthalIntegrator.AzimuthalIntegrator", since_version="0.16")
def AzimuthalIntegrator(*args, **kwargs):
    from .azimuthalIntegrator import AzimuthalIntegrator
    return AzimuthalIntegrator(*args, **kwargs)


def load(filename):
    """
    Load an azimuthal integrator from a filename description.

    :param str filename: name of the file to load
    :return: instance of Gerometry of AzimuthalIntegrator set-up with the parameter from the file.
    """
    from .azimuthalIntegrator import AzimuthalIntegrator
    return AzimuthalIntegrator.sload(filename)


def detector_factory(name, config=None):
    """
    Create a new detector.

    :param str name: name of a detector
    :param dict config: configuration of the detector supporting dict or JSON
        representation.
    :return: an instance of the right detector, set-up if possible
    :rtype: pyFAI.detectors.Detector
    """
    from .detectors import Detector
    return Detector.factory(name, config)


def tests(deprecation=False):
    """Runs the test suite of the installed version

    :param deprecation: enable/disables deprecation warning in the tests
    """
    if deprecation:
        decorators.depreclog.setLevel(logging.DEBUG)
    else:
        decorators.depreclog.setLevel(logging.ERROR)
    from . import test
    res = test.run_tests()
    decorators.depreclog.setLevel(logging.DEBUG)
    return res


def benchmarks(*arg, **kwarg):
    """Run the integrated benchmarks.

    See the documentation of pyFAI.benchmark.run_benchmark
    """
    from . import benchmark
    res = benchmark.run(*arg, **kwarg)
    return res
