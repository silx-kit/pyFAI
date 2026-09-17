#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Pytest configuration for the pyFAI test-suite.

The test-suite is written with `unittest`; pytest collects it as-is. This file
only reproduces what `run_tests.py` does around the suite, so that::

    pytest --pyargs pyFAI            # whole test-suite
    pytest -n auto --pyargs pyFAI    # ... in parallel, needs pytest-xdist

behave like `python run_tests.py`.

Options are read from the environment (same variables as `run_tests.py`):
WITH_QT_TEST, PYFAI_OPENCL, WITH_GL_TEST, PYFAI_LOW_MEM, PYFAI_RANDOM.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/09/2026"

import logging

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Configure the test options and pre-fetch the test images.

    The download is performed by the controller process only: with pytest-xdist
    several workers requesting the same file at the same time would race on the
    cache directory.
    """
    from .test.utilstest import test_options
    test_options.configure()

    if not hasattr(config, "workerinput"):  # controller (or no xdist at all)
        try:
            test_options.download_images()
        except Exception as err:  # offline: let the tests skip/fail on their own
            logger.warning("Could not pre-fetch the test images: %s", err)


def pytest_sessionfinish(session, exitstatus):
    """Remove the temporary directory created by the test-suite."""
    from .test.utilstest import test_options
    test_options.clean_up()


def pytest_ignore_collect(collection_path, config):
    """Skip the test modules which cannot even be imported.

    `unittest` never imports them: `pyFAI.opencl.test.suite()` and
    `pyFAI.gui.test.suite()` import their sub-modules only when the
    corresponding option is enabled. Pytest, on the contrary, imports every
    collected module, and `pyFAI.opencl.test.test_addition` (for instance)
    fails at import time when pyopencl is missing.
    """
    from .test.utilstest import test_options
    parts = collection_path.parts
    if "test" not in parts:
        return None
    if "opencl" in parts and not test_options.WITH_OPENCL_TEST:
        return True
    if "gui" in parts and not test_options.WITH_QT_TEST:
        return True
    return None
