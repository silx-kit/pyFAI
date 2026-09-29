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

"""Utilities for testing the graphical user interface."""

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/09/2026"

import gc
import logging

from silx.gui.utils import testutils

logger = logging.getLogger(__name__)


class TestCaseQt(testutils.TestCaseQt):
    """The `TestCaseQt` of silx, which also destroys the widgets of the test.

    PySide6 destroys the C++ widget when Python collects the wrapper object,
    hence at a moment decided by the garbage collector. silx looks for widgets
    left alive with the other bindings only -- and it is that check which
    triggers the collection -- so under PySide6 a widget a test forgot to
    delete may be destroyed while Qt is delivering events to it. The process
    then dies with an access violation: this was seen on Windows, where the
    test-suite runs in parallel, as a worker crashing in the `processEvents()`
    of `TestCaseQt.tearDown`.

    This class collects and deletes the widgets left behind at the end of each
    test, at a point where Qt is not processing events.
    """

    def setUp(self):
        super().setUp()
        self._widgets_before = self.qapp.allWidgets()

    def tearDown(self):
        # Collecting here is the point of this class: this is what hands the
        # C++ widgets back to Qt, and it must not happen in the middle of a
        # later processEvents().
        gc.collect()
        for widget in self.qapp.allWidgets():
            if widget in self._widgets_before:
                continue
            try:
                if widget.parent() is None:
                    # children are deleted along with their parent
                    widget.deleteLater()
            except RuntimeError:  # already gone on the C++ side
                logger.debug("Widget already destroyed", exc_info=True)
        self._widgets_before = None
        self.qapp.processEvents()
        super().tearDown()
