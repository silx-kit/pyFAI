#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#             Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
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


"""GUI interface for reduction of diffraction tomography experiments"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/10/2018"
__satus__ = "Production"

import logging
import sys
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger("diff_map")
from pyFAI.diffmap import DiffMap


def main():
    dt = DiffMap()
    options, config = dt.parse(with_config=True)

    if not options.gui:
        dt.setup_ai()
        dt.makeHDF5()
        dt.process()
        dt.show_stats()
    else:
        from silx.gui import qt
        from pyFAI.gui.diffmap_widget import DiffMapWidget
        app = qt.QApplication([])
        window = DiffMapWidget()
        window.set_config(config)
        # window.restore()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
