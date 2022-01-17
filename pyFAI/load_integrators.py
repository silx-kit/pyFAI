#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2022 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2022"
__status__ = "stable"
__docformat__ = 'restructuredtext'

"""
This module tries to load every possible type of integrator and registers them
into the registry   
"""
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
import numpy

from .method_registry import IntegrationMethod
from .engines import CSR_engine as py_CSR_engine
# Register numpy integrators which are fail-safe
from .engines import histogram_engine
IntegrationMethod(1, "no", "histogram", "python", old_method="numpy",
                  class_funct_ng=(None, histogram_engine.histogram1d_engine))
IntegrationMethod(2, "no", "histogram", "python", old_method="numpy",
                  class_funct_ng=(None, histogram_engine.histogram2d_engine))

try:
    from .ext import histogram
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.histogram"
                 " Cython histogram implementation: %s", error)
    histogram = None
else:
    # Register histogram integrators
    IntegrationMethod(1, "no", "histogram", "cython", old_method="cython",
                      class_funct_legacy=(None, histogram.histogram),
                      class_funct_ng=(None, histogram.histogram1d_engine))
    IntegrationMethod(2, "no", "histogram", "cython", old_method="cython",
                      class_funct_legacy=(None, histogram.histogram2d),
                      class_funct_ng=(None, histogram.histogram2d_engine))

try:
    from .ext import splitBBox  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitBBox"
                 " Bounding Box pixel splitting: %s", error)
    splitBBox = None
else:
    # Register splitBBox integrators
    IntegrationMethod(1, "bbox", "histogram", "cython", old_method="bbox",
                      class_funct_legacy=(None, splitBBox.histoBBox1d),
                      class_funct_ng=(None, splitBBox.histoBBox1d_engine))
    IntegrationMethod(2, "bbox", "histogram", "cython", old_method="bbox",
                      class_funct_legacy=(None, splitBBox.histoBBox2d),
                      class_funct_ng=(None, splitBBox.histoBBox2d_engine),)

try:
    from .ext import splitPixel
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitPixel full pixel splitting: %s", error)
    logger.debug("Backtrace", exc_info=True)
    splitPixel = None
else:
    # Register splitPixel integrators
    IntegrationMethod(1, "full", "histogram", "cython", old_method="splitpixel",
                      class_funct_legacy=(None, splitPixel.fullSplit1D),
                      class_funct_ng=(None, splitPixel.fullSplit1D_engine))
    IntegrationMethod(2, "full", "histogram", "cython", old_method="splitpixel",
                      class_funct_legacy=(None, splitPixel.fullSplit2D),
                      class_funct_ng=(None, splitPixel.fullSplit2D_engine))
    IntegrationMethod(2, "pseudo", "histogram", "cython", old_method="splitpixel",
                      class_funct_legacy=(None, splitPixel.fullSplit2D),
                      class_funct_ng=(None, splitPixel.pseudoSplit2D_engine))

try:
    from .ext import splitBBoxCSR  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitBBoxCSR"
                 " CSR based azimuthal integration: %s", error)
    splitBBoxCSR = None
else:
    # Register splitBBoxCSR integrators
    IntegrationMethod(1, "no", "CSR", "cython", old_method="nosplit_csr",
                      class_funct_legacy=(splitBBoxCSR.HistoBBox1d, splitBBoxCSR.HistoBBox1d.integrate_legacy),
                      class_funct_ng=(splitBBoxCSR.HistoBBox1d, splitBBoxCSR.HistoBBox1d.integrate_ng))
    IntegrationMethod(2, "no", "CSR", "cython", old_method="nosplit_csr",
                      class_funct_legacy=(splitBBoxCSR.HistoBBox2d, splitBBoxCSR.HistoBBox2d.integrate))
    IntegrationMethod(1, "bbox", "CSR", "cython", old_method="csr",
                      class_funct_legacy=(splitBBoxCSR.HistoBBox1d, splitBBoxCSR.HistoBBox1d.integrate_legacy),
                      class_funct_ng=(splitBBoxCSR.HistoBBox1d, splitBBoxCSR.HistoBBox1d.integrate_ng))
    IntegrationMethod(2, "bbox", "CSR", "cython", old_method="csr",
                      class_funct_legacy=(splitBBoxCSR.HistoBBox2d, splitBBoxCSR.HistoBBox2d.integrate))

    IntegrationMethod(1, "no", "CSR", "python",
                      class_funct_ng=(py_CSR_engine.CsrIntegrator1d, py_CSR_engine.CsrIntegrator1d.integrate))
    IntegrationMethod(2, "no", "CSR", "python",
                      class_funct_legacy=(py_CSR_engine.CsrIntegrator2d, py_CSR_engine.CsrIntegrator2d.integrate))
    IntegrationMethod(1, "bbox", "CSR", "python",
                      class_funct_ng=(py_CSR_engine.CsrIntegrator1d, py_CSR_engine.CsrIntegrator1d.integrate))
    IntegrationMethod(2, "bbox", "CSR", "python",
                      class_funct_legacy=(py_CSR_engine.CsrIntegrator2d, py_CSR_engine.CsrIntegrator2d.integrate))

try:
    from .ext import splitBBoxLUT
except ImportError as error:
    logger.warning("Unable to import pyFAI.ext.splitBBoxLUT for"
                   " Look-up table based azimuthal integration")
    logger.debug("Backtrace", exc_info=True)
    splitBBoxLUT = None
else:
    # Register splitBBoxLUT integrators
    IntegrationMethod(1, "bbox", "LUT", "cython", old_method="lut",
                      class_funct_legacy=(splitBBoxLUT.HistoBBox1d, splitBBoxLUT.HistoBBox1d.integrate),
                      class_funct_ng=(splitBBoxLUT.HistoBBox1d, splitBBoxLUT.HistoBBox1d.integrate_ng))
    IntegrationMethod(2, "bbox", "LUT", "cython", old_method="lut",
                      class_funct_legacy=(splitBBoxLUT.HistoBBox2d, splitBBoxLUT.HistoBBox2d.integrate))
    IntegrationMethod(1, "no", "LUT", "cython", old_method="nosplit_lut",
                      class_funct_legacy=(splitBBoxLUT.HistoBBox1d, splitBBoxLUT.HistoBBox1d.integrate_legacy),
                      class_funct_ng=(splitBBoxLUT.HistoBBox1d, splitBBoxLUT.HistoBBox1d.integrate_ng))
    IntegrationMethod(2, "no", "LUT", "cython", old_method="nosplit_lut",
                      class_funct_legacy=(splitBBoxLUT.HistoBBox2d, splitBBoxLUT.HistoBBox2d.integrate),
                      class_funct_ng=(splitBBoxLUT.HistoBBox2d, splitBBoxLUT.HistoBBox2d.integrate_ng))

try:
    from .ext import splitPixelFullLUT
except ImportError as error:
    logger.warning("Unable to import pyFAI.ext.splitPixelFullLUT for"
                   " Look-up table based azimuthal integration")
    logger.debug("Backtrace", exc_info=True)
    splitPixelFullLUT = None
else:
    # Register splitPixelFullLUT integrators
    IntegrationMethod(1, "full", "LUT", "cython", old_method="full_lut",
                      class_funct_legacy=(splitPixelFullLUT.HistoLUT1dFullSplit, splitPixelFullLUT.HistoLUT1dFullSplit.integrate),
                      class_funct_ng=(splitPixelFullLUT.HistoLUT1dFullSplit, splitPixelFullLUT.HistoLUT1dFullSplit.integrate_ng))
    IntegrationMethod(2, "full", "LUT", "cython", old_method="full_lut",
                      class_funct_legacy=(splitPixelFullLUT.HistoLUT2dFullSplit, splitPixelFullLUT.HistoLUT2dFullSplit.integrate),
                      class_funct_ng=(splitPixelFullLUT.HistoLUT2dFullSplit, splitPixelFullLUT.HistoLUT2dFullSplit.integrate_ng))

try:
    from .ext import splitPixelFullCSR  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitPixelFullCSR"
                 " CSR based azimuthal integration: %s", error)
    splitPixelFullCSR = None
else:
    # Register splitPixelFullCSR integrators
    IntegrationMethod(1, "full", "CSR", "cython", old_method="full_csr",
                      class_funct_legacy=(splitPixelFullCSR.FullSplitCSR_1d, splitPixelFullCSR.FullSplitCSR_1d.integrate_legacy),
                      class_funct_ng=(splitPixelFullCSR.FullSplitCSR_1d, splitPixelFullCSR.FullSplitCSR_1d.integrate_ng))
    IntegrationMethod(2, "full", "CSR", "cython", old_method="full_csr",
                      class_funct_legacy=(splitPixelFullCSR.FullSplitCSR_2d, splitPixelFullCSR.FullSplitCSR_2d.integrate),
                      class_funct_ng=(splitPixelFullCSR.FullSplitCSR_2d, splitPixelFullCSR.FullSplitCSR_2d.integrate_ng))
    IntegrationMethod(1, "full", "CSR", "python",
                      class_funct_legacy=(py_CSR_engine.CsrIntegrator2d, py_CSR_engine.CsrIntegrator2d.integrate),
                      class_funct_ng=(py_CSR_engine.CsrIntegrator1d, py_CSR_engine.CsrIntegrator1d.integrate))

    IntegrationMethod(2, "full", "CSR", "python",
                      class_funct_legacy=(py_CSR_engine.CsrIntegrator2d, py_CSR_engine.CsrIntegrator2d.integrate),
                      class_funct_ng=(py_CSR_engine.CsrIntegrator2d, py_CSR_engine.CsrIntegrator2d.integrate))

try:
    from .opencl import ocl
except ImportError:
    ocl = None

if ocl:
    devices_list = []
    devtype_list = []
    devices = OrderedDict()
    perf = []
    for platform in ocl.platforms:
        for device in platform.devices:
            perf.append(device.flops)
            devices_list.append((platform.id, device.id))
            devtype_list.append(device.type.lower())

    for idx in (len(perf) - 1 - numpy.argsort(perf)):
        device = devices_list[idx]
        devices[device] = (f"{ocl.platforms[device[0]].name} / {ocl.platforms[device[0]].devices[device[1]].name}",
                           devtype_list[idx])

    try:
        from .opencl import azim_hist as ocl_azim  # IGNORE:F0401
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.opencl.azim_hist: %s", error)
        ocl_azim = None
    else:
        for ids, name in devices.items():
            IntegrationMethod(1, "no", "histogram", "OpenCL",
                              class_funct_ng=(ocl_azim.OCL_Histogram1d, ocl_azim.OCL_Histogram1d.integrate),
                              target=ids, target_name=name[0], target_type=name[1])
            IntegrationMethod(2, "no", "histogram", "OpenCL",
                              class_funct_ng=(ocl_azim.OCL_Histogram2d, ocl_azim.OCL_Histogram2d.integrate),
                              target=ids, target_name=name[0], target_type=name[1])
    try:
        from .opencl import azim_csr as ocl_azim_csr  # IGNORE:F0401
    except ImportError as error:
        logger.error("Unable to import pyFAI.opencl.azim_csr: %s", error)
        ocl_azim_csr = None
    else:
        if splitBBoxCSR:
            for ids, name in devices.items():
                IntegrationMethod(1, "bbox", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "bbox", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(1, "no", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "no", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
        if splitPixelFullCSR:
            for ids, name in devices.items():
                IntegrationMethod(1, "full", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "full", "CSR", "OpenCL",
                                  class_funct_legacy=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_csr.OCL_CSR_Integrator, ocl_azim_csr.OCL_CSR_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])

    try:
        from .opencl import azim_lut as ocl_azim_lut  # IGNORE:F0401
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.opencl.azim_lut: %s", error)
        ocl_azim_lut = None
    else:
        if splitBBoxLUT:
            for ids, name in devices.items():
                IntegrationMethod(1, "bbox", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "bbox", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(1, "no", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "no", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
        if splitPixelFullLUT:
            for ids, name in devices.items():
                IntegrationMethod(1, "full", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])
                IntegrationMethod(2, "full", "LUT", "OpenCL",
                                  class_funct_legacy=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate),
                                  class_funct_ng=(ocl_azim_lut.OCL_LUT_Integrator, ocl_azim_lut.OCL_LUT_Integrator.integrate_ng),
                                  target=ids, target_name=name[0], target_type=name[1])

    try:
        from .opencl import sort as ocl_sort
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.opencl.sort: %s", error)
        ocl_sort = None
else:
    ocl_sort = ocl_azim = ocl_azim_csr = ocl_azim_lut = None
