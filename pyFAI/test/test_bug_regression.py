#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""test suite for non regression on some bugs.

Please refer to their respective bug number
https://github.com/silx-kit/pyFAI/issues
"""


from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "2015-2018 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/12/2018"

import sys
import os
import unittest
import numpy
import subprocess
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
import fabio
from .. import load
from ..azimuthalIntegrator import AzimuthalIntegrator
from .. import detectors
from .. import units
from math import pi

try:
    import importlib.util
    if "module_from_spec" not in dir(importlib.util):
        raise ImportError
except ImportError:
    import importlib

    def load_source(module_name, file_path):
        """Plugin loader which does not pollute sys.module,

        Not as powerful as the v3.5+
        """
        return importlib.import_module(module_name, module_name.split(".")[0])

else:

    def load_source(module_name, file_path):
        """Plugin loader which does not pollute sys.module,

        Python Version >=3.5"""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestBug170(unittest.TestCase):
    """
    Test a mar345 image with 2300 pixels size
    """

    def setUp(self):
        ponitxt = """
Detector: Mar345
PixelSize1: 0.00015
PixelSize2: 0.00015
Distance: 0.446642915189
Poni1: 0.228413453499
Poni2: 0.272291324302
Rot1: 0.0233130647508
Rot2: 0.0011735285628
Rot3: -7.22446379865e-08
SplineFile: None
Wavelength: 7e-11
"""
        self.ponifile = os.path.join(UtilsTest.tempdir, "bug170.poni")
        with open(self.ponifile, "w") as poni:
            poni.write(ponitxt)
        self.data = numpy.random.random((2300, 2300))

    def tearDown(self):
        if os.path.exists(self.ponifile):
            os.unlink(self.ponifile)
        self.data = None

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def test_bug170(self):
        ai = load(self.ponifile)
        logger.debug(ai.mask.shape)
        logger.debug(ai.detector.pixel1)
        logger.debug(ai.detector.pixel2)
        ai.integrate1d(self.data, 2000)


class TestBug211(unittest.TestCase):
    """
    Check the quantile filter in pyFAI-average
    """
    def setUp(self):
        shape = (100, 100)
        dtype = numpy.float32
        self.image_files = []
        self.outfile = os.path.join(UtilsTest.tempdir, "out.edf")
        res = numpy.zeros(shape, dtype=dtype)
        for i in range(5):
            fn = os.path.join(UtilsTest.tempdir, "img_%i.edf" % i)
            if i == 3:
                data = numpy.zeros(shape, dtype=dtype)
            elif i == 4:
                data = numpy.ones(shape, dtype=dtype)
            else:
                data = numpy.random.random(shape).astype(dtype)
                res += data
            e = fabio.edfimage.edfimage(data=data)
            e.write(fn)
            self.image_files.append(fn)
        self.res = res / 3.0
        # It is not anymore a script, but a module
        import pyFAI.app.average
        self.exe = pyFAI.app.average.__file__
        self.env = UtilsTest.get_test_env()

    def tearDown(self):
        for fn in self.image_files:
            os.unlink(fn)
        if os.path.exists(self.outfile):
            os.unlink(self.outfile)
        self.image_files = None
        self.res = None
        self.exe = self.env = None

    def test_quantile(self):
        command_line = [sys.executable, self.exe, "--quiet", "-q", "0.2-0.8", "-o", self.outfile] + self.image_files

        p = subprocess.Popen(command_line,
                             shell=False, env=self.env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rc = p.wait()
        if rc:
            logger.info(p.stdout.read())
            logger.error(p.stderr.read())
            logger.error(os.linesep + (" ".join(command_line)))
            env = "Environment:"
            for k, v in self.env.items():
                env += "%s    %s: %s" % (os.linesep, k, v)
            logger.error(env)
            self.fail()

        if fabio.hexversion < 262147:
            logger.error("Error: the version of the FabIO library is too old: %s, please upgrade to 0.4+. Skipping test for now", fabio.version)
            return

        self.assertEqual(rc, 0, msg="pyFAI-average return code %i != 0" % rc)
        self.assertTrue(numpy.allclose(fabio.open(self.outfile).data, self.res),
                        "pyFAI-average with quantiles gives good results")


class TestBugRegression(unittest.TestCase):
    "just a bunch of simple tests"
    def test_bug_232(self):
        """
        Check the copy and deepcopy methods of Azimuthal integrator
        """
        det = detectors.ImXPadS10()
        ai = AzimuthalIntegrator(dist=1, detector=det)
        data = numpy.random.random(det.shape)
        _result = ai.integrate1d(data, 100, unit="r_mm")
        import copy
        ai2 = copy.copy(ai)
        self.assertNotEqual(id(ai), id(ai2), "copy instances are different")
        self.assertEqual(id(ai.ra), id(ai2.ra), "copy arrays are the same after copy")
        self.assertEqual(id(ai.detector), id(ai2.detector), "copy detector are the same after copy")
        ai3 = copy.deepcopy(ai)
        self.assertNotEqual(id(ai), id(ai3), "deepcopy instances are different")
        self.assertNotEqual(id(ai.ra), id(ai3.ra), "deepcopy arrays are different after copy")
        self.assertNotEqual(id(ai.detector), id(ai3.detector), "deepcopy arrays are different after copy")

    def test_bug_174(self):
        """
        wavelength change not taken into account (memoization error)
        """
        ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        data = fabio.open(UtilsTest.getimage("Pilatus1M.edf")).data
        wl1 = 1e-10
        wl2 = 2e-10
        ai.wavelength = wl1
        q1, i1 = ai.integrate1d(data, 1000)
        # ai.reset()
        ai.wavelength = wl2
        q2, i2 = ai.integrate1d(data, 1000)
        dq = (abs(q1 - q2).max())
        _di = (abs(i1 - i2).max())
        # print(dq)
        self.assertAlmostEqual(dq, 3.79, 2, "Q-scale difference should be around 3.8, got %s" % dq)

    def test_bug_758(self):
        """check the stored "h*c" constant is almost 12.4"""
        hc = 12.398419292004204  # Old reference value
        self.assertAlmostEqual(hc, units.hc, 6, "hc is correct, got %s" % units.hc)

    def test_bug_808(self):
        """Try to import every single module in the package
        """
        import pyFAI
#         print(pyFAI.__file__)
#         print(pyFAI.__name__)
        pyFAI_root = os.path.split(pyFAI.__file__)[0]

        for root, dirs, files in os.walk(pyFAI_root, topdown=True):
            for adir in dirs:

                subpackage_path = os.path.join(root, adir, "__init__.py")
                subpackage = "pyFAI" + subpackage_path[len(pyFAI_root):-12].replace(os.sep, ".")
                if os.path.isdir(subpackage_path):
                    logger.info("Loading subpackage: %s from %s", subpackage, subpackage_path)
                    sys.modules[subpackage] = load_source(subpackage, subpackage_path)
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(root, name)
                    fqn = "pyFAI" + path[len(pyFAI_root):-3].replace(os.sep, ".")
                    logger.info("Importing %s from %s", fqn, path)
                    try:
                        load_source(fqn, path)
                    except Exception as err:
                        if ((isinstance(err, ImportError) and
                                "No Qt wrapper found" in err.__str__() or
                                "pyopencl is not installed" in err.__str__() or
                                "PySide" in err.__str__()) or
                            (isinstance(err, SystemError) and
                                "Parent module" in err.__str__())):

                            logger.info("Expected failure importing %s from %s with error: %s",
                                        fqn, path, err)
                        else:
                            logger.error("Failed importing %s from %s with error: %s%s: %s",
                                         fqn, path, os.linesep,
                                         err.__class__.__name__, err)
                            raise err

    def test_bug_816(self):
        "Ensure the chi-disontinuity is properly set"
        detector = detectors.detector_factory("Pilatus 300k")
        positions = detector.get_pixel_corners()
        # y_min = positions[..., 1].min()
        y_max = positions[..., 1].max()
        # x_min = positions[..., 0].min()
        x_max = positions[..., 2].max()

        expected = {  # poni -> azimuthal range in both convention
                    (0, 0): [(0, pi / 2), (0, pi / 2)],
                    (y_max / 2, x_max / 2): [(-pi, pi), (0, 2 * pi)],
                    (y_max, 0): [(-pi / 2, 0), (3 * pi / 2, 2 * pi)],
                    (0, x_max): [(pi / 2, pi), (pi / 2, pi)],
                    (y_max, x_max): [(-pi, -pi / 2), (pi, 3 * pi / 2)],
                   }

        for poni, chi_range in expected.items():
            logger.debug("%s, %s", poni, chi_range)
            ai = AzimuthalIntegrator(0.1, *poni, detector=detector)
            chi_pi_center = ai.chiArray()
            logger.debug("disc @pi center: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range, chi_pi_center.min(), chi_pi_center.max())
            chi_pi_corner = ai.array_from_unit(typ="corner", unit="r_m", scale=False)[1:-1, 1:-1, :, 1]
            logger.debug("disc @pi corner: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range, chi_pi_corner.min(), chi_pi_corner.max())

            self.assertAlmostEquals(chi_pi_center.min(), chi_range[0][0], msg="chi_pi_center.min", delta=0.1)
            self.assertAlmostEquals(chi_pi_corner.min(), chi_range[0][0], msg="chi_pi_corner.min", delta=0.1)
            self.assertAlmostEquals(chi_pi_center.max(), chi_range[0][1], msg="chi_pi_center.max", delta=0.1)
            self.assertAlmostEquals(chi_pi_corner.max(), chi_range[0][1], msg="chi_pi_corner.max", delta=0.1)

            ai.reset()
            ai.setChiDiscAtZero()

            logger.debug("Updated range %s %s %s %s", chi_range[0], chi_range[1], ai.chiDiscAtPi, list(ai._cached_array.keys()))
            chi_0_center = ai.chiArray()
            logger.debug("disc @0 center: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range[1], chi_0_center.min(), chi_0_center.max())
            chi_0_corner = ai.array_from_unit(typ="corner", unit="r_m", scale=False)[1:-1, 1:-1, :, 1]  # Discard pixel from border...
            logger.debug("disc @0 corner: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range[1], chi_0_corner.min(), chi_0_corner.max())

            dmin = lambda v: v - chi_range[1][0]
            dmax = lambda v: v - chi_range[1][1]
            self.assertAlmostEquals(dmin(chi_0_center.min()), 0, msg="chi_0_center.min", delta=0.1)
            self.assertAlmostEquals(dmin(chi_0_corner.min()), 0, msg="chi_0_corner.min", delta=0.1)
            self.assertAlmostEquals(dmax(chi_0_center.max()), 0, msg="chi_0_center.max", delta=0.1)
            self.assertAlmostEquals(dmax(chi_0_corner.max()), 0, msg="chi_0_corner.max", delta=0.1)

    def test_bug_924(self):
        "Regression on spline calculation for single pixel coordinate"
        dp = detectors.detector_factory("Pilatus100k")
        aip = AzimuthalIntegrator(detector=dp)
        aip.chi(numpy.array([1, 2]), numpy.array([3, 4]))
        aip.chi(numpy.array([1]), numpy.array([3]))
        # so far, so good
        df = detectors.detector_factory("Frelon",
                                        {"splineFile": UtilsTest.getimage("frelon.spline")})
        print(df.spline.splineFuncX(numpy.array([1, 2]), numpy.array([3, 4]), True))
        aif = AzimuthalIntegrator(detector=df)
        aif.chi(numpy.array([1, 2]), numpy.array([3, 4]))
        aif.chi(numpy.array([1]), numpy.array([3]))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBug170))
    testsuite.addTest(loader(TestBug211))
    testsuite.addTest(loader(TestBugRegression))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
