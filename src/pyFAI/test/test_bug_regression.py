#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "2015-2025 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/06/2025"

import sys
import os
import unittest
import numpy
import subprocess
import copy
import logging
logger = logging.getLogger(__name__)
from .utilstest import UtilsTest
from ..utils import mathutil
import fabio
from .. import load
from ..integrator.azimuthal import AzimuthalIntegrator, logger as ai_logger
from .. import detectors
from .. import units
from math import pi
from ..opencl import ocl

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
        rng = UtilsTest.get_rng()
        self.data = rng.random((2300, 2300))

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
        ai.integrate1d_ng(self.data, 2000)


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
        rng = UtilsTest.get_rng()
        for i in range(5):
            fn = os.path.join(UtilsTest.tempdir, "img_%i.edf" % i)
            if i == 3:
                data = numpy.zeros(shape, dtype=dtype)
            elif i == 4:
                data = numpy.ones(shape, dtype=dtype)
            else:
                data = rng.random(shape).astype(dtype)
                res += data
            e = fabio.edfimage.edfimage(data=data)
            e.write(fn)
            self.image_files.append(fn)
        self.res = res / 3.0
        # It is not anymore a script, but a module
        from ..app import average
        self.exe = average.__name__
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
        args = ["--quiet", "-q", "0.2-0.8", "-o", self.outfile] + self.image_files
        command_line = [sys.executable, "-c" ,f"""import {self.exe}; {self.exe}.main({args})"""]

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
        with fabio.open(self.outfile) as fimg:
            self.assertTrue(numpy.allclose(fimg.data, self.res),
                        "pyFAI-average with quantiles gives good results")


class TestBugRegression(unittest.TestCase):
    "just a bunch of simple tests"

    def test_bug_232(self):
        """
        Check the copy and deepcopy methods of Azimuthal integrator
        """
        det = detectors.ImXPadS10()
        ai = AzimuthalIntegrator(dist=1, detector=det)
        data = UtilsTest.get_rng().random(det.shape)
        _result = ai.integrate1d_ng(data, 100, unit="r_mm")

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
        with fabio.open(UtilsTest.getimage("Pilatus1M.edf")) as fimg:
            data = fimg.data
        wl1 = 1e-10
        wl2 = 2e-10
        ai.wavelength = wl1
        q1, i1 = ai.integrate1d_ng(data, 1000)
        # ai.reset()
        ai.wavelength = wl2
        q2, i2 = ai.integrate1d_ng(data, 1000)
        dq = (abs(q1 - q2).max())
        _di = (abs(i1 - i2).max())
        # print(dq)
        self.assertAlmostEqual(dq, 3.79, 2, "Q-scale difference should be around 3.8, got %s" % dq)

    def test_bug_758(self):
        """check the stored "h*c" constant is almost 12.4"""
        hc = 12.398419292004204  # Old reference value from pyFAI
        hc = 12.398419739640717  # calculated from scipy 1.3
        self.assertAlmostEqual(hc, units.hc, 6, "hc is correct, got %s" % units.hc)

    def test_import_all_modules(self):
        """Try to import every single module in the package
        """
        import pyFAI
        pyFAI_root = os.path.split(pyFAI.__file__)[0]

        def must_be_skipped(path):
            path = os.path.relpath(path, pyFAI_root)
            path = path.replace("\\", "/")
            elements = path.split("/")
            if "test" in elements:
                # Always skip test modules
                logger.warning("Skip test module %s", path)
                return True
            if not UtilsTest.WITH_OPENCL_TEST:
                if "opencl" in elements:
                    logger.warning("Skip %s. OpenCL tests disabled", path)
                    return True
            if not UtilsTest.WITH_QT_TEST:
                if "gui" in elements:
                    logger.warning("Skip %s. Qt tests disabled", path)
                    return True
            return False

        for root, dirs, files in os.walk(pyFAI_root, topdown=True):
            for adir in dirs:
                subpackage_path = os.path.join(root, adir, "__init__.py")
                if must_be_skipped(subpackage_path):
                    continue
                subpackage = "pyFAI" + subpackage_path[len(pyFAI_root):-12].replace(os.sep, ".")
                if os.path.isdir(subpackage_path):
                    logger.info("Loading subpackage: %s from %s", subpackage, subpackage_path)
                    sys.modules[subpackage] = load_source(subpackage, subpackage_path)
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                if must_be_skipped(path):
                    continue
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
            chi_pi_corner = ai.array_from_unit(typ="corner", unit="r_m", scale=False)[1:-1, 1:-1,:, 1]
            logger.debug("disc @pi corner: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range, chi_pi_corner.min(), chi_pi_corner.max())

            self.assertAlmostEqual(chi_pi_center.min(), chi_range[0][0], msg="chi_pi_center.min", delta=0.1)
            self.assertAlmostEqual(chi_pi_corner.min(), chi_range[0][0], msg="chi_pi_corner.min", delta=0.1)
            self.assertAlmostEqual(chi_pi_center.max(), chi_range[0][1], msg="chi_pi_center.max", delta=0.1)
            self.assertAlmostEqual(chi_pi_corner.max(), chi_range[0][1], msg="chi_pi_corner.max", delta=0.1)

            ai.reset()
            ai.setChiDiscAtZero()

            logger.debug("Updated range %s %s %s %s", chi_range[0], chi_range[1], ai.chiDiscAtPi, list(ai._cached_array.keys()))
            chi_0_center = ai.chiArray()
            logger.debug("disc @0 center: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range[1], chi_0_center.min(), chi_0_center.max())
            chi_0_corner = ai.array_from_unit(typ="corner", unit="r_m", scale=False)[1:-1, 1:-1,:, 1]  # Discard pixel from border...
            logger.debug("disc @0 corner: poni: %s; expected: %s; got: %.2f, %.2f", poni, chi_range[1], chi_0_corner.min(), chi_0_corner.max())

            dmin = lambda v: v - chi_range[1][0]
            dmax = lambda v: v - chi_range[1][1]
            self.assertAlmostEqual(dmin(chi_0_center.min()), 0, msg="chi_0_center.min", delta=0.1)
            self.assertAlmostEqual(dmin(chi_0_corner.min()), 0, msg="chi_0_corner.min", delta=0.1)
            self.assertAlmostEqual(dmax(chi_0_center.max()), 0, msg="chi_0_center.max", delta=0.1)
            self.assertAlmostEqual(dmax(chi_0_corner.max()), 0, msg="chi_0_corner.max", delta=0.1)

    def test_bug_924(self):
        "Regression on spline calculation for single pixel coordinate"
        dp = detectors.detector_factory("Imxpad S10")
        aip = AzimuthalIntegrator(detector=dp)
        aip.chi(numpy.array([1, 2]), numpy.array([3, 4]))
        aip.chi(numpy.array([1]), numpy.array([3]))
        # so far, so good
        df = detectors.detector_factory("Frelon",
                                        {"splineFile": UtilsTest.getimage("frelon.spline")})
        # print(df.spline.splineFuncX(numpy.array([1, 2]), numpy.array([3, 4]), True))
        aif = AzimuthalIntegrator(detector=df)
        aif.chi(numpy.array([1, 2]), numpy.array([3, 4]))
        aif.chi(numpy.array([1]), numpy.array([3]))

    def test_bug_1275(self):
        "This bug about major sectors not taken into account when performing intgrate1d on small azimuthal sectors"
        shape = (128, 128)
        detector = detectors.Detector(100e-4, 100e-4, max_shape=shape)
        ai = AzimuthalIntegrator(detector=detector, wavelength=1e-10)
        # ai_logger.setLevel(logging.ERROR)
        ai.setFit2D(1000, shape[1] / 2, shape[0] / 2)
        data = numpy.ones(shape)
        nb_pix = ai.integrate1d_ng(data, 100).count.sum()
        self.assertAlmostEqual(nb_pix, numpy.prod(shape), msg="All pixels are counted", delta=0.01)

        delta = 45
        target = numpy.prod(shape) / 180 * delta
        ai.setChiDiscAtPi()
        angles = numpy.arange(-180, 400, 90)
        # print(angles)
        for method in [("no", "histogram", "python"),
                       ("no", "histogram", "cython"),
                       ("no", "csr", "cython"),
                       ("no", "lut", "cython"),
                       ]:
            for angle in angles:
                res0 = ai.integrate1d_ng(data, 100, azimuth_range=(angle - delta, angle + delta), method=method)
                # try:
                #     print(ai.engines[res0.method].engine.pos1_range, ai.engines[res0.method].engine.pos1_min, ai.engines[res0.method].engine.pos1_maxin, ai.engines[res0.method].engine.pos1_max)
                # except:
                #     pass
                res = res0.count.sum()
                # print("disc at π",  method, angle, res)
                if angle in (-180, 180):
                    # We expect only half of the pixel
                    self.assertLess(abs(res / target - 0.5), 0.1, f"ChiDiscAtPi with {method} at {angle} expect half of the pixels ({target}/2), got {res}")
                else:
                    self.assertLess(abs(res / target - 1), 0.1, f"ChiDiscAtPi with {method} at {angle} expect all pixels ({target}) and got {res}")

        # Now with the azimuthal integrator set with the chi discontinuity at 0
        ai.setChiDiscAtZero()
        for method in [("no", "histogram", "python"),
                       ("no", "histogram", "cython"),
                       ("no", "csr", "cython"),
                       ("no", "lut", "cython")]:
            for angle in angles:
                res0 = ai.integrate1d_ng(data, 100, azimuth_range=(angle - delta, angle + delta), method=method)
                # try:
                #     print(ai.engines[res0.method].engine.pos1_range, ai.engines[res0.method].engine.pos1_min, ai.engines[res0.method].engine.pos1_maxin, ai.engines[res0.method].engine.pos1_max)
                # except: pass
                res = res0.count.sum()
                # print("disc at 0",  method, angle, res)
                if angle in (0, 360):
                    # We expect only half of the pixel
                    self.assertLess(abs(res / target - 0.5), 0.1, f"ChiDiscAtZero with {method} at {angle} expect half of the pixels ({target}/2), got {res}")
                else:
                    self.assertLess(abs(res / target - 1), 0.1, f"ChiDiscAtZero with {method} at {angle} expect all pixel ({target}), got {res}")

    def test_bug_1421(self):
        """This bug is about geometry refinement not working with SAXS-constrains in certain conditions
        Inspired by the Recalib tutorial
        """
        from .. import geometry
        from ..calibrant import CALIBRANT_FACTORY
        from ..goniometer import SingleGeometry
        filename = UtilsTest.getimage("Pilatus1M.edf")
        with fabio.open(filename) as fimg:
            frame = fimg.data

        # Approximatively the position of the beam center ...
        x = 200  # x-coordinate of the beam-center in pixels
        y = 300  # y-coordinate of the beam-center in pixels
        d = 1600  # This is the distance in mm (unit used by Fit2d)
        wl = 1e-10  # The wavelength is 1 Å

        # Definition of the detector and of the calibrant:
        pilatus = detectors.detector_factory("Pilatus1M")
        behenate = CALIBRANT_FACTORY("AgBh")
        behenate.wavelength = wl

        # Set the guessed geometry
        initial = geometry.Geometry(detector=pilatus, wavelength=wl)
        initial.setFit2D(d, x, y)
#         print(initial)
        # The SingleGeometry object (from goniometer) allows to extract automatically ring and calibrate
        sg = SingleGeometry("demo", frame, calibrant=behenate, detector=pilatus, geometry=initial)
        sg.extract_cp(max_rings=5)

        # Refine the geometry ... here in SAXS geometry, the rotation is fixed in orthogonal setup
        sg.geometry_refinement.refine2(fix=["rot1", "rot2", "rot3", "wavelength"])
        refined = sg.get_ai()

        self.assertNotEqual(initial.dist, refined.dist, "Distance got refined")
        self.assertNotEqual(initial.poni1, refined.poni1, "Poni1 got refined")
        self.assertNotEqual(initial.poni2, refined.poni2, "Poni2 got refined")
        self.assertEqual(initial.rot1, refined.rot1, "Rot1 is unchanged")
        self.assertEqual(initial.rot2, refined.rot2, "Rot2 is unchanged")
        self.assertEqual(initial.rot3, refined.rot3, "Rot3 is unchanged")
        self.assertEqual(initial.wavelength, refined.wavelength, "Wavelength is unchanged")

        sg.geometry_refinement.refine2(fix=[])
#         print(refined)
        refined2 = sg.get_ai()

        self.assertNotEqual(refined2.dist, refined.dist, "Distance got refined")
        self.assertNotEqual(refined2.poni1, refined.poni1, "Poni1 got refined")
        self.assertNotEqual(refined2.poni2, refined.poni2, "Poni2 got refined")
        self.assertNotEqual(refined2.rot1, refined.rot1, "Rot1 got refined")
        self.assertNotEqual(refined2.rot2, refined.rot2, "Rot2 got refined")
        # self.assertNotEqual(refined2.rot3, refined.rot3, "Rot3 got refined") #Rot can change or not ...
        self.assertEqual(refined2.wavelength, refined.wavelength, "Wavelength is unchanged (refine2)")
#         print(refined2)
#         raise
        sg.geometry_refinement.refine3(fix=[])
#         print(refined)
        refined2 = sg.get_ai()

        self.assertNotEqual(refined2.dist, refined.dist, "Distance got refined")
        self.assertNotEqual(refined2.poni1, refined.poni1, "Poni1 got refined")
        self.assertNotEqual(refined2.poni2, refined.poni2, "Poni2 got refined")
        self.assertNotEqual(refined2.rot1, refined.rot1, "Rot1 got refined")
        self.assertNotEqual(refined2.rot2, refined.rot2, "Rot2 got refined")
#         self.assertNotEqual(refined2.rot3, refined.rot3, "Rot3 got refined")
        self.assertNotEqual(refined2.wavelength, refined.wavelength, "Wavelength got refined (refine3)")

    def test_bug_1487(self):
        """
        Reported by Marco @ID15:
        Apparently the azimuthal range limitation is honored with CSR implementation and full-pixel splitting in pyFAI 0.20.
        at lease for integrate1d legacy.
        """
        wl = 1e-10
        ai = AzimuthalIntegrator(0.1, 0.03, 0.00, detector="Pilatus_200k", wavelength=wl)

        tth = ai.twoThetaArray()
        iso = -tth * (tth - 0.8)
        chi = ai.chiArray()
        ani = numpy.sin(4 * chi) + 1
        img = iso * ani

        npt = 10
        sector_size = 20
        out = numpy.empty((180 // sector_size, npt))
        for method in [("full", "histogram", "cython"),
                       ("full", "lut", "cython"),
                       ("full", "csr", "cython"),
                       ]:
            idx = 0
            for start in range(-90, 90, sector_size):
                end = start + sector_size
                res = ai.integrate1d(img, npt, method=method, azimuth_range=[start, end])
                out[idx] = res.intensity
                idx += 1
            # print(out)
            std = out.std(axis=0)
            self.assertGreater(std.min(), 0, f"output are not all the same with {method}")

    def test_bug_1510(self):
        """
        CSR engine got systematically discarded when radial range is provided
        """
        method = ("no", "csr", "cython")
        detector = detectors.ImXPadS10()
        ai = AzimuthalIntegrator(detector=detector)
        rm = max(detector.shape) * detector.pixel1 * 1000
        img = UtilsTest.get_rng().random(detector.shape)
        ai.integrate1d(img, 5, unit="r_mm", radial_range=[0, rm], method=method)
        id_before = None
        for v in ai.engines.values():
            csre = v.engine
            id_before = id(csre)

        ai.integrate1d(img, 5, unit="r_mm", radial_range=[0, rm], method=method)
        id_after = None
        for v in ai.engines.values():
            id_after = id(v.engine)

        self.assertEqual(id_before, id_after, "The CSR engine got reset")

    def test_bug_1536(self):
        """Ensure setPyFAI accepts the output of getPyFAI()
        and that the detector description matches !
        """
        detector = detectors.Detector(5e-4, 5e-4, max_shape=(1100, 1000))
        ref = AzimuthalIntegrator(detector=detector)
        geo = ref.getPyFAI()
        # print(geo)
        # print(ref.get_config())
        obt = AzimuthalIntegrator()
        obt.setPyFAI(**geo)

        # print(obt)
        self.assertEqual(ref.detector.max_shape, obt.detector.max_shape, "max_shape matches")

    def test_bug_1810(self):
        "impossible to deepcopy goniometer calibration"
        import pyFAI.control_points
        cp = pyFAI.control_points.ControlPoints(calibrant="LaB6", wavelength=1e-10)
        self.assertNotEqual(id(cp), id(copy.deepcopy(cp)), "control_points copy works and id differs")

        import pyFAI.geometryRefinement
        gr = pyFAI.geometryRefinement.GeometryRefinement([[1, 2, 3]], detector="Imxpad S10", wavelength=1e-10, calibrant="LaB6")
        self.assertNotEqual(id(gr), id(copy.deepcopy(gr)), "geometryRefinement copy works and id differs")

        import pyFAI.massif
        ary = numpy.arange(100).reshape(10, 10)
        massif = pyFAI.massif.Massif(ary)
        self.assertNotEqual(id(massif), id(copy.deepcopy(massif)), "Massif copy works and id differs")

        import pyFAI.gui.peak_picker
        pp = pyFAI.gui.peak_picker.PeakPicker(ary)
        self.assertNotEqual(id(pp), id(copy.deepcopy(pp)), "PeakPicker copy works and id differs")

        from pyFAI.goniometer import SingleGeometry
        import pyFAI.calibrant
        lab6 = pyFAI.calibrant.get_calibrant("LaB6", 1e-10)
        cp.append([[1, 2], [3, 4]], 0)
        sg = SingleGeometry("frame", ary, "frame", lambda x:x, cp, lab6, "Imxpad S10")
        self.assertNotEqual(id(sg), id(copy.deepcopy(sg)), "SingleGeometry copy works and id differs")

    def test_bug_1889(self):
        "reset cached arrays"
        ai = load({"detector": "Imxpad S10", "wavelength": 1.54e-10})
        ai.polarization(factor=0.9)
        img = numpy.empty(ai.detector.shape, "float32")
        ai.integrate2d(img, 10, 9, method=("no", "histogram", "cython"))
        ai.integrate2d(img, 10, 9, method=("bbox", "histogram", "cython"))
        ai.setChiDiscAtZero()
        ai.integrate2d(img, 10, 9, method=("no", "histogram", "cython"))
        ai.integrate2d(img, 10, 9, method=("bbox", "histogram", "cython"))
        ai.setChiDiscAtPi()

    def test_bug_1946(self):
        """
        method ("full", "CSC", "python):
        """
        from ..method_registry import IntegrationMethod
        res = IntegrationMethod.select_method(dim=1, split="full", algo="csc", impl="python", degradable=False)
        self.assertGreater(len(res), 0, "method actually exists")

    def test_bug_2072(self):
        from ..diffmap import DiffMap
        d = DiffMap()
        d.use_gpu # used to raise AttributeError
        d.use_gpu = True # used to raise AttributeError

    def test_bug_2151(self):
        """Some detector fail to integrate in 2D, the CSC matrix produced by cython has wrong shape.
        Faulty detectors: S10
        """
        ai = load({"detector": "imxpad_s10"})
        img = numpy.ones(ai.detector.shape)
        ai.integrate2d(img, 10, method=("full","csc","python"), unit="r_mm")
        #used to raise AssertionError assert self.size == len(indptr) - 1

    def test_bug_2525(self):
        """
        res1d_cp = copy.copy(res1d)
        used to raise TypeError: missing required positional argument
        """
        ai = load({"detector": "imxpad_s10"})
        img = numpy.ones(ai.detector.shape)
        res1d = ai.integrate1d(img, 10, unit="r_mm")

        res1d_cp = copy.copy(res1d)
        self.assertTrue(numpy.allclose(res1d.radial, res1d_cp.radial))
        self.assertTrue(numpy.allclose(res1d.intensity, res1d_cp.intensity))
        self.assertTrue(numpy.allclose(res1d.sum_signal, res1d_cp._sum_signal))
        self.assertTrue(numpy.allclose(res1d.sum_normalization, res1d_cp._sum_normalization))
        res1d_dp = copy.deepcopy(res1d)
        self.assertTrue(numpy.allclose(res1d.radial, res1d_dp.radial))
        self.assertTrue(numpy.allclose(res1d.intensity, res1d_dp.intensity))
        self.assertTrue(numpy.allclose(res1d.sum_signal, res1d_dp.sum_signal))
        self.assertTrue(numpy.allclose(res1d.sum_normalization, res1d_dp.sum_normalization))

        res2d = ai.integrate2d(img, 10, unit="r_mm")
        res2d_cp = copy.copy(res2d)
        self.assertTrue(numpy.allclose(res2d.radial, res2d_cp.radial))
        self.assertTrue(numpy.allclose(res2d.intensity, res2d_cp.intensity))
        self.assertTrue(numpy.allclose(res2d.sum_signal, res2d_cp.sum_signal))
        self.assertTrue(numpy.allclose(res2d.sum_normalization, res2d_cp.sum_normalization))
        res2d_dp = copy.deepcopy(res2d)
        self.assertTrue(numpy.allclose(res2d.radial, res2d_dp.radial))
        self.assertTrue(numpy.allclose(res2d.intensity, res2d_dp.intensity))
        self.assertTrue(numpy.allclose(res2d.sum_signal, res2d_dp.sum_signal))
        self.assertTrue(numpy.allclose(res2d.sum_normalization, res2d_dp.sum_normalization))

        ressp = ai.separate(img, 10, unit="r_mm")
        ressp_cp = copy.copy(ressp)
        self.assertTrue(numpy.allclose(ressp.bragg, ressp_cp.bragg))
        self.assertTrue(numpy.allclose(ressp.amorphous, ressp_cp.amorphous))
        self.assertTrue(numpy.allclose(ressp.sum_signal, ressp_cp.sum_signal))
        self.assertTrue(numpy.allclose(ressp.sum_normalization, ressp_cp.sum_normalization))

        ressp_dp = copy.deepcopy(ressp)
        self.assertTrue(numpy.allclose(ressp.bragg, ressp_dp.bragg))
        self.assertTrue(numpy.allclose(ressp.amorphous, ressp_dp.amorphous))
        self.assertTrue(numpy.allclose(ressp.sum_signal, ressp_dp.sum_signal))
        self.assertTrue(numpy.allclose(ressp.sum_normalization, ressp_dp.sum_normalization))

        from pyFAI.containers import SparseFrame
        sp = SparseFrame(numpy.arange(1,5),numpy.arange(4,9))
        sp._shape = (23,45)
        sp_cp = copy.copy(sp)
        self.assertTrue(numpy.allclose(sp.index, sp_cp.index))
        self.assertTrue(numpy.allclose(sp.intensity, sp_cp.intensity))
        self.assertEqual(sp.shape, sp_cp.shape)

        sp_dp = copy.deepcopy(sp)
        self.assertTrue(numpy.allclose(sp.index, sp_dp.index))
        self.assertTrue(numpy.allclose(sp.intensity, sp_dp.intensity))
        self.assertEqual(sp.shape, sp_dp.shape)

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    @unittest.skipUnless(ocl, "PyOpenCl is missing")
    def test_bug_2538(self):
        """ This bug is creating an infinite loop when some bins have no contributing pixels"""
        ai1 = load({"detector":"pilatus100k"})
        r = ai1.array_from_unit(unit="r_mm")
        ai1.detector.mask = r>60
        ai1.detector.mask[-1,-1] = 0 # this exposes the pixel in the corner !
        img = numpy.ones(ai1.detector.shape)
        ref = ai1.medfilt1d_ng(img, 100, unit="r_mm", method=("no","csr","cython"))
        res = ai1.medfilt1d_ng(img, 100, unit="r_mm", method=("no","csr","opencl"))
        # raise RuntimeError("infinite loop")
        self.assertTrue(numpy.allclose(ref[0], res[0]))
        self.assertTrue(numpy.allclose(ref[1], res[1]))

class TestBug1703(unittest.TestCase):
    """
    Check the normalization affect propely the propagated errors/intensity
    """

    @classmethod
    def setUpClass(cls):
        pix = 100e-6
        shape = (1024, 1024)
        npt = 1000
        wl = 1e-10
        I0 = 1e2
        unit = "q_nm^-1"
        cls.kwargs = {"npt":npt,
         "correctSolidAngle":False,
         "polarization_factor":None,
         "safe":False,
         "unit": unit
         }
        cls.methods = [("no", "csr", "python"),  # Those methods should be passing this test
                       # ("no", "csr", "cython"),  # Known broken
                       # ("no", "csr", "opencl"),  # Known broken
                       ]
        detector = detectors.Detector(pix, pix)
        detector.shape = detector.max_shape = shape

        ai_init = {"dist":1.0,
           "poni1":0.0,
           "poni2":0.0,
           "rot1":-0.05,
           "rot2":+0.05,
           "rot3":0.0,
           "detector":detector,
           "wavelength":wl}
        cls.ai = AzimuthalIntegrator(**ai_init)

        cls.q = numpy.linspace(0, cls.ai.array_from_unit(unit=unit).max(), npt)
        cls.I = I0 / (1 + cls.q ** 2)
        img_theo = cls.ai.calcfrom1d(cls.q, cls.I, dim1_unit=unit,
                         correctSolidAngle=True,
                         polarization_factor=None)
        cls.img = UtilsTest.get_rng().poisson(img_theo)

    @classmethod
    def tearDownClass(cls):
        cls.kwargs = cls.img = cls.methods = None
        cls.q = cls.I = None

    def test_integration(self):
        factor = 1e-4
        for method in self.methods:
            k = self.kwargs.copy()
            k["method"] = method
            ka = k.copy()
            ka["error_model"] = "azimuthal"
            kp = k.copy()
            kp["error_model"] = "poisson"
            res_azim_1 = self.ai.integrate1d(self.img, **ka)
            res_azim_f = self.ai.integrate1d(self.img, normalization_factor=factor, **ka)
            res_pois_f = self.ai.integrate1d(self.img, normalization_factor=factor, **kp)
            res_pois_1 = self.ai.integrate1d(self.img, **kp)
            # Check the intensity
            self.assertLess(mathutil.rwp(res_pois_1, (self.q, self.I), scale=1), 5, f"intensity Poisson, unscaled, {method}")
            self.assertLess(mathutil.rwp(res_pois_f, (self.q, self.I), scale=factor), 5, f"intensity Poisson, scaled, {method}")
            self.assertLess(mathutil.rwp(res_azim_1, (self.q, self.I), scale=1), 5, f"intensity Azimuthal, unscaled, {method}")
            self.assertLess(mathutil.rwp(res_azim_f, (self.q, self.I), scale=factor), 5, f"intensity Azimuthal, unscaled, {method}")

            self.assertLess(mathutil.rwp((res_pois_1[0], res_pois_1[2]), (res_pois_1[0], res_pois_1[2]), scale=1), 5, f"sigma Poisson, unscaled, {method}")
            self.assertLess(mathutil.rwp((res_pois_f[0], res_pois_f[2]), (res_pois_1[0], res_pois_1[2]), scale=factor), 5, f"sigma Poisson, scaled, {method}")
            self.assertLess(mathutil.rwp((res_azim_1[0], res_azim_1[2]), (res_pois_1[0], res_pois_1[2]), scale=1), 5, f"sigma Azimuthal, unscaled, {method}")
            self.assertLess(mathutil.rwp((res_azim_f[0], res_azim_f[2]), (res_pois_1[0], res_pois_1[2]), scale=factor), 5, f"sigma Azimuthal, unscaled, {method}")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBug170))
    testsuite.addTest(loader(TestBug211))
    testsuite.addTest(loader(TestBugRegression))
    testsuite.addTest(loader(TestBug1703))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
