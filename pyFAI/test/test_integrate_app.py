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


import json
import os
import fabio
import contextlib
import unittest
import pyFAI.app.integrate
import tempfile
from .utilstest import UtilsTest
import numpy


class TestIntegrateApp(unittest.TestCase):

    class Options(object):

        def __init__(self):
            self.version = None
            self.verbose = False
            self.output = None
            self.format = None
            self.slow = None
            self.rapid = None
            self.gui = False
            self.json = ".azimint.json"
            self.monitor_key = None

    @contextlib.contextmanager
    def jsontempfile(self, ponipath, nbpt_azim=1):
        data = {}
        data["poni"] = ponipath
        data["wavelength"] = 1
        data["nbpt_rad"] = 3
        data["nbpt_azim"] = nbpt_azim
        data["do_2D"] = nbpt_azim > 1
        fd, path = tempfile.mkstemp(prefix="pyfai_", suffix=".json")
        os.close(fd)
        with open(path, 'w') as fp:
            json.dump(data, fp)
        yield path
        os.remove(path)

    @contextlib.contextmanager
    def datatempfile(self, data, header):
        fd, path = tempfile.mkstemp(prefix="pyfai_", suffix=".edf")
        os.close(fd)
        img = fabio.edfimage.edfimage(data, header)
        img.save(path)
        img = None
        yield path
        os.remove(path)

    @contextlib.contextmanager
    def resulttempfile(self):
        fd, path = tempfile.mkstemp(prefix="pyfai_", suffix=".out")
        os.close(fd)
        os.remove(path)
        yield path
        os.remove(path)

    def test_integrate_default_output_dat(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        with self.datatempfile(data, {}) as datapath:
            with self.jsontempfile(ponifile) as jsonpath:
                options.json = jsonpath
                pyFAI.app.integrate.integrate_shell(options, [datapath])
                self.assertTrue(os.path.exists(datapath[:-4] + ".dat"))

    def test_integrate_default_output_azim(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        with self.datatempfile(data, {}) as datapath:
            with self.jsontempfile(ponifile, nbpt_azim=2) as jsonpath:
                options.json = jsonpath
                pyFAI.app.integrate.integrate_shell(options, [datapath])
                self.assertTrue(os.path.exists(datapath[:-4] + ".azim"))

    def test_integrate_file_output_dat(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        with self.datatempfile(data, {}) as datapath:
            with self.jsontempfile(ponifile) as jsonpath:
                options.json = jsonpath
                with self.resulttempfile() as resultpath:
                    options.output = resultpath
                    pyFAI.app.integrate.integrate_shell(options, [datapath])
                    self.assertTrue(os.path.exists(resultpath))

    def test_integrate_no_monitor(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = numpy.array([[2.0, 17.0], [2.0, 26.0], [2., 0.]])
        with self.datatempfile(data, {}) as datapath:
            with self.jsontempfile(ponifile) as jsonpath:
                options.json = jsonpath
                with self.resulttempfile() as resultpath:
                    options.output = resultpath
                    pyFAI.app.integrate.integrate_shell(options, [datapath])
                    result = numpy.loadtxt(resultpath)
                    numpy.testing.assert_almost_equal(result, expected, decimal=1)

    def test_integrate_monitor(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        options.monitor_key = "my_mon"
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = numpy.array([[2.0, 33.9], [2.0, 52.0], [2., 0.]])
        with self.datatempfile(data, {"my_mon": "0.5"}) as datapath:
            with self.jsontempfile(ponifile) as jsonpath:
                options.json = jsonpath
                with self.resulttempfile() as resultpath:
                    options.output = resultpath
                    pyFAI.app.integrate.integrate_shell(options, [datapath])
                    result = numpy.loadtxt(resultpath)
                    numpy.testing.assert_almost_equal(result, expected, decimal=1)

    def test_integrate_counter_monitor(self):
        ponifile = UtilsTest.getimage("Pilatus1M.poni")
        options = self.Options()
        options.monitor_key = "counter/my_mon"
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = numpy.array([[2.0, 8.5], [2.0, 13.0], [2., 0.]])
        with self.datatempfile(data, {"counter_mne": "my_mon", "counter_pos": "2.0"}) as datapath:
            with self.jsontempfile(ponifile) as jsonpath:
                options.json = jsonpath
                with self.resulttempfile() as resultpath:
                    options.output = resultpath
                    pyFAI.app.integrate.integrate_shell(options, [datapath])
                    result = numpy.loadtxt(resultpath)
                    numpy.testing.assert_almost_equal(result, expected, decimal=1)


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestIntegrateApp))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
