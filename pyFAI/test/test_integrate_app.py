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
import unittest
import numpy
import shutil
import h5py

import pyFAI.app.integrate
from .utilstest import UtilsTest
from pyFAI.io import integration_config


class TestIntegrateApp(unittest.TestCase):

    def setUp(self):
        self.tempDir = os.path.join(UtilsTest.tempdir, self.id())
        os.makedirs(self.tempDir)

    def tearDown(self):
        shutil.rmtree(self.tempDir)
        self.tempDir = None

    class Options(object):

        def __init__(self):
            self.version = None
            self.verbose = False
            self.output = None
            self.format = None
            self.slow = None
            self.rapid = None
            self.gui = False
            self.json = None
            self.monitor_key = None
            self.write_mode = None

    def get_path(self, filename):
        path = os.path.join(self.tempDir, filename)
        return path

    def create_edf_file(self, filename, data, header={}):
        path = os.path.join(self.tempDir, filename)
        if isinstance(data, list):
            image = fabio.edfimage.EdfImage(data=data[0], header=header)
            for d in data[1:]:
                image.appendFrame(data=d, header=header)
        else:
            image = fabio.edfimage.EdfImage(data=data, header=header)
        image.save(path)
        return path

    def is_file_exists(self, filename):
        path = os.path.join(self.tempDir, filename)
        return os.path.exists(path)

    def create_json(self, ponipath=None, nbpt_azim=1):
        if ponipath is None:
            ponipath = UtilsTest.getimage("Pilatus1M.poni")
        data = {"poni": ponipath}
        integration_config.normalize(data, inplace=True)
        data["wavelength"] = 1
        data["nbpt_rad"] = 3
        data["nbpt_azim"] = nbpt_azim
        data["do_2D"] = nbpt_azim > 1
        data["method"] = ("bbox", "histogram", "cython")
        path = os.path.join(self.tempDir, "config.json")
        with open(path, 'w') as fp:
            json.dump(data, fp)
        return path

    def create_h5_cube_file(self, filename, datapath, data, monitor=None):
        path = os.path.join(self.tempDir, filename)
        with h5py.File(path) as h5:
            h5[datapath] = numpy.array(data)
            if monitor is not None:
                h5["/monitor/data"] = numpy.array(monitor)
        return path

    def test_path(self):
        path = os.path.join(self.tempDir)
        return path

    def create_result_path(self, name="result"):
        path = os.path.join(self.tempDir, self.id(), name)
        os.makedirs(path)
        return path

    def test_integrate_default_output_dat(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data)
        options.json = self.create_json()
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        self.assertTrue(os.path.exists(datapath[:-4] + ".dat"))

    def test_integrate_default_output_azim(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data)
        options.json = self.create_json(nbpt_azim=2)
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        self.assertTrue(os.path.exists(datapath[:-4] + ".azim"))

    def test_integrate_file_output_dat(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data)
        options.json = self.create_json()
        options.output = self.create_result_path("foo")
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        self.assertTrue(os.path.exists(options.output))
        result_file = os.path.join(options.output, "data.dat")
        self.assertTrue(os.path.exists(result_file))

    def test_integrate_no_monitor(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data)
        expected = numpy.array([[2.0, 17.0], [2.0, 26.0], [2., 0.]])
        options.json = self.create_json()
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        result = numpy.loadtxt(self.get_path("data.dat"))
        numpy.testing.assert_almost_equal(result, expected, decimal=1)

    def test_integrate_monitor(self):
        options = self.Options()
        options.monitor_key = "my_mon"
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data, header={"my_mon": "0.5"})
        expected = numpy.array([[2.0, 33.9], [2.0, 52.0], [2., 0.]])
        options.json = self.create_json()
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        result = numpy.loadtxt(self.get_path("data.dat"))
        numpy.testing.assert_almost_equal(result, expected, decimal=1)

    def test_integrate_counter_monitor(self):
        options = self.Options()
        options.monitor_key = "counter/my_mon"
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        datapath = self.create_edf_file("data.edf", data, header={"counter_mne": "my_mon", "counter_pos": "2.0"})
        expected = numpy.array([[2.0, 8.5], [2.0, 13.0], [2., 0.]])
        options.json = self.create_json()
        pyFAI.app.integrate.integrate_shell(options, [datapath])
        result = numpy.loadtxt(self.get_path("data.dat"))
        numpy.testing.assert_almost_equal(result, expected, decimal=1)

    def test_integrate_with_hdf5_monitor(self):
        options = self.Options()
        options.monitor_key = "counter/my_mon"
        coef = numpy.array([2.0, 4.0, 10.0, -20.0])
        data = numpy.array([[0, 0], [0, 100], [0, 0]]) * coef.reshape(-1, 1, 1)
        file1 = self.create_h5_cube_file("data.h5", "image/data",
                                         data=data,
                                         monitor=coef)
        expected = numpy.array([17.0, 26.0, 0.0])
        options.json = self.create_json()
        options.monitor_key = "/monitor/data"
        options.output = os.path.join(self.test_path(), "result.h5")

        pyFAI.app.integrate.integrate_shell(options, [file1 + "::image/data"])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            result = h5["entry_0000/integrate/results/data"]
            self.assertEquals(result.shape[0], 4)
            for iframe in range(result.shape[0]):
                numpy.testing.assert_almost_equal(result[iframe], expected, decimal=1)

    def test_fileseries_to_h5(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        file2 = self.create_edf_file("data2.edf", data)
        options.json = self.create_json()
        options.output = os.path.join(self.test_path(), "result.h5")
        pyFAI.app.integrate.integrate_shell(options, [file1, file2])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["entry_0000/integrate/results/data"].shape[0], 2)

    def test_multiframes_to_h5(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", [data, data])
        options.json = self.create_json()
        options.output = os.path.join(self.test_path(), "result.h5")
        pyFAI.app.integrate.integrate_shell(options, [file1])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["entry_0000/integrate/results/data"].shape[0], 2)

    def test_multiframes_fileseries_to_h5(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", [data, data])
        file2 = self.create_edf_file("data2.edf", [data, data])
        options.json = self.create_json()
        options.output = os.path.join(self.test_path(), "result.h5")
        pyFAI.app.integrate.integrate_shell(options, [file1, file2])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["entry_0000/integrate/results/data"].shape[0], 4)

    def test_h5_to_h5(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_h5_cube_file("data.h5", "image/data", [data, data, data, data])
        options.json = self.create_json()
        options.output = os.path.join(self.test_path(), "result.h5")
        pyFAI.app.integrate.integrate_shell(options, [file1 + "::image/data"])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["entry_0000/integrate/results/data"].shape[0], 4)

    def test_fileseries_to_h5_entry(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        file2 = self.create_edf_file("data2.edf", data)
        options.json = self.create_json()
        output_file = os.path.join(self.test_path(), "result.h5")
        options.output = output_file + "::my/entry"
        pyFAI.app.integrate.integrate_shell(options, [file1, file2])
        self.assertTrue(os.path.exists(output_file))
        with h5py.File(output_file, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["my/entry/integrate/results/data"].shape[0], 2)

    def test_write_h5_delete_mode(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        options.json = self.create_json()
        options.write_mode = pyFAI.app.integrate.HDF5Writer.MODE_DELETE
        options.output = os.path.join(self.test_path(), "result.h5")
        with h5py.File(options.output, mode="w") as h5:
            # Store something
            h5["entry_0000"] = 10
        pyFAI.app.integrate.integrate_shell(options, [file1])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["/entry_0000/integrate/results/data"].shape[0], 1)

    def test_write_h5_overwrite_mode(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        options.json = self.create_json()
        options.write_mode = pyFAI.app.integrate.HDF5Writer.MODE_OVERWRITE
        options.output = os.path.join(self.test_path(), "result.h5")
        with h5py.File(options.output, mode="w") as h5:
            # Store something
            h5["entry_0000"] = 10
            h5["my_entry"] = 10
        pyFAI.app.integrate.integrate_shell(options, [file1])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["/entry_0000/integrate/results/data"].shape[0], 1)
            self.assertEquals(h5["/my_entry"][()], 10)

    def test_write_h5_append_mode(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        options.json = self.create_json()
        options.write_mode = pyFAI.app.integrate.HDF5Writer.MODE_APPEND
        options.output = os.path.join(self.test_path(), "result.h5")
        with h5py.File(options.output, mode="w") as h5:
            # Store something
            h5["entry_0000"] = 10
        pyFAI.app.integrate.integrate_shell(options, [file1])
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["/entry_0000"][()], 10)
            self.assertEquals(h5["/entry_0001/integrate/results/data"].shape[0], 1)

    def test_write_h5_error_mode(self):
        options = self.Options()
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        file1 = self.create_edf_file("data1.edf", data)
        options.json = self.create_json()
        options.write_mode = pyFAI.app.integrate.HDF5Writer.MODE_ERROR
        options.output = os.path.join(self.test_path(), "result.h5")
        with h5py.File(options.output, mode="w") as h5:
            # Store something
            h5["entry_0000"] = 10
        result = pyFAI.app.integrate.integrate_shell(options, [file1])
        self.assertNotEqual(result, 0)
        self.assertTrue(os.path.exists(options.output))
        with h5py.File(options.output, mode="r") as h5:
            self.assertIsNotNone(h5)
            self.assertEquals(h5["/entry_0000"][()], 10)
            self.assertNotIn("/entry_0001", h5)


class _ResultObserver(pyFAI.app.integrate.IntegrationObserver):

    def __init__(self):
        super(_ResultObserver, self).__init__()
        self.result = []

    def data_result(self, data_id, result):
        self.result.append(result)


class TestProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = {"poni": UtilsTest.getimage("Pilatus1M.poni")}
        integration_config.normalize(config, inplace=True)
        cls.base_config = config

    def setUp(self):
        self.tempDir = os.path.join(UtilsTest.tempdir, self.id())
        os.makedirs(self.tempDir)

    def tearDown(self):
        shutil.rmtree(self.tempDir)
        self.tempDir = None

    def test_process_no_data(self):
        config = {"nbpt_rad": 2}
        config.update(self.base_config)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 0)

    def test_process_numpy_1d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected_result = [23.5, 9.9]
        expected_radial = [1.9, 1.9]
        params = {"do_2D": False,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 1)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)

    def test_process_numpy_2d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected_result = [[5.6, 4.5], [41.8, 9.3]]
        expected_radial = [2.0, 2.0]
        expected_azimuthal = [-124.5, -124.2]
        params = {"do_2D": True,
                  "nbpt_azim": 2,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 1)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)
        numpy.testing.assert_array_almost_equal(result.azimuthal, expected_azimuthal, decimal=1)

    def test_process_numpy_3d(self):
        data = numpy.array([[[0, 0], [0, 100], [0, 0]], [[0, 0], [0, 200], [0, 0]]])
        expected_result = [[5.6, 4.5], [41.8, 9.3]]
        expected_radial = [2.0, 2.0]
        expected_azimuthal = [-124.5, -124.2]
        params = {"do_2D": True,
                  "nbpt_azim": 2,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 2)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)
        numpy.testing.assert_array_almost_equal(result.azimuthal, expected_azimuthal, decimal=1)

    def test_fabio_integration1d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        data = fabio.numpyimage.NumpyImage(data=data)
        expected_result = [23.5, 9.9]
        expected_radial = [1.9, 1.9]
        params = {"do_2D": False,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 1)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)

    def test_fabio_multiframe_integration2d(self):
        data1 = numpy.array([[0, 0], [0, 100], [0, 0]])
        data2 = numpy.array([[0, 0], [0, 200], [0, 0]])
        data = fabio.edfimage.EdfImage(data=data1)
        data.appendFrame(data=data2)

        expected_result = [[5.6, 4.5], [41.8, 9.3]]
        expected_radial = [2.0, 2.0]
        expected_azimuthal = [-124.5, -124.2]
        params = {"do_2D": True,
                  "nbpt_azim": 2,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 2)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)
        numpy.testing.assert_array_almost_equal(result.azimuthal, expected_azimuthal, decimal=1)

    def test_unsupported_types(self):
        params = {"do_2D": True,
                  "nbpt_azim": 2,
                  "nbpt_rad": 2,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process(["##unexisting_file##.edf", 10], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 0)

    def test_normalization_monitor_name(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        header = {"monitor": 0.5}
        data = fabio.numpyimage.NumpyImage(data=data, header=header)
        expected_result = [47.0, 19.8]
        expected_radial = [1.9, 1.9]
        params = {"do_2D": False,
                  "nbpt_rad": 2,
                  "monitor_name": "monitor",
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 1)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)

    def test_normalization_factor_monitor_name(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        header = {"monitor": 0.5}
        data = fabio.numpyimage.NumpyImage(data=data, header=header)
        expected_result = [4695.9, 1984.5]
        expected_radial = [1.9, 1.9]
        params = {"do_2D": False,
                  "nbpt_rad": 2,
                  "monitor_name": "monitor",
                  "normalization_factor": 0.01,
                  "method": ("bbox", "histogram", "cython")}
        config = self.base_config.copy()
        config.update(params)
        observer = _ResultObserver()
        pyFAI.app.integrate.process([data], self.tempDir, config, monitor_name=None, observer=observer)
        self.assertEqual(len(observer.result), 1)
        result = observer.result[0]
        numpy.testing.assert_array_almost_equal(result.intensity, expected_result, decimal=1)
        numpy.testing.assert_array_almost_equal(result.radial, expected_radial, decimal=1)


class TestMain(unittest.TestCase):

    def callable(self, *args, **kwargs):
        pass

    def setUp(self):
        self._gui = pyFAI.app.integrate.integrate_gui
        self._shell = pyFAI.app.integrate.integrate_shell
        pyFAI.app.integrate.integrate_gui = self.callable
        pyFAI.app.integrate.integrate_shell = self.callable

    def tearDown(self):
        pyFAI.app.integrate.integrate_gui = self._gui
        pyFAI.app.integrate.integrate_shell = self._shell

    def test(self):
        pyFAI.app.integrate._main(["myimage.edf"])


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestIntegrateApp))
    testsuite.addTest(loader(TestProcess))
    testsuite.addTest(loader(TestMain))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
