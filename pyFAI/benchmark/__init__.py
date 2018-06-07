#!/usr/bin/env python
# coding: utf-8
#
#    Copyright (C) 2016-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"Benchmark for Azimuthal integration of PyFAI"


from __future__ import print_function, division


__author__ = "Jérôme Kieffer"
__date__ = "07/06/2018"
__license__ = "MIT"
__copyright__ = "2012-2017 European Synchrotron Radiation Facility, Grenoble, France"


from collections import OrderedDict
import json
import sys
import time
import timeit
import os
import platform
import subprocess
import fabio
import os.path as op

# To use use the locally build version of PyFAI, use ../bootstrap.py

from .. import load
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..utils import mathutil
from ..test import utilstest
from ..opencl import pyopencl, ocl
from ..third_party import six
try:
    from ..gui.matplotlib import pylab
    from ..gui.utils import update_fig
except ImportError:
    pylab = None

    def update_fig(*args, **kwargs):
        pass


ds_list = ["Pilatus1M.poni",
           "halfccd.poni",
           "Frelon2k.poni",
           "Pilatus6M.poni",
           "Mar3450.poni",
           "Fairchild.poni"]


datasets = {"Fairchild.poni": "Fairchild.edf",
            "halfccd.poni": "halfccd.edf",
            "Frelon2k.poni": "Frelon2k.edf",
            "Pilatus6M.poni": "Pilatus6M.cbf",
            "Pilatus1M.poni": "Pilatus1M.edf",
            "Mar3450.poni": "LaB6_260210.mar3450"
            }

PONIS = {
    "Pilatus6M.poni": {'dist': 0.3, 'poni2': 0.2115772, 'poni1': 0.225406, 'detector': 'Pilatus6M'},
    "Fairchild.poni": {'dist': 0.0882065396596, 'poni2': 0.0449457803015, 'rot1': -0.506766875792, 'rot3': -1.13774685128e-05, 'rot2': 0.0167069809441, 'poni1': 0.0302286347503, 'detector': 'Fairchild'},
    "halfccd.poni": {'dist': 0.0994744403007, 'poni2': 0.0481217639198, 'rot1': -0.000125830018938, 'rot3': 1.57079531561, 'rot2': -0.0160719674782, 'poni1': 0.026453455358, 'pixel2': 4.684483e-05, 'pixel1': 4.8422519999999994e-05},
    "Pilatus1M.poni": {'dist': 1.58323111834, 'poni2': 0.0412277798782, 'rot1': 0.00648735642526, 'rot3': 4.12987220385e-08, 'rot2': 0.00755810191106, 'poni1': 0.0334170169115, 'detector': 'Pilatus1M'},
    "Mar3450.poni": {'dist': 0.222549826201, 'poni2': 0.172625538874, 'rot1': 0.00164880041469, 'rot3': -1.43412739468e-08, 'rot2': 0.0438631777747, 'wavelength': 3.738e-11, 'splineFile': None, 'poni1': 0.161137340974, 'detector': 'Mar345'},
    "Frelon2k.poni": {'dist': 0.1057363, 'poni2': 0.05660461, 'rot1': 0.027767, 'rot3': -1.8e-05, 'rot2': 0.016991, 'poni1': 0.05301968, 'pixel2': 4.722437999999999e-05, 'pixel1': 4.6831519999999995e-05}
}

# Handle to the Bench instance: allows debugging from outside if needed
bench = None


class BenchTest(object):
    """Generic class for benchmarking with `timeit.Timer`"""

    def setup(self):
        """Setup.

        The method do not have arguments. Everything must be set before, from
        the constructor for example.
        """
        pass

    def stmt(self):
        """Statement.

        The method do not have arguments. Everything must be set before, from
        the constructor, loaded from the `setup` to a class attribute.
        """
        pass

    def setup_and_stmt(self):
        """Execute the setup then the statement."""
        self.setup()
        return self.stmt()

    def clean(self):
        """Clean up stored data"""
        pass


class BenchTest1D(BenchTest):
    """Test 1d integration"""

    def __init__(self, azimuthal_params, file_name, unit, method):
        BenchTest.__init__(self)
        self.azimuthal_params = azimuthal_params
        self.file_name = file_name
        self.unit = unit
        self.method = method

    def setup(self):
        self.ai = AzimuthalIntegrator(**self.azimuthal_params)
        self.data = fabio.open(self.file_name).data
        self.N = min(self.data.shape)

    def stmt(self):
        return self.ai.integrate1d(self.data, self.N, safe=False, unit=self.unit, method=self.method)

    def clean(self):
        self.ai = None
        self.data = None


class BenchTest2D(BenchTest):
    """Test 2d integration"""

    def __init__(self, azimuthal_params, file_name, unit, method, output_size):
        BenchTest.__init__(self)
        self.azimuthal_params = azimuthal_params
        self.file_name = file_name
        self.unit = unit
        self.method = method
        self.output_size = output_size

    def setup(self):
        self.ai = AzimuthalIntegrator(**self.azimuthal_params)
        self.data = fabio.open(self.file_name).data
        self.N = self.output_size

    def stmt(self):
        return self.ai.integrate2d(self.data, self.output_size[0], self.output_size[1], unit=self.unit, method=self.method)

    def clean(self):
        self.ai = None
        self.data = None


class BenchTestGpu(BenchTest):
    """Test XRPD in OpenCL"""

    def __init__(self, azimuthal_params, file_name, devicetype, useFp64, platformid, deviceid):
        BenchTest.__init__(self)
        self.azimuthal_params = azimuthal_params
        self.file_name = file_name
        self.devicetype = devicetype
        self.useFp64 = useFp64
        self.platformid = platformid
        self.deviceid = deviceid

    def setup(self):
        self.ai = load(self.azimuthal_params)
        self.data = fabio.open(self.file_name).data
        self.N = min(self.data.shape)
        self.ai.xrpd_OpenCL(self.data, self.N, devicetype=self.devicetype, useFp64=self.useFp64, platformid=self.platformid, deviceid=self.deviceid)

    def stmt(self):
        self.ai.xrpd_OpenCL(self.data, self.N, safe=False)

    def clean(self):
        self.ai = None
        self.data = None


class Bench(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    LABELS = {"splitBBox": "CPU_serial",
              "lut": "CPU_LUT_OpenMP",
              "lut_ocl": "LUT",
              "csr": "CPU_CSR_OpenMP",
              "csr_ocl": "CSR",
              }

    def __init__(self, nbr=10, repeat=1, memprofile=False, unit="2th_deg", max_size=None):
        self.reference_1d = {}
        self.LIMIT = 8
        self.repeat = repeat
        self.nbr = nbr
        self.results = OrderedDict()
        self.meth = []
        self._cpu = None
        self.fig = None
        self.ax = None
        self.starttime = time.time()
        self.plot = None
        self.plot_x = []
        self.plot_y = []
        self.do_memprofile = memprofile
        self.fig_mp = None
        self.ax_mp = None
        self.plot_mp = None
        self.memory_profile = ([], [])
        self.unit = unit
        self.out_2d = (500, 360)
        self.max_size = max_size or sys.maxunicode

    def get_cpu(self):
        if self._cpu is None:
            if os.name == "nt":
                self._cpu = platform.processor()
            elif os.path.exists("/proc/cpuinfo"):
                cpuinfo = [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("model name")]
                if not cpuinfo:
                    cpuinfo = [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("cpu")]
                self._cpu = cpuinfo[0].strip()
            elif os.path.exists("/usr/sbin/sysctl"):
                proc = subprocess.Popen(["sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE)
                proc.wait()
                self._cpu = proc.stdout.read().strip()
                if six.PY3:
                    self._cpu = self._cpu.decode("ASCII")
            old = self._cpu
            self._cpu = old.replace("  ", " ")
            while old != self._cpu:
                old = self._cpu
                self._cpu = old.replace("  ", " ")
        return self._cpu

    def get_gpu(self, devicetype="gpu", useFp64=False, platformid=None, deviceid=None):
        if ocl is None:
            return "NoGPU"
        try:
            ctx = ocl.create_context(devicetype, useFp64, platformid, deviceid)
        except Exception:
            return "NoGPU"
        else:
            return ctx.devices[0].name

    def get_mem(self):
        """
        Returns the occupied memory for memory-leak hunting in MByte
        """
        pid = os.getpid()
        if os.path.exists("/proc/%i/status" % pid):
            for l in open("/proc/%i/status" % pid):
                if l.startswith("VmRSS"):
                    mem = int(l.split(":", 1)[1].split()[0]) / 1024.
        else:
            mem = 0
        return mem

    def print_init(self, t):
        print(" * Initialization time: %.1f ms" % (1000.0 * t))
        self.update_mp()

    def print_exec(self, t):
        print(" * Execution time rep : %.1f ms" % (1000.0 * t))
        self.update_mp()

    def print_sep(self):
        print("*" * 80)
        self.update_mp()

    def get_ref(self, param):
        if param not in self.reference_1d:
            file_name = utilstest.UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest1D(poni, file_name, self.unit, "splitBBox")
            bench_test.setup()
            res = bench_test.stmt()
            self.reference_1d[param] = res
            bench_test.clean()
        return self.reference_1d[param]

    def bench_1d(self, method="splitBBox", check=False, opencl=None):
        """
        :param method: method to be bechmarked
        :param check: check results vs ref if method is LUT based
        :param opencl: dict containing platformid, deviceid and devicetype
        """
        self.update_mp()
        if opencl:
            if (ocl is None):
                print("No pyopencl")
                return
            if (opencl.get("platformid") is None) or (opencl.get("deviceid") is None):
                platdev = ocl.select_device(opencl.get("devicetype"))
                if not platdev:
                    print("No such OpenCL device: skipping benchmark")
                    return
                platformid, deviceid = opencl["platformid"], opencl["deviceid"] = platdev
            else:
                platformid, deviceid = opencl["platformid"], opencl["deviceid"]
            devicetype = opencl["devicetype"] = ocl.platforms[platformid].devices[deviceid].type
            platform = str(ocl.platforms[platformid]).split()[0]
            if devicetype == "CPU":
                cpu_name = (str(ocl.platforms[platformid].devices[deviceid]).split("@")[0]).split()
                device = ""
                while cpu_name and len(device) < 5:
                    device = cpu_name.pop() + "" + device
            else:
                device = ' '.join(str(ocl.platforms[platformid].devices[deviceid]).split())
            print("Working on device: %s platform: %s device: %s" % (devicetype, platform, device))
            label = ("1D %s %s %s %s" % (devicetype, self.LABELS[method], platform, device)).replace(" ", "_")
            method += "_%i,%i" % (opencl["platformid"], opencl["deviceid"])
            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)
        else:
            print("Working on processor: %s" % self.get_cpu())
            label = "1D_" + self.LABELS[method]
            memory_error = (MemoryError, RuntimeError)
        results = OrderedDict()
        first = True
        for param in ds_list:
            self.update_mp()
            file_name = utilstest.UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest1D(poni, file_name, self.unit, method)
            bench_test.setup()
            size = bench_test.data.size / 1.0e6
            if size > self.max_size:
                continue
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(file_name), size, bench_test.N))
            try:
                t0 = time.time()
                res = bench_test.stmt()
                self.print_init(time.time() - t0)
            except memory_error as error:
                print(error)
                break
            self.update_mp()
            if check:
                module = sys.modules.get(AzimuthalIntegrator.__module__)
                if module:
                    if "lut" in method:
                        key = module.EXT_LUT_ENGINE
                    elif "csr" in method:
                        key = module.EXT_CSR_ENGINE
                    else:
                        key = None
                if key and module:
                    try:
                        integrator = bench_test.ai.engines.get(key).engine
                    except MemoryError as error:
                        print(error)
                    else:
                        if "lut" in method:
                            print("lut: shape= %s \t nbytes %.3f MB " % (integrator.lut.shape, integrator.lut_nbytes / 2 ** 20))
                        else:
                            print("csr: size= %s \t nbytes %.3f MB " % (integrator.data.size, integrator.lut_nbytes / 2 ** 20))
            bench_test.clean()
            self.update_mp()
            try:
                t = timeit.Timer(bench_test.stmt, bench_test.setup_and_stmt)
                tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            except memory_error as error:
                print(error)
                break
            self.update_mp()
            self.print_exec(tmin)
            tmin *= 1000.0
            if check:
                ref = self.get_ref(param)
                R = mathutil.rwp(res, ref)
                print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
                self.update_mp()
                if R < self.LIMIT:
                    results[size] = tmin
                    self.update_mp()
                    if first:
                        if opencl:
                            self.new_curve(results, label, style="--")
                        else:
                            self.new_curve(results, label, style="-")
                        first = False
                    else:
                        self.new_point(size, tmin)
            else:
                results[size] = tmin
                if first:
                    self.new_curve(results, label)
                    first = False
                else:
                    self.new_point(size, tmin)
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.update_mp()

    def bench_2d(self, method="splitBBox", check=False, opencl=None):
        self.update_mp()
        if opencl:
            if (ocl is None):
                print("No pyopencl")
                return
            if (opencl.get("platformid") is None) or (opencl.get("deviceid") is None):
                platdev = ocl.select_device(opencl.get("devicetype"))
                if not platdev:
                    print("No such OpenCL device: skipping benchmark")
                    return
                platformid, deviceid = opencl["platformid"], opencl["deviceid"] = platdev
            devicetype = opencl["devicetype"] = ocl.platforms[platformid].devices[deviceid].type
            platform = str(ocl.platforms[platformid]).split()[0]
            if devicetype == "CPU":
                device = (str(ocl.platforms[platformid].devices[deviceid]).split("@")[0]).split()[-1]
            else:
                device = ' '.join(str(ocl.platforms[platformid].devices[deviceid]).split())

            print("Working on device: %s platform: %s device: %s" % (devicetype, platform, device))
            method += "_%i,%i" % (opencl["platformid"], opencl["deviceid"])
            label = ("2D %s %s %s %s" % (devicetype, self.LABELS[method], platform, device)).replace(" ", "_")
            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)

        else:
            print("Working on processor: %s" % self.get_cpu())
            label = "2D_" + self.LABELS[method]
            memory_error = (MemoryError, RuntimeError)

        results = OrderedDict()
        first = True
        for param in ds_list:
            self.update_mp()
            file_name = utilstest.UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest2D(poni, file_name, self.unit, method, self.out_2d)
            bench_test.setup()
            size = bench_test.data.size / 1.0e6
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(file_name), size, bench_test.N))
            try:
                t0 = time.time()
                _res = bench_test.stmt()
                self.print_init(time.time() - t0)
            except memory_error as error:
                print(error)
                break
            self.update_mp()
            if check:
                module = sys.modules.get(AzimuthalIntegrator.__module__)
                if module:
                    if "lut" in method:
                        key = module.EXT_LUT_ENGINE
                    elif "csr" in method:
                        key = module.EXT_CSR_ENGINE
                    else:
                        key = None
                if key and module:
                    try:
                        integrator = bench_test.ai.engines.get(key).engine
                    except MemoryError as error:
                        print(error)
                    else:
                        if "lut" in method:
                            print("lut: shape= %s \t nbytes %.3f MB " % (integrator.lut.shape, integrator.lut_nbytes / 2 ** 20))
                        else:
                            print("csr: size= %s \t nbytes %.3f MB " % (integrator.data.size, integrator.lut_nbytes / 2 ** 20))

            bench_test.ai.reset()
            bench_test.clean()
            try:
                t = timeit.Timer(bench_test.stmt, bench_test.setup_and_stmt)
                tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            except memory_error as error:
                print(error)
                break
            self.update_mp()
            del t
            self.update_mp()
            self.print_exec(tmin)
            tmin *= 1000.0
            results[size] = tmin
            if first:
                self.new_curve(results, label)
                first = False
            else:
                self.new_point(size, tmin)
            self.update_mp()
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.update_mp()

    def bench_gpu1d(self, devicetype="gpu", useFp64=True, platformid=None, deviceid=None):
        self.update_mp()
        print("Working on %s, in " % devicetype + ("64 bits mode" if useFp64 else"32 bits mode") + "(%s.%s)" % (platformid, deviceid))
        if ocl is None or not ocl.select_device(devicetype):
            print("No pyopencl or no such device: skipping benchmark")
            return
        results = OrderedDict()
        label = "Forward_OpenCL_%s_%s_bits" % (devicetype, ("64" if useFp64 else"32"))
        first = True
        for param in ds_list:
            self.update_mp()
            file_name = utilstest.UtilsTest.getimage(datasets[param])
            ai = load(param)
            data = fabio.open(file_name).data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (op.basename(file_name), size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

            try:
                t0 = time.time()
                res = ai.xrpd_OpenCL(data, N, devicetype=devicetype, useFp64=useFp64, platformid=platformid, deviceid=deviceid)
                t1 = time.time()
            except Exception as error:
                print("Failed to find an OpenCL GPU (useFp64:%s) %s" % (useFp64, error))
                continue
            self.print_init(t1 - t0)
            self.update_mp()
            ref = ai.xrpd(data, N)
            R = mathutil.rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            test = BenchTestGpu(param, file_name, devicetype, useFp64, platformid, deviceid)
            t = timeit.Timer(test.stmt, test.setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t
            self.update_mp()
            self.print_exec(tmin)
            print("")
            if R < self.LIMIT:
                size /= 1e6
                tmin *= 1000.0
                results[size] = tmin
                if first:
                    self.new_curve(results, label)
                    first = False
                else:
                    self.new_point(size, tmin)
                self.update_mp()
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.update_mp()

    def save(self, filename="benchmark.json"):
        self.update_mp()
        json.dump(self.results, open(filename, "w"), indent=4)

    def print_res(self):
        self.update_mp()
        print("Summary: execution time in milliseconds")
        print("Size/Meth\t" + "\t".join(self.meth))
        for i in self.size:
            print("%7.2f\t\t" % i + "\t\t".join("%.2f" % (self.results[j].get(i, 0)) for j in self.meth))

    def init_curve(self):
        self.update_mp()
        if self.fig:
            print("Already initialized")
            return
        if pylab and (sys.platform in ["win32", "darwin"]) or ("DISPLAY" in os.environ):
            self.fig = pylab.figure()
            self.fig.show()
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_autoscale_on(False)
            self.ax.set_xlabel("Image size in mega-pixels")
            self.ax.set_ylabel("Frame per second (log scale)")
            self.ax.set_yscale("log", basey=2)
            t = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            self.ax.set_yticks([float(i) for i in t])
            self.ax.set_yticklabels([str(i)for i in t])
            self.ax.set_xlim(0.5, 17)
            self.ax.set_ylim(0.5, 1500)
            self.ax.set_title(self.get_cpu() + " / " + self.get_gpu())
            update_fig(self.fig)

    def new_curve(self, results, label, style="-"):
        """
        Create a new curve within the current graph

        :param results: dict with execution time in function of size
        :param label: string with the title of the curve
        :param style: the style of the line: "-" for plain line, "--" for dashed
        """
        self.update_mp()
        if not self.fig:
            return
        self.plot_x = list(results.keys())
        self.plot_x.sort()
        self.plot_y = [1000.0 / results[i] for i in self.plot_x]
        self.plot = self.ax.plot(self.plot_x, self.plot_y, "o" + style, label=label)[0]
        self.ax.legend()
        update_fig(self.fig)

    def new_point(self, size, exec_time):
        """
        Add new point to current curve

        :param size: of the system
        :param exec_time: execution time in ms
        """
        self.update_mp()
        if not self.plot:
            return

        self.plot_x.append(size)
        self.plot_y.append(1000.0 / exec_time)
        self.plot.set_data(self.plot_x, self.plot_y)
        update_fig(self.fig)

    def display_all(self):
        if not self.fig:
            return
        for k in self.meth:
            self.new_curve(self.results[k], k)
        self.ax.legend()
        self.fig.savefig("benchmark.png")
        self.fig.show()
#        plt.ion()

    def update_mp(self):
        """
        Update memory profile curve
        """
        if not self.do_memprofile:
            return
        self.memory_profile[0].append(time.time() - self.starttime)
        self.memory_profile[1].append(self.get_mem())
        if pylab:
            if not self.fig_mp:
                self.fig_mp = pylab.figure()
                self.fig_mp.show()
                self.ax_mp = self.fig_mp.add_subplot(1, 1, 1)
                self.ax_mp.set_autoscale_on(False)
                self.ax_mp.set_xlabel("Run time (s)")
                self.ax_mp.set_xlim(0, 100)
                self.ax_mp.set_ylim(0, 2 ** 10)
                self.ax_mp.set_ylabel("Memory occupancy (MB)")
                self.ax_mp.set_title("Memory leak hunter")
                self.plot_mp = self.ax_mp.plot(*self.memory_profile)[0]
            else:
                self.plot_mp.set_data(*self.memory_profile)
                tmax = self.memory_profile[0][-1]
                mmax = max(self.memory_profile[1])
                if tmax > self.ax_mp.get_xlim()[-1]:
                    self.ax_mp.set_xlim(0, tmax)
                if mmax > self.ax_mp.get_ylim()[-1]:
                    self.ax_mp.set_ylim(0, mmax)
        if self.fig_mp.canvas:
            update_fig(self.fig_mp)

    def get_size(self):
        if len(self.meth) == 0:
            return []
        size = list(self.results[self.meth[0]].keys())
        for i in self.meth[1:]:
            s = list(self.results[i].keys())
            if len(s) > len(size):
                size = s
        size.sort()
        return size
    size = property(get_size)


def run_benchmark(number=10, repeat=1, memprof=False, max_size=1000,
                  do_1d=True, do_2d=False, devices="all"):
    """Run the integrated benchmark using the most common algorithms (method parameter)

    :param number: Measure timimg over number of executions
    :param repeat: number of measurement, takes the best of them
    :param memprof: set to True to enable memory profiling to hunt memory leaks
    :param max_size: maximum image size in megapixel, set it to 2 to speed-up the tests.
    :param do_1d: perfrom benchmarking using integrate1d
    :param do_2d: perfrom benchmarking using integrate2d
    :devices: "all", "cpu", "gpu" or "acc" or a list of devices [(proc_id, dev_id)]
    """
    print("Averaging over %i repetitions (best of %s)." % (number, repeat))
    bench = Bench(number, repeat, memprof, max_size=max_size)
    bench.init_curve()

    ocl_devices = []
    if ocl:
        if devices and isinstance(devices, (tuple, list)) and len(devices[0]) == 2:
            ocl_devices = devices
        else:
            ocl_devices = []
            for i in ocl.platforms:
                if devices == "all":
                    ocl_devices += [(i.id, j.id) for j in i.devices]
                else:
                    if "cpu" in devices:
                        ocl_devices += [(i.id, j.id) for j in i.devices if j.type == "CPU"]
                    if "gpu" in devices:
                        ocl_devices += [(i.id, j.id) for j in i.devices if j.type == "GPU"]
                    if "acc" in devices:
                        ocl_devices += [(i.id, j.id) for j in i.devices if j.type == "ACC"]
        print("Devices:", ocl_devices)
    if do_1d:
        bench.bench_1d("splitBBox")
        bench.bench_1d("lut", True)
        bench.bench_1d("csr", True)
        for device in ocl_devices:
            print("Working on device: " + str(device))
            bench.bench_1d("lut_ocl", True, {"platformid": device[0], "deviceid": device[1]})
            bench.bench_1d("csr_ocl", True, {"platformid": device[0], "deviceid": device[1]})

    if do_2d:
        bench.bench_2d("splitBBox")
        bench.bench_2d("lut", True)
        for device in ocl_devices:
            bench.bench_1d("lut_ocl", True, {"platformid": device[0], "deviceid": device[1]})
            bench.bench_1d("csr_ocl", True, {"platformid": device[0], "deviceid": device[1]})

    bench.save()
    bench.print_res()
    bench.update_mp()

    return bench.results


run = run_benchmark
