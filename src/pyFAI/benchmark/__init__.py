#!/usr/bin/env python
# coding: utf-8
#
#    Copyright (C) 2016-2024 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__date__ = "13/05/2025"
__license__ = "MIT"
__copyright__ = "2012-2024 European Synchrotron Radiation Facility, Grenoble, France"

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
from math import ceil
import numpy

from .. import load, detector_factory
from ..integrator.azimuthal import AzimuthalIntegrator
from ..method_registry import IntegrationMethod, Method
from ..utils import mathutil
from ..test.utilstest import UtilsTest
from ..opencl import pyopencl, ocl
try:
    from ..gui.matplotlib import pyplot, pylab
    from ..gui.utils import update_fig as _update_fig

    def update_fig(*args, **kwargs):
        pyplot.pause(0.1)
        _update_fig(*args, **kwargs)

except ImportError:
    pylab = pyplot = None

    def update_fig(*args, **kwargs):
        pass

detector_names = ["Pilatus1M",
                  "Pilatus2M",
                  "Eiger4M",
                  "Pilatus6M",
                  "Eiger9M",
                  "Mar3450",
                  "Fairchild", ]

ds_list = [d + ".poni" for d in detector_names]

data_sizes = [numpy.prod(detector_factory(d).shape) * 1e-6
              for d in detector_names]

datasets = {"Fairchild.poni": "Fairchild.edf",
            # "halfccd.poni": "halfccd.edf",
            # "Frelon2k.poni": "Frelon2k.edf",
            "Pilatus6M.poni": "Pilatus6M.cbf",
            "Pilatus1M.poni": "Pilatus1M.edf",
            "Mar3450.poni": "LaB6_260210.mar3450",
            "Pilatus2M.poni":"Pilatus2M.cbf",
            "Eiger4M.poni":"Eiger4M.edf",
            "Eiger9M.poni":"Eiger9M.h5"
            }

PONIS = { i: UtilsTest.getimage(i) for i in ds_list}

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

    def get_device(self):
        res = None
        if "ai" in dir(self):
            if "engines" in dir(self.ai):
                for method in self.ai.engines:
                    if isinstance(method, Method) and method.impl == "opencl":
                        res = self.ai.engines[method].engine.ctx.devices[0]
                        break
                else:
                    if ("ocl_csr_integr" in self.ai.engines):
                        res = self.ai.engines["ocl_csr_integr"].engine.ctx.devices[0]
        return res


class BenchTest1D(BenchTest):
    """Test 1d integration"""

    def __init__(self, poni, file_name, unit, method, function=None,
                 error_model=None):
        BenchTest.__init__(self)
        self.poni = poni
        self.file_name = file_name
        self.unit = unit
        self.method = method
        self.compute_engine = None
        self.function_name = function or "integrate1d"
        self.error_model = error_model
        self.function = None

    def setup(self):
        self.ai = AzimuthalIntegrator.sload(self.poni)
        with fabio.open(self.file_name) as fimg:
            self.data = fimg.data
        self.N = min(self.data.shape)
        self.function = self.ai.__getattribute__(self.function_name)

    def stmt(self):
        return self.function(self.data, self.N, safe=False, unit=self.unit,
                             method=self.method, error_model=self.error_model)

    def clean(self):
        self.ai = None
        self.data = None


class BenchTest2D(BenchTest):
    """Test 2d integration"""

    def __init__(self, poni, file_name, unit, method, output_size):
        BenchTest.__init__(self)
        self.poni = poni
        self.file_name = file_name
        self.unit = unit
        self.method = method
        self.output_size = output_size

    def setup(self):
        self.ai = AzimuthalIntegrator.sload(self.poni)
        with fabio.open(self.file_name) as fimg:
            self.data = fimg.data
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
        with fabio.open(self.file_name) as fimg:
            self.data = fimg.data
        self.N = min(self.data.shape)
        self.ai.xrpd_OpenCL(self.data, self.N, devicetype=self.devicetype, useFp64=self.useFp64, platformid=self.platformid, deviceid=self.deviceid)

    def stmt(self):
        return self.ai.xrpd_OpenCL(self.data, self.N, safe=False)

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
    LABELS = {("bbox", "histogram", "cython"): "CPU_serial",
              ("bbox", "lut", "cython"): "CPU_LUT_OpenMP",
              ("bbox", "lut", "opencl"): "LUT",
              ("bbox", "csr", "cython"): "CPU_CSR_OpenMP",
              ("bbox", "csr", "opencl"): "CSR",
              ("bbox", "csc", "cython"): "CPU_CSC_Serial",
              }

    def __init__(self, nbr=10, repeat=1, memprofile=False, unit="2th_deg", max_size=None):
        self.reference_1d = {}
        self.LIMIT = 8
        self.repeat = repeat
        self.nbr = nbr
        self._cpu = None
        self.results = {"host": platform.node(),
                        "argv": sys.argv,
                        "cpu": self.get_cpu(),
                        "gpu": self.get_gpu()}
        self.meth = []
        self.fig = None
        self.ax = None
        self.starttime = time.perf_counter()
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
        self.plot_y_range = [1, 1000]

    def get_cpu(self):
        if self._cpu is None:
            if os.name == "nt":
                self._cpu = platform.processor()
            elif os.path.exists("/proc/cpuinfo"):
                cpuinfo = [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("model name")]
                if not cpuinfo:
                    cpuinfo = [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("cpu")]
                if not cpuinfo:
                    self._cpu = "cpu"
                else:
                    self._cpu = cpuinfo[0].strip()
            elif os.path.exists("/usr/sbin/sysctl"):
                proc = subprocess.Popen(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE)
                proc.wait()
                self._cpu = proc.stdout.read().strip().decode("ASCII")
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
        status_file = f"/proc/{pid}/status"
        if os.path.exists(status_file):
            for l in open(status_file):
                if l.startswith("VmRSS"):
                    mem = int(l.split(":", 1)[1].split()[0]) / 1024.
        else:
            mem = 0
        return mem

    def print_init(self, t):
        print(" * Initialization time: %.1f ms" % (1000.0 * t))
        self.update_mp()

    def print_init2(self, tinit, trep, loops):
        print(" * Initialization time: %.1f ms, Repetition time: %.1f ms, executing %i loops" %
              (1000.0 * tinit, 1000.0 * trep, loops))
        self.update_mp()

    def print_exec(self, t):
        print(" * Execution time rep : %.1f ms" % (1000.0 * t))
        self.update_mp()

    def print_sep(self):
        print("*" * 80)
        self.update_mp()

    def get_ref(self, param):
        if param not in self.reference_1d:
            file_name = UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest1D(poni, file_name, self.unit, ("bbox", "histogram", "cython"), function="integrate1d_ng")
            bench_test.setup()
            res = bench_test.stmt()
            bench_test.compute_engine = res.compute_engine
            self.reference_1d[param] = res
            bench_test.clean()
        return self.reference_1d[param]

    def bench_1d(self, method="splitBBox", check=False, opencl=None, function="integrate1d"):
        """
        :param method: method to be bechmarked
        :param check: check results vs ref if method is LUT based
        :param opencl: dict containing platformid, deviceid and devicetype
        """
        method = IntegrationMethod.select_one_available(method, dim=1, default=None, degradable=True)
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
            # label = ("%s %s %s %s %s" % (function, devicetype, self.LABELS[method.method[1:4]], platform, device)).replace(" ", "_")
            label = f'{devicetype}:{platform}:{device} / {function}: ({method.split_lower},{method.algo_lower},{method.impl_lower})'
            method = IntegrationMethod.select_method(dim=1, split=method.split_lower,
                                                      algo=method.algo_lower, impl=method.impl_lower,
                                                      target=(opencl["platformid"], opencl["deviceid"]))[0]
            print(f"function: {function} \t method: {method}")
            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)
        else:
            print("Working on processor: %s" % self.get_cpu())
            # label = function + " " + self.LABELS[method.method[1:4]]
            label = f'CPU / {function}: {method.split_lower}_{method.algo_lower}_{method.impl_lower}'
            memory_error = (MemoryError, RuntimeError)
        results = OrderedDict()
        first = True
        for param in ds_list:
            self.update_mp()
            file_name = UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest1D(poni, file_name, self.unit, method, function=function)
            bench_test.setup()
            size = bench_test.data.size / 1.0e6
            if size > self.max_size:
                continue
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(file_name), size, bench_test.N))
            try:
                t0 = time.perf_counter()
                res = bench_test.stmt()
                t1 = time.perf_counter()
                res2 = bench_test.stmt()
                t2 = time.perf_counter()
                loops = int(ceil(self.nbr / (t2 - t1)))
                self.print_init2(t1 - t0, t2 - t1, loops)

            except memory_error as error:
                print("MemoryError: %s" % error)
                break
            if first:
                actual_device = bench_test.get_device()
                if actual_device:
                    print("Actual device used: %s" % actual_device)

            self.update_mp()
            if method.algo_lower in ("lut", "csr"):
                key = Method(1, bench_test.method.split_lower, method.algo_lower, "cython", None)
                if key and key in bench_test.ai.engines:
                    engine = bench_test.ai.engines.get(key)
                    if engine:
                        integrator = engine.engine
                        if method.algo_lower == "lut":
                            print("lut: shape= %s \t nbytes %.3f MB " % (integrator.lut.shape, integrator.lut_nbytes / 2 ** 20))
                        else:
                            print("csr: size= %s \t nbytes %.3f MB " % (integrator.data.size, integrator.lut_nbytes / 2 ** 20))
            bench_test.clean()
            self.update_mp()
            try:
                t = timeit.Timer(bench_test.stmt, bench_test.setup_and_stmt)
                tmin = min([i / loops for i in t.repeat(repeat=self.repeat, number=loops)])
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
                            self.new_curve(results, label, style="--", marker="s" if "legacy" in function else "o")
                        else:
                            self.new_curve(results, label, style="-", marker="s" if "legacy" in function else "o")
                        first = False
                    else:
                        self.new_point(size, tmin)
            else:
                results[size] = tmin
                if first:
                    self.new_curve(results, label, marker="s" if "legacy" in function else "o")
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
            else:
                platformid, deviceid = opencl["platformid"], opencl["deviceid"]
            devicetype = ocl.platforms[platformid].devices[deviceid].type
            platform = str(ocl.platforms[platformid]).split()[0]
            if devicetype == "CPU":
                device = (str(ocl.platforms[platformid].devices[deviceid]).split("@")[0]).split()[-1]
            else:
                device = ' '.join(str(ocl.platforms[platformid].devices[deviceid]).split())

            print(f"Working on device: {devicetype} platform: {platform} device: {device} method {method}")
            # label = ("2D %s %s %s %s" % (devicetype, self.LABELS[method[1:4]], platform, device)).replace(" ", "_")
            label = f'{devicetype}:{platform}:{device} / 2D: {method.split_lower}_{method.algo_lower}_{method.impl_lower}'
            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)

        else:
            print("Working on processor: %s" % self.get_cpu())
            # label = "2D_" + self.LABELS[method[1:4]]
            label = f'CPU / 2D: {method.split_lower}_{method.algo_lower}_{method.impl_lower}'
            memory_error = (MemoryError, RuntimeError)

        results = OrderedDict()
        first = True
        for param in ds_list:
            self.update_mp()
            file_name = UtilsTest.getimage(datasets[param])
            poni = PONIS[param]
            bench_test = BenchTest2D(poni, file_name, self.unit, method, self.out_2d)
            bench_test.setup()
            size = bench_test.data.size / 1.0e6
            if size > self.max_size:
                continue
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(file_name), size, bench_test.N))
            try:
                t0 = time.perf_counter()
                _res = bench_test.stmt()
                t2 = time.perf_counter()
                self.print_init(t2 - t0)
                loops = int(ceil(self.nbr / (t2 - t0)))
            except memory_error as error:
                print(error)
                break
            self.update_mp()
            if check:
                module = sys.modules.get(AzimuthalIntegrator.__module__)
                if module:
                    if "lut" in method.algo_lower:  # Edgar
                    # if "lut" in method:
                        key = module.EXT_LUT_ENGINE
                    elif "csr" in method.algo_lower:  # Edgar
                    # elif "csr" in method:
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
                tmin = min([i / loops for i in t.repeat(repeat=self.repeat, number=loops)])
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
                self.new_curve(results, label, marker="o")
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
            file_name = UtilsTest.getimage(datasets[param])
            ai = load(param)
            with fabio.open(file_name) as fimg:
                data = fimg.data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (op.basename(file_name), size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

            try:
                t0 = time.perf_counter()
                res = ai.xrpd_OpenCL(data, N, devicetype=devicetype, useFp64=useFp64, platformid=platformid, deviceid=deviceid)
                t1 = time.perf_counter()
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
                    self.new_curve(results, label, marker="o")
                    first = False
                else:
                    self.new_point(size, tmin)
                self.update_mp()
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.update_mp()

    def save(self, filename=None):
        if filename is None:
            filename = f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
        self.update_mp()
        json.dump(self.results, open(filename, "w"), indent=4)
        if self.fig is not None:
            self.fig.savefig(filename[:-4] + "svg")

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
        if pyplot and ((sys.platform in ["win32", "darwin"]) or ("DISPLAY" in os.environ)):
            self.fig, self.ax = pyplot.subplots(figsize=(12, 6))
            self.fig.show()
            self.ax.set_autoscale_on(False)
            self.ax.set_xlabel("Image size in mega-pixels")
            self.ax.set_ylabel("Frame per second (log scale)")

            try:
                self.ax.set_yscale("log", base=2)
            except Exception:
                self.ax.set_yscale("log", basey=2)
            t = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            self.ax.set_yticks([float(i) for i in t])
            self.ax.set_yticklabels([str(i)for i in t])
            self.ax.set_xlim(0.0, 20)
            self.ax.set_ylim(0.75 * self.plot_y_range[0],
                             1.5 * self.plot_y_range[1])
            self.ax.set_title(f'CPU: {self.get_cpu()}\nGPU: {self.get_gpu()}')

            # Display detector markers (vertical lines)
            self.ax.vlines(
                x=data_sizes,
                ymin=0,
                ymax=100*self.plot_y_range[1],
                linestyles='dashed',
                alpha=0.4,
                colors='black',
            )
            for size, detector_label in zip(data_sizes, detector_names):
                self.ax.text(x=size, y=0.8, s=detector_label, rotation=270, fontsize=7)

            update_fig(self.fig)

    def new_curve(self, results, label, style="-", marker="x"):
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
        self.plot_y_range = [min(min(self.plot_y_range), min(self.plot_y)),
                             max(max(self.plot_y_range), max(self.plot_y))]
        self.plot = self.ax.plot(self.plot_x, self.plot_y, marker + style, label=label)[0]
        self.ax.set_ylim(0.75 * self.plot_y_range[0],
                         1.5 * self.plot_y_range[1])

        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10,
        )
        self.fig.subplots_adjust(right=0.5)
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

        y_value = 1000.0 / exec_time
        self.plot_x.append(size)
        self.plot_y.append(y_value)
        self.plot.set_data(self.plot_x, self.plot_y)
        self.plot_y_range = [min(self.plot_y_range[0], y_value),
                             max(self.plot_y_range[1], y_value)]
        self.ax.set_ylim(0.75 * self.plot_y_range[0],
                         1.5 * self.plot_y_range[1])
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

    def display_detector_markers(self):
        if not self.fig:
            return

    def update_mp(self):
        """
        Update memory profile curve
        """
        if not self.do_memprofile:
            return
        self.memory_profile[0].append(time.perf_counter() - self.starttime)
        self.memory_profile[1].append(self.get_mem())
        if pyplot:
            if self.fig_mp is None:
                self.fig_mp, self.ax_mp = pyplot.subplots()
                self.ax_mp.set_autoscale_on(False)
                self.ax_mp.set_xlabel("Run time (s)")
                self.ax_mp.set_xlim(0, 100)
                self.ax_mp.set_ylim(0, 2 ** 10)
                self.ax_mp.set_ylabel("Memory occupancy (MB)")
                self.ax_mp.set_title("Memory leak hunter")
                self.plot_mp = self.ax_mp.plot(*self.memory_profile)[0]
                self.fig_mp.show()
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
                  do_1d=True, do_2d=False, processor=True, devices="all",
                  split_list=["bbox"], algo_list=["histogram", "CSR"], impl_list=["cython", "opencl"], function="all",
                  all=False,):
    """Run the integrated benchmark using the most common algorithms (method parameter)

    :param number: Measure timimg over number of executions or average over this time
    :param repeat: number of measurement, takes the best of them
    :param memprof: set to True to enable memory profiling to hunt memory leaks
    :param max_size: maximum image size in megapixel, set it to 2 to speed-up the tests.
    :param do_1d: perfrom benchmarking using integrate1d
    :param do_2d: perfrom benchmarking using integrate2d
    :devices: "all", "cpu", "gpu" or "acc" or a list of devices [(proc_id, dev_id)]
    """
    print(f"Benchmarking over {number} seconds (best of {repeat} repetitions).")
    bench = Bench(number, repeat, memprof, max_size=max_size)
    bench.init_curve()

    ocl_devices = []
    if ocl:
        try:
            ocl_devices = [(int(i), int(j)) for i, j in devices]
        except Exception as err:
            # print(f"{type(err)}: {err}\ndevices is not a list of 2-tuple of integrers, parsing the list")
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
        print(f"Devices: {devices} --> {ocl_devices}")

    # Gather available methods
    if all:
        methods_available = IntegrationMethod.select_method()
    else:
        methods_available = []

        # Clear list if "all" to avoid redundances
        if "all" in split_list:
            split_list = [None]
        if "all" in algo_list:
            algo_list = [None]
        if "all" in impl_list:
            impl_list = [None]

        for split in split_list:
            for algo in algo_list:
                for impl in impl_list:
                    if do_1d:
                        for method in IntegrationMethod.select_method(dim=1, split=split, algo=algo, impl=impl):
                            if method not in methods_available:
                                methods_available.append(method)
                    if do_2d:
                        for method in IntegrationMethod.select_method(dim=2, split=split, algo=algo, impl=impl):
                            if method not in methods_available:
                                methods_available.append(method)
    # Separate non-opencl from opencl methods
    methods_non_ocl, methods_ocl = [], []
    for method in methods_available:
        # print(method, method.impl_lower, method.target, ocl_devices)
        if method.impl_lower == 'opencl':
            if method.target in ocl_devices:
                methods_ocl.append(method)
        else:
            if processor:
                methods_non_ocl.append(method)

    # Benchmark 1d integration
    if do_1d:
        if function == "all":
            function_list = ["integrate1d_legacy", "integrate1d_ng"]
        elif function == "ng":
            function_list = ["integrate1d_ng"]
        elif function == "legacy":
            function_list = ["integrate1d_legacy"]

        # Benchmark No OpenCL devices
        for method in methods_non_ocl:
            if method.dimension == 1:
                for function in function_list:
                    bench.bench_1d(
                        method=method,
                        check=True,
                        opencl=None,
                        function=function,
                    )

        # Benchmark OpenCL devices
        for method in methods_ocl:
            if method.dimension == 1:
                print("Working on device: " + str(method.target_name))
                for function in function_list:
                    bench.bench_1d(
                        method=method,
                        check=True,
                        opencl={"platformid": method.target[0], "deviceid": method.target[1]},
                        function=function,
                    )

    # Benchmark 2d integration
    if do_2d:

        # Benchmark No OpenCL devices
        for method in methods_non_ocl:
            if method.dimension == 2:
                bench.bench_2d(
                    method=method,
                    check=False,
                    opencl=False,
                )

        # Benchmark OpenCL devices
        for method in methods_ocl:
            if method.dimension == 2:
                bench.bench_2d(
                    method=method,
                    check=False,
                    opencl={"platformid": method.target[0], "deviceid": method.target[1]},
                )

    bench.save()
    bench.print_res()
    bench.update_mp()

    return bench.results


run = run_benchmark
