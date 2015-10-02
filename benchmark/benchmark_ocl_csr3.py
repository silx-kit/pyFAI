#!/usr/bin/python
# coding: utf-8
# author: Jérôme Kieffer
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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
"""
Benchmark for Azimuthal integration of PyFAI
"""
from __future__ import print_function, division

import json, sys, time, timeit, os, platform, subprocess
import numpy
from numpy import log2
import fabio
import os.path as op
sys.path.append(op.join(op.dirname(op.dirname(op.abspath(__file__))), "test"))
import utilstest

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    print("No socket opened for debugging -> please install rfoo")

# We use the locally build version of PyFAI
pyFAI = utilstest.UtilsTest.pyFAI
ocl = pyFAI.opencl.ocl
from matplotlib import pyplot as plt
plt.ion()

ds_list = ["Pilatus1M.poni", "halfccd.poni", "Frelon2k.poni", "Pilatus6M.poni", "Mar3450.poni", "Fairchild.poni"]
datasets = {"Fairchild.poni":utilstest.UtilsTest.getimage("1880/Fairchild.edf"),
            "halfccd.poni":utilstest.UtilsTest.getimage("1882/halfccd.edf"),
            "Frelon2k.poni":utilstest.UtilsTest.getimage("1881/Frelon2k.edf"),
            "Pilatus6M.poni":utilstest.UtilsTest.getimage("1884/Pilatus6M.cbf"),
            "Pilatus1M.poni":utilstest.UtilsTest.getimage("1883/Pilatus1M.edf"),
            "Mar3450.poni":utilstest.UtilsTest.getimage("2201/LaB6_260210.mar3450")
            }



# Handle to the Bench instance: allows debugging from outside if needed
bench = None


class Bench(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    LABELS = {"splitBBox": "CPU_serial",
          "lut": "CPU_LUT_OpenMP",
          "lut_ocl": "%s_LUT_OpenCL",
          "csr": "CPU_CSR_OpenMP",
          "csr_ocl": "%s_CSR_OpenCL",
          }

    def __init__(self, nbr=10, repeat=3, memprofile=False, unit="2th_deg"):
        self.reference_1d = {}
        self.LIMIT = 8
        self.repeat = repeat
        self.nbr = nbr
        self.results = {}
        self.flops = {}
        self.mem_band = {}
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
        self.setup = """import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
"""
        self.unsetup = "ai=None"
        self.setup_1d = self.setup + "N=min(data.shape)" + os.linesep
        self.setup_2d = self.setup + "N=(%i,%i)%s" % (self.out_2d[0], self.out_2d[0], os.linesep)
        self.stmt_1d = "ai.integrate1d(data, N, safe=False, unit='" + self.unit + "', method='%s', block_size=%i, profile=True)"
        self.stmt_2d = ("ai.integrate2d(data, %i, %i, unit='" % self.out_2d) + self.unit + "', method='%s')"



    def get_cpu(self):
        if self._cpu is None:
            if os.name == "nt":
                self._cpu = platform.processor()
            elif os.path.exists("/proc/cpuinfo"):
                self._cpu = [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("model name")][0].strip()
            elif os.path.exists("/usr/sbin/sysctl"):
                proc = subprocess.Popen(["sysctl", "-n", "machdep.cpu.brand_string"], stdout=subprocess.PIPE)
                proc.wait()
                self._cpu = proc.stdout.read().strip()
            old = self._cpu
            self._cpu = old.replace("  ", " ")
            while old != self._cpu:
                old = self._cpu
                self._cpu = old.replace("  ", " ")
        return self._cpu

    def get_gpu(self, devicetype="gpu", useFp64=False, platformid=None, deviceid=None):
        if ocl is None:
            return "NoGPU"
        ctx = ocl.create_context(devicetype, useFp64, platformid, deviceid)
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
        print(" * Execution time of integration : %.1f ms" % (t))
        self.update_mp()


    def print_sep(self):
        print("*"*80)
        self.update_mp()

    def get_ref(self, param):
        if param not in self.reference_1d:
            fn = datasets[param]
            setup = self.setup_1d % (param, fn)
            exec setup
            res = eval(self.stmt_1d % ("splitBBox", 32))
            self.reference_1d[param] = res
            del ai, data
        return self.reference_1d[param]

    def bench_1d_ocl_csr(self, check=False, opencl=None):
        """
        @param method: method to be bechmarked
        @param check: check results vs ref if method is LUT based
        @param opencl: dict containing platformid, deviceid and devicetype
        """
        method = "ocl_csr"
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
            print("Working on device: %s platform: %s device: %s" % (devicetype, ocl.platforms[platformid], ocl.platforms[platformid].devices[deviceid]))
            label = "1D_" + method + "_" + devicetype
            method += "_%i,%i" % (opencl["platformid"], opencl["deviceid"])
        else:
            print("Working on processor: %s" % self.get_cpu())
            label = "1D_" + self.LABELS[method]
        results = {}
        flops = {}
        mem_band = {}
        first = True
        param = "Pilatus1M.poni"
        block_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for block_size in block_size_list:
            self.update_mp()
            fn = datasets[param]
            setup = self.setup_1d % (param, fn)
            stmt = self.stmt_1d % (method, block_size)
            exec setup
            size = data.size / 1.0e6
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), size, N))
            t0 = time.time()
            res = eval(stmt)
            self.print_init(time.time() - t0)
            self.update_mp()
            if check:
                if "csr" in method:
                    print("csr: size= %s \t nbytes %.3f MB " % (ai._csr_integrator.data.size, ai._csr_integrator.lut_nbytes / 2 ** 20))

            bins = ai._csr_integrator.bins
            nnz = ai._csr_integrator.nnz
            parallel_reduction = sum([2 ** i for i in range(1, int(log2(block_size)))])

            FLOPs = 9 * nnz + 11 * parallel_reduction + 1 * bins
            mem_access = (2 * block_size * bins + 5 * nnz + 7 * bins) * 4

            del ai, data
            self.update_mp()

            t_repeat = []
            for j in range(self.repeat):
                t = []
                exec setup
                for i in range(self.nbr):
                    eval(stmt)
                for e in ai._ocl_csr_integr.events:
                    if "integrate" in e[0]:
                        et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                        t.append(et)
                exec(self.unsetup)
                t_repeat.append(numpy.mean(t))

            tmin = min(t_repeat)
            self.update_mp()
            self.print_exec(tmin)
            if check:
                ref = self.get_ref(param)
                R = utilstest.Rwp(res, ref)
                print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
                self.update_mp()
                if R < self.LIMIT:
                    results[block_size ] = tmin
                    flops[block_size ] = (FLOPs / tmin) * 1e-6
                    mem_band[block_size ] = (mem_access / tmin) * 1e-6
                    self.update_mp()
            else:
                results[block_size ] = tmin
                flops[block_size ] = FLOPs / tmin
                mem_band[block_size ] = mem_access / tmin
            if first:
                self.new_curve(results, label)
                first = False
            else:
                self.new_point(block_size, tmin)
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.flops[label] = flops
        self.mem_band[label] = mem_band
        self.update_mp()


    def save(self, filename="benchmark.json"):
        self.update_mp()
        json.dump(self.results, open(filename, "w"))

    def print_res(self, summary, results):
        self.update_mp()
        print(summary)
        print("Size/Meth\t" + "\t".join(self.meth))
        for i in self.size:
            print("%7.2f\t\t" % i + "\t\t".join("%.2f" % (results[j].get(i, 0)) for j in self.meth))

    def init_curve(self):
        self.update_mp()
        if self.fig:
            print("Already initialized")
            return
        if "DISPLAY" in os.environ:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_autoscale_on(False)
            self.ax.set_xlabel("Workgroup size")
            self.ax.set_ylabel("Time for integration in ms")
            self.ax.set_xscale("log", basey=2)
            t = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            self.ax.set_xticks([float(i) for i in t])
            self.ax.set_xticklabels([str(i)for i in t])
            self.ax.set_ylim(0, 70)
            self.ax.set_xlim(1, 300)
            self.ax.set_title(self.get_cpu() + " / " + self.get_gpu())

            if self.fig.canvas:
                self.fig.canvas.draw()
#            plt.show()

    def new_curve(self, results, label):
        self.update_mp()
        if not self.fig:
            return
        self.plot_x = list(results.keys())
        self.plot_x.sort()
        self.plot_y = [results[i] for i in self.plot_x]
        self.plot = self.ax.plot(self.plot_x, self.plot_y, "o-", label=label)[0]
        self.ax.legend()
        if self.fig.canvas:
            self.fig.canvas.draw()

    def new_point(self, size, exec_time):
        """
        Add new point to current curve
        @param size: of the system
        @parm exec_time: execution time in ms
        """
        self.update_mp()
        if not self.plot:
            return

        self.plot_x.append(size)
        self.plot_y.append(exec_time)
        self.plot.set_data(self.plot_x, self.plot_y)
        if self.fig.canvas:
            self.fig.canvas.draw()

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
        if not self.do_memprofile:
            return
        self.memory_profile[0].append(time.time() - self.starttime)
        self.memory_profile[1].append(self.get_mem())
        if not self.fig_mp:
            self.fig_mp = plt.figure()
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
            self.fig_mp.canvas.draw()

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


if __name__ == "__main__":
    try:
        from argparse import ArgumentParser
    except:
        from pyFAI.argparse import ArgumentParser
    description = """Benchmark for Azimuthal integration
    """
    epilog = """  """
    usage = """benchmark [options] """
    version = "pyFAI benchmark version " + pyFAI.version
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-v", "--version", action='version', version=version)
    parser.add_argument("-d", "--debug",
                          action="store_true", dest="debug", default=False,
                          help="switch to verbose/debug mode")
    parser.add_argument("-s", "--small",
                      action="store_true", dest="small", default=False,
                      help="Limit the size of the dataset to 6 Mpixel images (for computer with limited memory)")
    parser.add_argument("-c", "--cpu",
                      action="store_true", dest="opencl_cpu", default=False,
                      help="perform benchmark using OpenCL on the CPU")
    parser.add_argument("-g", "--gpu",
                      action="store_true", dest="opencl_gpu", default=False,
                      help="perform benchmark using OpenCL on the GPU")
    parser.add_argument("-a", "--acc",
                      action="store_true", dest="opencl_acc", default=False,
                      help="perform benchmark using OpenCL on the Accelerator (like XeonPhi/MIC)")
    parser.add_argument("-n", "--number",
                      dest="number", default=10, type=int,
                      help="Number of repetition of the test, by default 10")
    parser.add_argument("-r", "--repeat",
                      dest="repeat", default=1, type=int,
                      help="Repeat each benchmark x times to take the best")

    options = parser.parse_args()
    if options.small:
        ds_list = ds_list[:4]
    if options.debug:
            pyFAI.logger.setLevel(logging.DEBUG)
    print("Averaging over %i repetitions (best of %s)." % (options.number, options.repeat))
    bench = Bench(options.number, options.repeat)
    bench.init_curve()
    if options.opencl_cpu:
            bench.bench_1d_ocl_csr(True, {"devicetype":"CPU"})
    if options.opencl_gpu:
            bench.bench_1d_ocl_csr(True, {"devicetype":"GPU"})
    if options.opencl_acc:
            bench.bench_1d_ocl_csr(True, {"devicetype":"ACC"})

    bench.save()
    results = bench.results
    flops = bench.flops
    mem_band = bench.mem_band

    bench.print_res("Summary: Execution time in milliseconds", results)
    bench.print_res("Summary: MFLOPS", flops)
    bench.print_res("Summary: Memory Bandwidth in MB/s", mem_band)
    bench.update_mp()

    bench.ax.set_ylim(1, 200)
    # plt.show()
    plt.ion()
#    raw_input("Enter to quit")
