#!/usr/bin/python
import fabio, sys, time, timeit, os, platform, subprocess, gc
import os.path as op

sys.path.append(op.join(op.dirname(op.dirname(op.abspath(__file__))), "test"))
import utilstest
pyFAI = utilstest.UtilsTest.pyFAI
ocl = pyFAI.opencl.ocl
from matplotlib import pyplot as plt

ds_list = ["Pilatus1M.poni", "halfccd.poni", "Frelon2k.poni", "Pilatus6M.poni", "Mar3450.poni", "Fairchild.poni"]
datasets = {"Fairchild.poni":utilstest.UtilsTest.getimage("1880/Fairchild.edf"),
            "halfccd.poni":utilstest.UtilsTest.getimage("1882/halfccd.edf"),
            "Frelon2k.poni":utilstest.UtilsTest.getimage("1881/Frelon2k.edf"),
            "Pilatus6M.poni":utilstest.UtilsTest.getimage("1884/Pilatus6M.cbf"),
            "Pilatus1M.poni":utilstest.UtilsTest.getimage("1883/Pilatus1M.edf"),
            "Mar3450.poni":utilstest.UtilsTest.getimage("2201/LaB6_260210.mar3450")
      }
b = None
class Bench(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    def __init__(self, nbr=10):
        self.reference_1d = {}
        self.LIMIT = 8
        self.repeat = 1
        self.nbr = nbr
        self.results = {}
        self.meth = []
        self._cpu = None
        self.fig = None
        self.ax = None

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

    def print_init(self, t):
        print(" * Initialization time: %.1f ms" % (1000.0 * t))


    def print_exec(self, t):
        print(" * Execution time rep : %.1f ms" % (1000.0 * t))


    def print_sep(self):
        print("*"*80)

    def get_ref(self, param):
        if param not in self.reference_1d:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            res = ai.xrpd(data, N)
            self.reference_1d[param] = res
            del ai, data
        return self.reference_1d[param]

    def bench_cpu1d(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            ref = self.get_ref(param)
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            res = ai.xrpd(data, N)
            t1 = time.time()
            self.print_init(t1 - t0)
            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
N=min(data.shape)
out=ai.xrpd(data,N)""" % (param, fn)
            t = timeit.Timer("ai.xrpd(data,N)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
#            R = utilstest.Rwp(res, ref)
#            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
#            if R < self.LIMIT:
            results[size / 1e6] = tmin * 1000.0
        gc.collect()
        self.print_sep()
        label = "1D_CPU_serial"
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)

    def bench_cpu1d_lut(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            ref = self.get_ref(param)
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            res = ai.xrpd_LUT(data, N)
            t1 = time.time()
            self.print_init(t1 - t0)
            print "lut.shape=", ai._lut_integrator.lut.shape, "lut.nbytes (MB)", ai._lut_integrator.size * 8 / 1e6
            del ai, data
            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
N=min(data.shape)
out=ai.xrpd_LUT(data,N)""" % (param, fn)
            t = timeit.Timer("ai.xrpd_LUT(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            if R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
        self.print_sep()
        label = "1D_CPU_parallel_OpenMP"
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)
        gc.collect()

    def bench_cpu1d_lut_ocl(self, devicetype="ALL", platformid=None, deviceid=None):
        if (ocl is None):
            print("No pyopencl")
            return
        if (platformid is None) or (deviceid is None):
            platdev = ocl.select_device(devicetype)
            if not platdev:
                print("No such OpenCL device: skipping benchmark")
                return
            platformid, deviceid = platdev
        print("Working on device: %s platform: %s device: %s" % (devicetype, ocl.platforms[platformid], ocl.platforms[platformid].devices[deviceid]))

        results = {}
        for param in ds_list:
            ref = self.get_ref(param)
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            try:
                res = ai.xrpd_LUT_OCL(data, N, devicetype=devicetype, platformid=platformid, deviceid=deviceid)
            except MemoryError as error:
                print(error)
                break
            t1 = time.time()
            self.print_init(t1 - t0)
            del ai, data
            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
N=min(data.shape)
out=ai.xrpd_LUT_OCL(data,N,devicetype=r"%s",platformid=%s,deviceid=%s)""" % (param, fn, devicetype, platformid, deviceid)
            t = timeit.Timer("ai.xrpd_LUT_OCL(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t
            self.print_exec(tmin)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            if R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
            gc.collect()
        label = "1D_%s_parallel_OpenCL" % devicetype
        self.print_sep()
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)


    def bench_cpu2d(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = (500, 360)
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            _ = ai.xrpd2(data, N[0], N[1])
            t1 = time.time()
            self.print_init(t1 - t0)
            del ai, data
            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
out=ai.xrpd2(data,%s,%s)""" % (param, fn, N[0], N[1])
            t = timeit.Timer("ai.xrpd2(data,%s,%s)" % N, setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t

            self.print_exec(tmin)
            print("")
            if 1:  # R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
            gc.collect()
        self.print_sep()
        label = "2D_CPU_serial"
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)

    def bench_cpu2d_lut(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = (500, 360)
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            _ = ai.integrate2d(data, N[0], N[1], unit="2th_deg", method="lut")
            t1 = time.time()
            self.print_init(t1 - t0)
            print("Size of the LUT: %.3fMByte" % (ai._lut_integrator.lut.nbytes / 1e6))
            del ai, data

            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
out=ai.integrate2d(data,%s,%s,unit="2th_deg", method="lut")""" % (param, fn, N[0], N[1])
            t = timeit.Timer("out=ai.integrate2d(data,%s,%s,unit='2th_deg', method='lut')" % N, setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t

            self.print_exec(tmin)
            print("")
            if 1:  # R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
            gc.collect()
        self.print_sep()
        label = "2D_CPU_parallel_OpenMP"
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)

    def bench_cpu2d_lut_ocl(self, devicetype="ALL", platformid=None, deviceid=None):
        if (ocl is None):
            print("No pyopencl")
            return
        if (platformid is None) or (deviceid is None):
            platdev = ocl.select_device(devicetype)
            if not platdev:
                print("No such OpenCL device: skipping benchmark")
                return
            platformid, deviceid = platdev
        print("Working on device: %s platform: %s device: %s" % (devicetype, ocl.platforms[platformid], ocl.platforms[platformid].devices[deviceid]))
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = (500, 360)
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(fn), size / 1e6, N))
            t0 = time.time()
            try:
                _ = ai.integrate2d(data, N[0], N[1], unit="2th_deg", method="lut_ocl_%i,%i" % (platformid, deviceid))
            except MemoryError as error:
                print(error)
                break
            t1 = time.time()
            self.print_init(t1 - t0)
            print("Size of the LUT: %.3fMByte" % (ai._lut_integrator.lut.nbytes / 1e6))
            del ai, data

            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
out=ai.integrate2d(data,%s,%s,unit="2th_deg", method="lut_ocl_%i,%i")""" % (param, fn, N[0], N[1], platformid, deviceid)
            t = timeit.Timer("out=ai.integrate2d(data,%s,%s,unit='2th_deg', method='lut_ocl')" % N, setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t

            self.print_exec(tmin)
            print("")
            if 1:  # R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
            gc.collect()
        self.print_sep()
        label = "2D_%s_parallel_OpenCL" % devicetype.upper()
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)


    def bench_gpu1d(self, devicetype="gpu", useFp64=True, platformid=None, deviceid=None):
        print("Working on %s, in " % devicetype + ("64 bits mode" if useFp64 else"32 bits mode") + "(%s.%s)" % (platformid, deviceid))
        if ocl is None or not ocl.select_device(devicetype):
            print("No pyopencl or no such device: skipping benchmark")
            return
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            size = data.size
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (op.basename(fn), size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

            try:
                t0 = time.time()
                res = ai.xrpd_OpenCL(data, N, devicetype=devicetype, useFp64=useFp64, platformid=platformid, deviceid=deviceid)
                t1 = time.time()
            except Exception as error:
                print("Failed to find an OpenCL GPU (useFp64:%s) %s" % (useFp64, error))
                continue
            self.print_init(t1 - t0)
            ref = ai.xrpd(data, N)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            del ai, data
            setup = """
#gc.enable()
import pyFAI,fabio
ai=pyFAI.load(r"%s")
data = fabio.open(r"%s").data
N=min(data.shape)
out=ai.xrpd_OpenCL(data,N, devicetype=r"%s", useFp64=%s, platformid=%s, deviceid=%s)""" % (param, fn, devicetype, useFp64, platformid, deviceid)
            t = timeit.Timer("ai.xrpd_OpenCL(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            del t
            gc.collect()
            self.print_exec(tmin)
            print("")
            if R < self.LIMIT:
                results[size / 1e6] = tmin * 1000.0
        self.print_sep()
        label = "Forward_OpenCL_%s_%s_bits" % (devicetype , ("64" if useFp64 else"32"))
        self.meth.append(label)
        self.results[label] = results
        self.new_curve(results, label)

    def save(self, filename="benchmark.json"):
        import json
        json.dump(self.results, open(filename, "w"))

    def print_res(self):

        print("Summary: execution time in milliseconds")
        print "Size/Meth\t" + "\t".join(b.meth)
        for i in self.size:
            print "%7.2f\t\t" % i + "\t\t".join("%.2f" % (b.results[j].get(i, 0)) for j in b.meth)

    def init_curve(self):
        if self.fig:
            print("Already initialized")
            return
        if "DISPLAY" in os.environ:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_xlabel("Image size in Mega-Pixels")
            self.ax.set_ylabel("Frames processed per second")
            self.ax.set_yscale("log", basey=2)
            t = [1, 2, 5, 10, 20, 50, 100, 200]
            self.ax.set_yticks([float(i) for i in t])
            self.ax.set_yticklabels([str(i)for i in t])
            self.ax.set_title(self.get_cpu() + " / " + self.get_gpu())
            if self.fig.canvas:
                self.fig.canvas.draw()
#            plt.show()

    def new_curve(self, results, label):
        if not self.fig:
            return
        s = []
        p = []
        for i in self.size:
            if i in results:
                s.append(i)
                p.append(1000.0 / results[i])
        self.ax.plot(s, p, label=label)

        self.ax.legend()
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
    from optparse import OptionParser
    description = """Benchmark for Azimuthal integration
    """
    epilog = """  """
    usage = """usage: %prog [options] """
    version = "%prog " + pyFAI.version
    parser = OptionParser(usage=usage, version=version,
                          description=description, epilog=epilog)
    parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose", default=False,
                          help="switch to verbose/debug mode")
    parser.add_option("-c", "--cpu",
                      action="store_true", dest="opencl_cpu", default=False,
                      help="perform benchmark using OpenCL on the CPU")
    parser.add_option("-g", "--gpu",
                      action="store_true", dest="opencl_gpu", default=False,
                      help="perform benchmark using OpenCL on the GPU")
    parser.add_option("-a", "--acc",
                      action="store_true", dest="opencl_acc", default=False,
                      help="perform benchmark using OpenCL on the Accelerator (like XeonPhi/MIC")
    parser.add_option("-s", "--small",
                      action="store_true", dest="small", default=False,
                      help="Limit the size of the dataset to 6 Mpixel images (adviced for python 2.6 where there is a memory-leak)" )
    parser.add_option("-n", "--number",
                      dest="number", default=10, type="int",
                      help="Number of repetition of the test, by default 10" )
    (options, args) = parser.parse_args()
    if options.small:
        ds_list = ds_list[:4]
    if options.verbose:
            pyFAI.logger.setLevel(logging.DEBUG)
    print("Averaging over %i repetitions (best of 3)." % options.number)
    b = Bench(options.number)
    b.init_curve()
    b.bench_cpu1d()
    b.bench_cpu1d_lut()
    if options.opencl_cpu:
        b.bench_cpu1d_lut_ocl("CPU")
    if options.opencl_gpu:
        b.bench_cpu1d_lut_ocl("GPU")
    if options.opencl_acc:
        b.bench_cpu1d_lut_ocl("ACC")

#    b.bench_cpu1d_ocl_lut("CPU")
#    b.bench_gpu1d("gpu", True)
#    b.bench_gpu1d("gpu", False)
#    b.bench_gpu1d("cpu", True)
#    b.bench_gpu1d("cpu", False)
    b.bench_cpu2d()
    b.bench_cpu2d_lut()
    if options.opencl_cpu:
        b.bench_cpu2d_lut_ocl("CPU")
    if options.opencl_gpu:
        b.bench_cpu2d_lut_ocl("GPU")
    if options.opencl_acc:
        b.bench_cpu2d_lut_ocl("ACC")

#    b.bench_cpu2d_lut()
#    b.bench_cpu2d_lut_ocl()
    b.save()
    b.print_res()
#    b.display_all()
    b.ax.set_ylim(1, 200)
    # plt.show()
    plt.ion()
    raw_input("Enter to quit")
