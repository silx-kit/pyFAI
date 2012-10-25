#!/usr/bin/python
import fabio, sys, time, timeit
import os.path as op

sys.path.append(op.join(op.dirname(op.dirname(op.abspath(__file__))), "test"))
import utilstest
pyFAI = utilstest.UtilsTest.pyFAI


ds_list = ["Pilatus1M.poni", "halfccd.poni", "Frelon2k.poni", "Pilatus6M.poni", "Mar3450.poni", "Fairchild.poni"]
#ds_list = ["Pilatus6M.poni"]
datasets = {"Fairchild.poni":utilstest.UtilsTest.getimage("1880/Fairchild.edf"),
            "halfccd.poni":utilstest.UtilsTest.getimage("1882/halfccd.edf"),
            "Frelon2k.poni":utilstest.UtilsTest.getimage("1881/Frelon2k.edf"),
            "Pilatus6M.poni":utilstest.UtilsTest.getimage("1884/Pilatus6M.cbf"),
            "Pilatus1M.poni":utilstest.UtilsTest.getimage("1883/Pilatus1M.edf"),
            "Mar3450.poni":utilstest.UtilsTest.getimage("2201/LaB6_260210.mar3450")
      }

print pyFAI
class Bench(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    reference_1d = {}
    LIMIT = 8
    repeat = 3
    nbr = 10
    results = {}
    def get_cpu(self):
        return [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("model name")][0].strip()


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
            ai = None
        return self.reference_1d[param]



    def bench_cpu1d(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            ref = self.get_ref(param)
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), data.size / 1e6, N))
            t0 = time.time()
            res = ai.xrpd_LUT(data, N)
            t1 = time.time()
            self.print_init(t1 - t0)
            print "lut.shape=", ai._lut_integrator.lut.shape, "lut.nbytes (MB)", ai._lut_integrator.size * 8 / 1e6
            del ai
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd_LUT(data,N)""" % (param, fn)
            t = timeit.Timer("ai.xrpd_LUT(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            if R < self.LIMIT:
                results[data.size / 1e6] = tmin * 1000.0
        self.print_sep()
        self.results["LUT_Cython_OpenMP"] = results

    def bench_cpu1d_ocl_lut(self, devicetype="all", platformid=None, deviceid=None):
        print("Working on device: %s" % devicetype)
        results = {}
        for param in ds_list:
            ref = self.get_ref(param)
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (op.basename(fn), data.size / 1e6, N))
            t0 = time.time()
            try:
                res = ai.xrpd_LUT_OCL(data, N, devicetype=devicetype, platformid=platformid, deviceid=deviceid)
            except MemoryError:
                print("Not enough memory")
                return
            t1 = time.time()
            self.print_init(t1 - t0)
            del ai
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd_LUT_OCL(data,N,devicetype="%s",platformid=%s,deviceid=%s)""" % (param, fn, devicetype, platformid, deviceid)
            t = timeit.Timer("ai.xrpd_LUT_OCL(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > self.LIMIT else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            if R < self.LIMIT:
                results[data.size / 1e6] = tmin * 1000.0
        self.print_sep()
        self.results["LUT_OpenCL_%s" % devicetype] = results

    def bench_cpu2d(self):
        print("Working on processor: %s" % self.get_cpu())
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = (500, 360)
            print("2D integration of %s %.1f Mpixel -> %s bins" % (op.basename(fn), data.size / 1e6, N))
            t0 = time.time()
            _ = ai.xrpd2(data, N[0], N[1])
            t1 = time.time()
            self.print_init(t1 - t0)
            del ai
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
out=ai.xrpd2(data,500,360)""" % (param, fn)
            t = timeit.Timer("ai.xrpd2(data,500,360)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
            print("")
            if 1:#R < self.LIMIT:
                results[data.size / 1e6] = tmin * 1000.0
        self.print_sep()
        self.results["Foward_2D_CPU" ] = results


    def bench_gpu1d(self, devicetype="gpu", useFp64=True, platformid=None, deviceid=None):
        print("Working on %s, in " % devicetype + ("64 bits mode" if useFp64 else"32 bits mode") + "(%s.%s)" % (platformid, deviceid))
        results = {}
        for param in ds_list:
            fn = datasets[param]
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (op.basename(fn), data.size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

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
            del ai
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd_OpenCL(data,N, devicetype="%s", useFp64=%s, platformid=%s, deviceid=%s)""" % (param, fn, devicetype, useFp64, platformid, deviceid)
            t = timeit.Timer("ai.xrpd_OpenCL(data,N,safe=False)", setup)
            tmin = min([i / self.nbr for i in t.repeat(repeat=self.repeat, number=self.nbr)])
            self.print_exec(tmin)
            print("")
            if R < self.LIMIT:
                results[data.size / 1e6] = tmin * 1000.0
        self.print_sep()
        self.results["Foward_OpenCL_%s_%s_bits" % (devicetype , ("64" if useFp64 else"32"))] = results
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
    else:
        n = 1
    print("Averaging over %i repetitions (best of 3)." % n)
    b = Bench()
    b.nbr = n
    b.bench_cpu1d()
    b.bench_cpu1d_ocl_lut("GPU")
    b.bench_cpu1d_ocl_lut("CPU", 2, 0)
    b.bench_cpu1d_ocl_lut()
    b.bench_cpu2d()
    b.bench_gpu1d("gpu", True)
    b.bench_gpu1d("gpu", False)
    b.bench_gpu1d("cpu", True)
    b.bench_gpu1d("cpu", False)
    meth = list(b.results.keys())
    size = list(b.results[meth[0]].keys())
    size.sort()
    print("Summary: execution time in milliseconds")
    print "Size/Meth\t" + "\t".join(meth)
    for i in size:
        print "%7.2f\t\t" % i + "\t\t".join("%.2f" % (b.results[j].get(i, 0)) for j in meth)


