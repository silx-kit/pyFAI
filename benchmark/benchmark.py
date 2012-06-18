#!/usr/bin/python
import fabio, sys, time, timeit
from os.path import abspath, join, abspath, dirname

sys.path.append(join(dirname(dirname(abspath(__file__))), "test"))
import utilstest
pyFAI = utilstest.UtilsTest.pyFAI

datasets = {"Fairchild.poni":utilstest.UtilsTest.getimage("1880/Fairchild.edf"),
            "halfccd.poni":utilstest.UtilsTest.getimage("1882/halfccd.edf"),
            "Frelon2k.poni":utilstest.UtilsTest.getimage("1881/Frelon2k.edf"),
            "Pilatus6M.poni":utilstest.UtilsTest.getimage("1884/Pilatus6M.cbf"),
            "Pilatus1M.poni":utilstest.UtilsTest.getimage("1883/Pilatus1M.edf"),
      }

print pyFAI
class Bench(object):

    def get_cpu(self):
        return [i.split(": ", 1)[1] for i in open("/proc/cpuinfo") if i.startswith("model name")][0].strip()


    def print_init(self, t):
        print(" * Initialization time: %.1f ms" % (1000.*t))


    def print_exec(self, t):
        print(" * Execution time rep : %.1f ms" % (t * 1000))


    def print_sep(self):
        print("*"*80)


    def bench_cpu1d(self, n=10):
        print("Working on processor: %s" % self.get_cpu())
        for param, fn in datasets.items():
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins" % (fn, data.size / 1e6, N))
            t0 = time.time()
            _ = ai.xrpd(data, N)
            t1 = time.time()
            self.print_init(t1 - t0)
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd(data,N)""" % (param, fn)
            t = timeit.Timer("ai.xrpd(data,N)", setup)
            tmin = min([i / n for i in t.repeat(repeat=5, number=n)])
            self.print_exec(tmin)
            print("")
        self.print_sep()


    def bench_cpu2d(self, n=10):
        print("Working on processor: %s" % self.get_cpu())
        for param, fn in datasets.items():
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = (500, 360)
            print("2D integration of %s %.1f Mpixel -> %s bins" % (fn, data.size / 1e6, N))
            t0 = time.time()
            _ = ai.xrpd2(data, N[0], N[1])
            t1 = time.time()
            self.print_init(t1 - t0)
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
out=ai.xrpd2(data,500,360)""" % (param, fn)
            t = timeit.Timer("ai.xrpd2(data,500,360)", setup)
            tmin = min([i / n for i in t.repeat(repeat=5, number=n)])
            self.print_exec(tmin)
            print("")
        self.print_sep()


    def bench_gpu1d(self, n=10, useFp64=True):
        print("Working on default GPU, in " + ("64 bits mode" if useFp64 else"32 bits mode"))

        for param, fn in datasets.items():
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            print("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (fn, data.size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

            try:
                t0 = time.time()
                _ = ai.xrpd_OpenCL(data, N, devicetype="gpu", useFp64=useFp64)
                t1 = time.time()
            except Exception as error:
                print("Failed to find an OpenCL GPU (useFp64:%s) %s" % (useFp64, error))
                continue
            else:
                ai._ocl.print_devices()

            ai = None
            self.print_init(t1 - t0)
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd_OpenCL(data,N,devicetype="gpu", useFp64=%s)""" % (param, fn, useFp64)
            t = timeit.Timer("ai.xrpd_OpenCL(data,N,safe=False)", setup)
            tmin = min([i / n for i in t.repeat(repeat=5, number=n)])
            self.print_exec(tmin)
            print("")
        self.print_sep()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
    else:
        n = 1
    print("Averaging over %i repetitions (best of 5)." % n)
    b = Bench()
    b.bench_cpu1d(n)
    b.bench_cpu2d(n)
    b.bench_gpu1d(n, True)
    b.bench_gpu1d(n, False)

