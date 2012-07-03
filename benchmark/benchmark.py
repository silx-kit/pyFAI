#!/usr/bin/python
import fabio, sys, time, timeit
import os.path as op

sys.path.append(op.join(op.dirname(op.dirname(op.abspath(__file__))), "test"))
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
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

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


    def bench_gpu1d(self, n=10, devicetype="gpu", useFp64=True, platformid=None, deviceid=None):
        print("Working on %s, in " % devicetype + ("64 bits mode" if useFp64 else"32 bits mode") + "(%s.%s)" % (platformid, deviceid))

        for param, fn in datasets.items():
            ai = pyFAI.load(param)
            data = fabio.open(fn).data
            N = min(data.shape)
            sys.stdout.write("1D integration of %s %.1f Mpixel -> %i bins (%s)" % (fn, data.size / 1e6, N, ("64 bits mode" if useFp64 else"32 bits mode")))

            try:
                t0 = time.time()
#                print "bench", devicetype, useFp64, platformid, deviceid
                res = ai.xrpd_OpenCL(data, N, devicetype=devicetype, useFp64=useFp64, platformid=platformid, deviceid=deviceid)
                t1 = time.time()
            except Exception as error:
                print("Failed to find an OpenCL GPU (useFp64:%s) %s" % (useFp64, error))
                continue
#            print(" device: %s.%s" % ai._ocl.get_contexed_Ids())
#            print(ai._ocl.get_platform_info())
#            print(ai._ocl.get_device_info())

            self.print_init(t1 - t0)
            ref = ai.xrpd(data, N)
            R = utilstest.Rwp(res, ref)
            print("%sResults are bad with R=%.3f%s" % (self.WARNING, R, self.ENDC) if R > 6 else"%sResults are good with R=%.3f%s" % (self.OKGREEN, R, self.ENDC))
            ai = None
            setup = """
import pyFAI,fabio
ai=pyFAI.load("%s")
data = fabio.open("%s").data
N=min(data.shape)
out=ai.xrpd_OpenCL(data,N, devicetype="%s", useFp64=%s, platformid=%s, deviceid=%s)""" % (param, fn, devicetype, useFp64, platformid, deviceid)
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
#    b.bench_cpu2d(n)
    b.bench_gpu1d(n, "gpu", True,0,0)
    b.bench_gpu1d(n, "gpu", False,0,0)
#    b.bench_gpu1d(n, "all", True, 0, 1)
#    b.bench_gpu1d(n, "all", False, 0, 1)
#    b.bench_gpu1d(n, "all", True, 1, 0)
#    b.bench_gpu1d(n, "all", False, 1, 0)
#    b.bench_gpu1d(n, "all", True, 2, 0)
#    b.bench_gpu1d(n, "all", False, 2, 0)
