import sys,os
import distutils.util
platform = distutils.util.get_platform()
architecture = "lib.%s-%i.%i" % (platform,
                                            sys.version_info[0], sys.version_info[1])

sys.path.insert(0,os.path.join("build",architecture))
import numpy
import pyFAI
print pyFAI
import sys
import gc
def get_mem():
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
pos0=numpy.arange(2048*2048).reshape(2048,2048)
dpos0=numpy.ones_like(pos0)
print("Instancition 1")
lut = pyFAI.splitBBoxLUT.HistoBBox1d(pos0,dpos0,bins=800)
print("Size of LUT: %s"%lut.lut.nbytes)
print("ref count of lut.lut: %s %s"%(sys.getrefcount(lut),sys.getrefcount(lut.lut)))
print(sys.getrefcount(lut.cpos0),sys.getrefcount(lut.dpos0),sys.getrefcount(lut.lut))
print()
print("Cpos0, refcount=: %s %s"%(sys.getrefcount(lut.cpos0),len(gc.get_referrers(lut.cpos0))))
for obj in gc.get_referrers(lut.cpos0):
    print("Cpos0: %s"%str(obj)[:100])
print()
#print(gc.get_referrers(lut.dpos0))
print("Lut, refcount=: %s %s"%(sys.getrefcount(lut.lut), len(gc.get_referrers(lut.lut))))
for obj in gc.get_referrers(lut.lut):
    print("Lut: %s"%str(obj)[:100])
import pyFAI.splitBBoxCSR
lut = pyFAI.splitBBoxCSR.HistoBBox1d(pos0, dpos0, bins=800)
print("Size of LUT: %s" % lut.nnz)
print("ref count of lut.lut: %s %s" % (sys.getrefcount(lut), sys.getrefcount(lut.data)))
print(sys.getrefcount(lut.cpos0), sys.getrefcount(lut.dpos0), sys.getrefcount(lut.data))
print()
print("Cpos0, refcount=: %s %s" % (sys.getrefcount(lut.cpos0), len(gc.get_referrers(lut.cpos0))))
for obj in gc.get_referrers(lut.cpos0):
    print("Cpos0: %s" % str(obj)[:100])
print()
#print(gc.get_referrers(lut.dpos0))
print("Lut, refcount=: %s %s" % (sys.getrefcount(lut.data), len(gc.get_referrers(lut.data))))
for obj in gc.get_referrers(lut.data):
    print("Lut: %s" % str(obj)[:100])


print("Finished ")
while True:
    lut = pyFAI.splitBBoxLUT.HistoBBox1d(pos0,dpos0,bins=800)
    print(sys.getrefcount(lut.lut))
    lut.integrate(numpy.random.random(pos0.shape))
    print("Memory: %s, lut size: %s, refcount: %s" % (get_mem(), lut.lut.nbytes / 2 ** 20, sys.getrefcount(lut.lut)))
