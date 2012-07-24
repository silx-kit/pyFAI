#!/usr/bin/python
import logging, time, sys, os
import numpy
import fabio
import h5py
import pyFAI
logger = logging.getLogger("diff_tomo")



class DiffTomo(object):
    """
    Basic class for diffraction tomography using pyFAI
    """
    def __init__(self, nTrans, nRot, nDiff=1000):
        """
        Contructor of the class
        
        @param nTrans: number of translations
        @param nRot: number of translations
        @param nDiff: number of points in diffraction pattern
        """
        self.nTrans = None
        self.nRot = None
        self.nDiff = nDiff
        self.poni = None
        self.dark = None
        self.flat = None
        self.I0 = None

    def __repr__(self):
        return "Diffraction Tomography with r=%s t: %s, d:%s" % (self.nRot, self.nTrans, self.nDiff)

    def parse(self):

def get_pos(fn, nt, offset=0):
    n = int(fn.split(".")[0].split("_")[-1]) - offset
    return n // nt, n % nt

def main(dirname, out, nt, ai, offset=0):
    t0 = time.time()
    timing = []
    lst = [os.path.abspath(os.path.join(dirname, i)) for i in os.listdir(dirname) if i.endswith(".edf")]
    lst.sort()
    if os.path.exists(out):
        os.unlink(out)
    h = h5py.File(out)
    ds = h.require_dataset(name="data",
                           shape=(1, nt, SIZE),
                           dtype="float32",
                           chunks=(1, nt, SIZE),
                           maxshape=(None, nt, SIZE))
    no_max = 1
    for i in lst:
        t = time.time()
        j, k = get_pos(i, nt, offset)
        if  j + 1 > no_max:
            no_max = j + 1
            ds.resize((no_max, nt, SIZE))
        elif j < 0 or k < 0:
            continue
        ds[j, k, :] = ai.xrpd_OpenCL(fabio.open(i).data, SIZE)[-1]
        t -= time.time()
        print("Processing %s took %3.1fm" % (os.path.basename(i), -1000 * t))
        timing.append(-t)
    tot = time.time() - t0
    cnt = len(timing)
    print("Execution time for %i frames: %.3fs; Average execution time: %.1fms" % (cnt, tot, 1000.*tot / cnt))
    return timing

if __name__ == "__main__":
    offset = 0
    for k in sys.argv[1:]:
        if "-offset" in k:
            offset = int(k.split("=")[-1])
            sys.argv.remove(k)
    try:
       dirname, out, nt, calib = sys.argv[1:5]
    except:
       print ("analyse_xrd xrd /tmp/toto.h5 225 calib.poni")
    else:
       x = main(dirname, out, int(nt), pyFAI.load(calib), offset)
       import numpy as np
       import matplotlib.mlab as mlab
       import matplotlib.pyplot as plt
       n, bins, patches = plt.hist(x, 500, facecolor='green', alpha=0.75)
       plt.xlabel('Execution time in sec')
       plt.title("Execution time")
       plt.grid(True)
       plt.show()



