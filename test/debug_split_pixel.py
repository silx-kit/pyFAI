#!/usr/bin/python
import pyFAI, numpy
img = numpy.zeros((512,512))
for i in range(1,6):img[i*100,i*100]=1
det = pyFAI.detectors.Detector(1e-4,1e-4)
det.shape=(512,512)
ai=pyFAI.AzimuthalIntegrator(1,detector=det)
import pylab
from utilstest import Rwp
results = {}
for i, meth in enumerate(["cython", "splitbbox", "splitpixel", "csr_no", "csr_bbox", "csr_full"]):
    tth, I = ai.integrate1d(img, 10000, method=meth, unit="2th_deg")
    pylab.plot(tth, I + i * 1e-3, label=meth)
    ai.reset()
    results[meth]=tth, I
print("no_split R=%.3f" % Rwp(results["csr_no"], results["cython"]))
print("split_bbox R=%.3f" % Rwp(results["csr_bbox"], results["splitbbox"]))
print("split_full R=%.3f" % Rwp(results["csr_full"], results["splitpixel"]))
pylab.legend()
pylab.ion()
pylab.show()
raw_input("enter_to_quit")
