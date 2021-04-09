#!/usr/bin/python3
import logging
logging.basicConfig(level=logging.INFO)
import numpy
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.opencl.peak_finder
shape = 2048, 2048
npeaks = 100
nbins = 512
numpy.random.seed(0)

img = numpy.ones(shape, dtype="float32")
variance = img.copy()
peaks = numpy.random.randint(0, shape[0] * shape[1], size=npeaks)
img.ravel()[peaks] = 4e9
print(img.shape, img.mean(), img.std())
# or a in zip(peaks//shape[1], peaks%shape[1]): print(a)

JF4 = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=shape)
ai = AzimuthalIntegrator(detector=JF4)
ai.setFit2D(100, shape[1] // 2, shape[0] // 2)
csr = ai.setup_CSR(None, nbins, unit="r_m", split="no").lut

r2 = ai.array_from_unit(unit="r_m")
res = ai.integrate1d(img, nbins, unit="r_m")
pf = pyFAI.opencl.peak_finder.OCL_PeakFinder(csr, img.size, bin_centers=res[0], radius=r2, profile=True)
print(pf.count(img, error_model="azimuthal", cutoff_clip=6), npeaks)
# res = pf(img, variance=variance)
# for a in zip(res[0] // shape[1], res[0] % shape[1], res[1]): print(a)
pf.log_profile(stats=True)
