#!/usr/bin/python3
import logging
logger = logging.basicConfig(level=logging.INFO)
import numpy, pyFAI, pyFAI.azimuthalIntegrator
method = ("no", "csr", "cython")
detector = pyFAI.detector_factory("Pilatus_100k")
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(detector=detector)
rm = max(detector.shape) * detector.pixel1
img = numpy.random.random(detector.shape)
print(ai.integrate1d(img, 5, unit="r_m", radial_range=[0, rm], method=method))
# print(ai.integrate1d(img, 5, unit="r_m", method=method))
for k, v in ai.engines.items():
    print(k, v, id(v.engine))

print(ai.integrate1d(img, 5, unit="r_m", radial_range=[0, rm], method=method))
# print(ai.integrate1d(img, 5, unit="r_m", method=method))
for k, v in ai.engines.items():
    print(k, v, id(v.engine))
