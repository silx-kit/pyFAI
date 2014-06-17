#!/usr/bin/python

#Benchmark for Distortion correction in PyFAI

from __future__ import print_function, division

import json, sys, time, timeit, os, platform, subprocess, gc
import numpy
import fabio
import os.path as op
sys.path.append(op.join(op.dirname(op.dirname(op.abspath(__file__))), "test"))
import utilstest

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    print("No socket opened for debugging -> please install rfoo")

#We use the locally build version of PyFAI
pyFAI = utilstest.UtilsTest.pyFAI
ocl = pyFAI.opencl.ocl
import matplotlib
matplotlib.use("gtk")
from matplotlib import pyplot as plt
plt.ion()

splinefile = utilstest.UtilsTest.getimage("1900/frelon.spline")

detector = pyFAI.detectors.FReLoN(splinefile)

repeat = 3
number = 10

print("Number of iteration: %s average over %s processing" % (repeat, number))
for method in ("LUT","CSR"):
    for device in ("None",(0,0),(0,1),(1,0),(2,0)):
        setup = """
import pyFAI, pyFAI.distortion, numpy
detector = pyFAI.detectors.FReLoN("%s")
dis = pyFAI.distortion.Distortion(detector, method='%s', device=%s)
data = numpy.random.random(detector.shape).astype(numpy.float32)
dis.calc_init()""" % (splinefile, method, device)
        t = timeit.Timer("dis.correct(data)", setup)
        tmin = min([i / number for i in t.repeat(repeat=repeat, number=number)])
        print("%s %s t=%.6fs" % (method, device, tmin))
        gc.collect
