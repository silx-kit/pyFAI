#!/usr/bin/python

#Benchmark for Distortion correction in PyFAI

from __future__ import print_function, division

import json, sys, time, timeit, os, platform, subprocess, gc, logging
logging.basicConfig(level=logging.ERROR)
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

repeat = 1
number = 1

print("Number of iteration: %s average over %s processing" % (repeat, number))
for method in ("LUT","CSR"):
    for device in ("None", '"CPU"', '"GPU"'):
        for wg in [1, 2, 4, 8, 16, 32]:
            if (method != "CSR" or device == "None")and wg > 1:
                continue
            setup = """
import pyFAI, pyFAI.distortion, numpy
detector = pyFAI.detectors.FReLoN("%s")
dis = pyFAI.distortion.Distortion(detector, method='%s', device=%s, workgroup=%s)
data = numpy.random.randint(0,65000,size=detector.shape[0]*detector.shape[1]).reshape(detector.shape).astype(numpy.uint16)
dis.calc_init()""" % (splinefile, method, device, wg)
            t = timeit.Timer("dis.correct(data)", setup)
            tmin = min([i / number for i in t.repeat(repeat=repeat, number=number)])
            print("%s %s (wg=%2s) t=%.3fms" % (method, device, wg, tmin * 1000.0))
            gc.collect
