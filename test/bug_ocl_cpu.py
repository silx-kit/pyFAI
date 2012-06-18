#!/usr/bin/python
import pyFAI, numpy
ai = pyFAI.load("moke.poni")
shape = (600, 600)
ai.xrpd_OpenCL(numpy.ones(shape), 500, devicetype="cpu", useFp64=False)
