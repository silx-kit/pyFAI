#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import absolute_import, print_function, division, with_statement

import os
import numpy
import sys
import gc
import pyFAI

print(pyFAI)

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

pos0 = numpy.arange(2048 * 2048).reshape(2048, 2048)
dpos0 = numpy.ones_like(pos0)
print("Instancition 1")
lut = pyFAI.splitBBoxLUT.HistoBBox1d(pos0, dpos0, bins=800)
print("Size of LUT: %s" % lut.lut.nbytes)
print("ref count of lut.lut: %s %s" % (sys.getrefcount(lut), sys.getrefcount(lut.lut)))
print(sys.getrefcount(lut.cpos0), sys.getrefcount(lut.dpos0), sys.getrefcount(lut.lut))
print()
print("Cpos0, refcount=: %s %s" % (sys.getrefcount(lut.cpos0), len(gc.get_referrers(lut.cpos0))))
for obj in gc.get_referrers(lut.cpos0):
    print("Cpos0: %s" % str(obj)[:100])
print()
# print(gc.get_referrers(lut.dpos0))
print("Lut, refcount=: %s %s" % (sys.getrefcount(lut.lut), len(gc.get_referrers(lut.lut))))
for obj in gc.get_referrers(lut.lut):
    print("Lut: %s" % str(obj)[:100])
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
# print(gc.get_referrers(lut.dpos0))
print("Lut, refcount=: %s %s" % (sys.getrefcount(lut.data), len(gc.get_referrers(lut.data))))
for obj in gc.get_referrers(lut.data):
    print("Lut: %s" % str(obj)[:100])


print("Finished ")
while True:
    lut = pyFAI.splitBBoxLUT.HistoBBox1d(pos0, dpos0, bins=800)
    print(sys.getrefcount(lut.lut))
    lut.integrate(numpy.random.random(pos0.shape))
    print("Memory: %s, lut size: %s, refcount: %s" % (get_mem(), lut.lut.nbytes / 2 ** 20, sys.getrefcount(lut.lut)))
