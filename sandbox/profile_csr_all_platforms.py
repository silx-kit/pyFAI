# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:      Giannis Ashiotis
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
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from  pyFAI.test import utilstest
import fabio, pyopencl
from pylab import *
try:
    from pyFAI.third_party import six
except (ImportError, Exception):
    import six
from six.moves import range
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
from pyFAI import splitBBoxLUT
from pyFAI import splitBBoxCSR
from pyFAI import ocl_azim_csr


def prof_inte(csr, data, device, block_size, repeat=10, nbr=3, platformid=None, deviceid=None):
    runtimes = []
    for foo in range(nbr):
        t = []
        ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(csr, data.size, device, profile=True, block_size=block_size, platformid=platformid, deviceid=deviceid)
        for boo in range(repeat + 1):
            ocl_csr.integrate(data)
        for e in ocl_csr.events:
            if "integrate" in e[0]:
                et = 1e-6 * (e[1].profile.end - e[1].profile.start)
#                print("%50s:\t%.3fms" % (e[0], et))
                t.append(et)
        runtimes.append(numpy.average(t[1:]))
    return numpy.min(runtimes)

if __name__ == "__main__":
    ponifile = utilstest.UtilsTest.getimage("Pilatus1M.poni")
    datafile = utilstest.UtilsTest.getimage("Pilatus1M.edf")

    ai = pyFAI.load(ponifile)
    data = fabio.open(datafile).data
    ai.xrpd_LUT(data, 1000)[1]

    t0 = time.time()
    cyt_csr = pyFAI.splitBBoxCSR.HistoBBox1d(
                    ai._ttha,
                    ai._dttha,
                    bins=1000,
                    unit="2th_deg")
    t1 = time.time()
    timimgs = {}
    print("Time to create cython CSR: ", t1 - t0)
    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for device in [(0, 0), (1, 0), (2, 0), (2, 1)]:
        timimgs[device] = []
        for block_size in block_sizes:
            t = prof_inte(cyt_csr.lut, data, "ALL", block_size, nbr=3, repeat=10, platformid=device[0], deviceid=device[1])
            timimgs[device].append(t)

    for i in timimgs:
        plot(block_sizes, timimgs[i], label=str(i))
    legend()
    show()
    six.moves.input()
