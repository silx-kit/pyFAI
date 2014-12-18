# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from . import utilstest
import fabio, pyopencl
from pylab import *
from six.moves import range
from six.moves import input
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
from pyFAI import splitBBoxLUT
from pyFAI import splitBBoxCSR
from pyFAI import ocl_azim_csr


def prof_inte(csr, data, device, block_size, repeat=10, nbr=3, platformid=None,deviceid=None ):
    runtimes=[]
    for foo in range(nbr):
        t=[]
        ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(csr, data.size, device, profile=True, block_size=block_size, platformid=platformid, deviceid=deviceid )
        for boo in range(repeat+1):
            ocl_csr.integrate(data)
        for e in ocl_csr.events:
            if "integrate" in e[0]:
                et = 1e-6 * (e[1].profile.end - e[1].profile.start)
#                print("%50s:\t%.3fms" % (e[0], et))
                t.append(et)
        runtimes.append(numpy.average(t[1:]))
    return numpy.min(runtimes)



if __name__ == "__main__":
    ai = pyFAI.load("testimages/Pilatus1M.poni")
    data = fabio.open("testimages/Pilatus1M.edf").data
    ai.xrpd_LUT(data, 1000)[1]

    t0 = time.time()                
    cyt_csr = pyFAI.splitBBoxCSR.HistoBBox1d(
                    ai._ttha,
                    ai._dttha,
                    bins=1000,
                    unit="2th_deg")
    t1 = time.time()
    timimgs={}
    print("Time to create cython CSR: ", t1-t0)
    block_sizes = [1,2,4,8,16,32,64,128]

    for device in [(0,0),(1,0),(2,0),(2,1)]:
        timimgs[device]=[]
        for block_size in block_sizes:
             t=prof_inte(cyt_csr.lut, data, "ALL", block_size, nbr=3, repeat=10, platformid=device[0], deviceid=device[1])
             timimgs[device].append(t)
    
    for i in timimgs:
        plot(block_sizes, timimgs[i], label=str(i))
    legend()
    show()
    input()

