# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from . import utilstest
import fabio
import pyopencl as cl
from pylab import *
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import splitPixelFullLUT_float32
from pyFAI import ocl_hist_pixelsplit
#from pyFAI import splitBBoxLUT
from pyFAI import splitBBoxCSR
from pyFAI import splitPixelFull
import scipy
#logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/halfccd.poni")
data = fabio.open("testimages/halfccd.edf").data

workgroup_size = 256
bins = 1000

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size/8,4,2)

pos_size = pos.size
#size = data.size
size = pos_size/8



boo = splitPixelFullLUT_float32.HistoLUT1dFullSplit(pos,bins, unit="2th_deg")

matrix_32 =  scipy.sparse.csr_matrix((boo.data,boo.indices,boo.indptr), shape=(bins,data.size))
mat32d = matrix_32.todense()
#mat32d.shape = (mat32d.size,)
#out = boo.integrate(data)

#ai.xrpd_LUT(data, 1000)

#ref = ai.integrate1d(data,bins,unit="2th_deg", correctSolidAngle=False, method="lut")

foo = splitPixelFullLUT.HistoLUT1dFullSplit(pos,bins, unit="2th_deg")

matrix_64 =  scipy.sparse.csr_matrix((foo.data,foo.indices,foo.indptr), shape=(bins,data.size))
mat64d = matrix_64.todense()
#mat64d.shape = (mat64d.size,)
#foo = splitBBoxCSR.HistoBBox1d(ai._ttha, ai._dttha, bins=bins, unit="2th_deg")


bools_bad = (abs(mat32d - mat64d) > 0.000001)
#bools_good = (abs(mat32d - mat64d) <= 0.000001)

del mat32d
del mat64d
del matrix_32
del matrix_64

tmp = numpy.where(bools_bad)[1].ravel()
pixels_bad = numpy.copy(tmp)
pixels_bad.sort()

#tmp = numpy.where(bools_good)[1]
#pixels_good = numpy.copy(tmp)
#pixels_good.sort()







#ref = splitPixelFull.fullSplit1D(pos, data, bins)

#ref = foo.integrate(data)
#assert(numpy.allclose(ref[1],outMerge))

#plot(ref[0],outMerge, label="ocl_lut_merge")
#plot(ref[0],outData, label="ocl_lut_data")
#plot(ref[0],outCount, label="ocl_lut_count")

#plot(out[0], out[1], label="ocl_lut_merge")
#plot(out[0], out[2], label="ocl_lut_data")
#plot(out[0], out[3], label="ocl_lut_count")

#plot(ref[0], ref[1], label="ref_merge")
#plot(ref[0], ref[2], label="ref_data")
#plot(ref[0], ref[3], label="ref_count")
####plot(abs(ref-outMerge)/outMerge, label="ocl_csr_fullsplit")
#legend()
#show()
#raw_input()

  
#aaa = 0
#bbb = 0
#for i in range(bins):
    #ind_tmp1 = numpy.copy(indices[idx_ptr[i]:idx_ptr[i+1]])
    #ind_tmp2 = numpy.copy(foo.indices[idx_ptr[i]:idx_ptr[i+1]])
    #data_tmp1 = numpy.copy(data_lut[idx_ptr[i]:idx_ptr[i+1]])
    #data_tmp2 = numpy.copy(foo.data[idx_ptr[i]:idx_ptr[i+1]])
    #sort1 = numpy.argsort(ind_tmp1)
    #sort2 = numpy.argsort(ind_tmp2)
    #data_1 = data_tmp1[sort1]
    #data_2 = data_tmp2[sort2]
    #for j in range(data_1.size):
        #aaa += 1
        #if not numpy.allclose(data_1[j],data_2[j]):
            #bbb += 1
            #print data_1[j],data_2[j],numpy.allclose(data_1[j],data_2[j]), idx_ptr[i]+j


#print aaa,bbb