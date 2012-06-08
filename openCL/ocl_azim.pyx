# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration 
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "19/05/2012"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
cimport ocl_xrpd1d
from libcpp cimport bool as cpp_bool
cimport numpy
import numpy
import threading

cdef class Integrator1d:
    """
    Simple wrapper for ocl_xrpd1d.ocl_xrpd1D_fullsplit C++ class
    """
    cdef ocl_xrpd1d.ocl_xrpd1D_fullsplit* cpp_integrator
    cdef char* _devicetype
    cdef int _nBins,_nData,_platformid,_devid
    cdef cpp_bool _useFp64
    
    def __cinit__(self, filename=None):
        self._nBins = -1
        self._nData = -1
        self._useFp64 = False
        self._devicetype = "gpu"
        if filename is None:
            self.cpp_integrator = new ocl_xrpd1d.ocl_xrpd1D_fullsplit()
        else:
            name = str(filename) 
            self.cpp_integrator = new ocl_xrpd1d.ocl_xrpd1D_fullsplit(name)
            
    def __dealloc__(self):
        del self.cpp_integrator

    def getConfiguration(self, int Nimage, int Nbins, useFp64=None):
        """getConfiguration gets the description of the integrations to be performed and keeps an internal copy
        @param Nimage: number of pixel in image
        @param Nbins: number of bins in regrouped histogram
        @param useFp64: use double precision. By default the same as init!
        """
        cdef int rc
        if useFp64 is not None:
            self._useFp64 = <cpp_bool> bool(useFp64)
        self._nBins = Nbins
        self._nData = Nimage
        with nogil:
            rc = self.cpp_integrator.getConfiguration(<int> 1024, <int> Nimage, <int> Nbins, <cpp_bool> self._useFp64)
        return rc

    def configure(self, kernel="ocl_azim_kernel_2.cl"):
        """configure is possibly the most crucial method of the class.
        It is responsible of allocating the required memory and compile the OpenCL kernels
        based on the configuration of the integration.
        It also "ties" the OpenCL memory to the kernel arguments.
        If ANY of the arguments of getConfiguration needs to be changed, configure must
        be called again for them to take effect
        
        @param kernel: name or path to the file containing the kernel
        """
        kernel = str(kernel)
        cdef char* ckernel = <char*> kernel
        cdef int rc
        with nogil:
            rc = self.cpp_integrator.configure(ckernel)
        return rc
    
    def loadTth(self, numpy.ndarray tth not None, numpy.ndarray dtth not None, float tth_min, float tth_max):
        """Load the 2th arrays along with the min and max value.
        loadTth maybe be recalled at any time of the execution in order to update
        the 2th arrays.
        
        loadTth is required and must be called at least once after a configure()"""
        cdef numpy.ndarray[numpy.float32_t, ndim = 1]  tthc,dtthc
        if tth.dtype == numpy.float32:
            tthc = numpy.ascontiguousarray(tth.ravel())
        else:
            tthc = numpy.ascontiguousarray(tth.astype(numpy.float32).ravel())
        if dtth.dtype == numpy.float32:
            dtthc = numpy.ascontiguousarray(dtth.ravel())
        else:
            dtthc = numpy.ascontiguousarray(dtth.astype(numpy.float32).ravel())

        return self.cpp_integrator.loadTth(<float*> tthc.data, <float*> dtthc.data, <float> tth_min, <float> tth_max)


    def setSolidAngle(self, numpy.ndarray solidAngle not None):
        """ Enables SolidAngle correction and uploads the suitable array to the OpenCL device.
         By default the program will assume no solidangle correction unless setSolidAngle() is called.
         From then on, all integrations will be corrected via the SolidAngle array.
        
         If the SolidAngle array needs to be changes, one may just call setSolidAngle() again
         with that array
         @param solidAngle: numpy array representing the solid angle of the given pixel
         @return: integer
         """
        cdef numpy.ndarray[numpy.float32_t, ndim = 1]  cSolidAngle
        if solidAngle.dtype == numpy.float32:
            cSolidAngle = numpy.ascontiguousarray(solidAngle.ravel())
        else:
            cSolidAngle = numpy.ascontiguousarray(solidAngle.astype(numpy.float32).ravel())
         
        return self.cpp_integrator.setSolidAngle(<float*> cSolidAngle.data)

    def unsetSolidAngle(self):
        """
        Instructs the program to not perform solidangle correction from now on.
        SolidAngle correction may be turned back on at any point
        @return: integer
        """
        return self.cpp_integrator.unsetSolidAngle()

    def setMask(self, numpy.ndarray mask not None):
        """ 
        Enables the use of a Mask during integration. The Mask can be updated by
        recalling setMask at any point.
        
        The Mask must be a PyFAI Mask. Pixels with 0 are masked out.
        @param mask: numpy.ndarray of integer.
        @return integer 
        """
        cdef numpy.ndarray[numpy.int_t, ndim = 1]  cMask
        if mask.dtype == numpy.int:
            cMask = numpy.ascontiguousarray(mask.ravel())
        else:
            cMask = numpy.ascontiguousarray(mask.astype(numpy.int).ravel())
         
        return self.cpp_integrator.setMask(<int*> cMask.data)

    def unsetMask(self): 
        """
        Disables the use of a Mask from that point.
        It may be re-enabled at any point via setMask
        """
        return self.cpp_integrator.unsetMask()

    def setDummyValue(self, float dummyVal):
        """
        Enables dummy value functionality and uploads the value to the OpenCL device.
        Image values that are similar to the dummy value are set to 0.
        """
        return self.cpp_integrator.setDummyValue(dummyVal)
    
    def unsetDummyValue(self):
        """Disable a dummy value. 
        May be re-enabled at any time by setDummyValue
        """
        return self.cpp_integrator.unsetDummyValue()

    def setRange(self, float lowerBound, float upperBound):
        """Sets the active range to integrate on. By default the range is set to tth_min and tth_max
        By calling this functions, one may change to different bounds
        
        @param lowerBound: usually tth_min
        @param upperBound: usually tth_max
        @return: integer 
        """
        return self.cpp_integrator.setRange(lowerBound, upperBound)

    def unsetRange(self):
        "Resets the 2th integration range back to tth_min, tth_max"
        return self.cpp_integrator.unsetRange()

    def execute(self, numpy.ndarray image not None):
        """Take an image, integrate and return the histogram and weights
        set / unset and loadTth methods have a direct impact on the execute() method.
        All the rest of the methods will require at least a new configuration via configure()"""
        cdef int rc
        cdef numpy.ndarray[numpy.float32_t, ndim = 1] cimage, histogram,bins
        cimage = numpy.ascontiguousarray(image.ravel(),dtype="float32")
        histogram = numpy.zeros(self._nBins,dtype="float32")
        bins = numpy.zeros(self._nBins,dtype="float32")
        assert cimage.size == self._nData
        with nogil:
            rc = self.cpp_integrator.execute(<float*> cimage.data, <float*> histogram.data, <float*> bins.data)
        if rc!=0:
            raise RuntimeError("OpenCL integrator failed with RC=%s"%rc)
        return histogram,bins

    def clean(self, int preserve_context=0):
        """Free OpenCL related resources.
        It may be asked to preserve the context created by init or completely clean up OpenCL.
        
        Guard / Status flags that are set will be reset. All the Operation flags are also reset"""
        return  self.cpp_integrator.clean(preserve_context)

################################################################################
# Methods inherited from ocl_base class 
################################################################################
    def init(self,*args, **kwarg):
        """Initial configuration: Choose a device and initiate a context. Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        Suggested are GPU,CPU. For each setting to work there must be such an OpenCL device and properly installed.
        E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail. The AMD SDK kit is required for CPU via OpenCL.
        @param useFp64: boolean specifying if double precision will be used 
        @param devicetype: string in ["cpu","gpu"]
        @param platformid: integer
        @param devid: integer
        """
        if "useFp64" in kwarg:
            self._useFp64 = <cpp_bool> kwarg["useFp64"]
        if "devicetype"  in kwarg:
            self._devicetype = <char*> kwarg["devicetype"]
        if "platformid" in kwarg:
            self._platformid = <int> kwarg["platformid"]
        if "devid"  in kwarg:
            self._devid = <int> kwarg["devid"]
        ids=[]
        for arg in args:
            if isinstance(arg, bool):
                self._useFp64 = <cpp_bool> arg
            elif isinstance(arg, str):
                self._devicetype = arg
            elif isinstance(arg,int):
                ids.append(arg)
        if len(ids)==2:
            self.cpp_integrator.init(<char*>self._devicetype, <int>ids[0], <int>ids[1], <cpp_bool> self._useFp64)
        else:
            return self.cpp_integrator.init(<char*>self._devicetype, <cpp_bool> self._useFp64)
#    TODO
#        virtual int init(char * devicetype, cpp_bool useFp64=true)
#        virtual int init(char * devicetype, int platformid, int devid, cpp_bool useFp64=true)

    def show_devices(self):
        "Prints a list of OpenCL capable devices, their platforms and their ids"
        self.cpp_integrator.show_devices()


    def print_devices(self):
        """Same as show_devices but displays the results always on stdout even
        if the stream is set to a file"""
        self.cpp_integrator. print_devices()
    
    def  show_device_details(self):
        "Print details of a selected device"
        self.cpp_integrator.show_device_details()

    def reset_time(self):
        'Resets the internal profiling timers to 0'  
        self.cpp_integrator.reset_time()

    def get_exec_time(self):
        "Returns the internal profiling timer for the kernel executions"
        return self.cpp_integrator.get_exec_time()

    def get_exec_count(self):
        "Returns how many integrations have been performed"
        return self.cpp_integrator.get_exec_count()

    def get_memCpy_time(self):
        "Returns the time spent on memory copies"
        return self.cpp_integrator.get_memCpy_time()

_INTEGRATORS_1D={} #key=(Nimage,NBins), value=instance of Integrator1d
lock =threading.Semaphore()


def histGPU1d(numpy.ndarray weights not None,
                numpy.ndarray pos0 not None,
                numpy.ndarray delta_pos0 not None,
                long bins=100,
                pos0Range=None,
                float dummy=0.0,
                mask=None,
                devicetype="all",
                useFp64=True,
                platformid=None,
                deviceid=None,
              ):
    """
    Calculates histogram of pos0 (tth) weighted by weights
    
    Splitting is done on the pixel's bounding box like fit2D
    
    @param weights: array with intensities
    @param pos0: 1D array with pos0: tth or q_vect
    @param delta_pos0: 1D array with delta pos0: max center-corner distance
    @param bins: number of output bins
    @param pos0Range: minimum and maximum  of the 2th range
    @param dummy: value for bins without pixels
    @param deviceType: "cpu" or "gpu" or "all"  or "def"
    @param useFp64: shall histogram be done in double precision (adviced)
    @param platformid: platform number 
    @param deviceid: device number
    @return 2theta, I, weighted histogram, unweighted histogram
    """
    cdef long  size = weights.size
    assert pos0.size == size
    assert delta_pos0.size == size
    assert  bins > 1
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cdata = numpy.ascontiguousarray(weights.ravel(),dtype="float32")
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos0, dpos0
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype="float32")
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype="float32")
    kernel = os.path.join(os.path.dirname(__file__),"ocl_azim_kernel_2.cl")
    cdef float pos0_min,pos0_max,pos0_maxin
    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        if pos0_min < 0.0:
            pos0_min = 0.0
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = max(0,cpos0.min())
        pos0_maxin = cpos0.max()
    pos0_max = pos0_maxin * (1.0 + numpy.finfo(numpy.float32).eps)


    if  (size,bins) not in _INTEGRATORS_1D:
        with lock:
            if  (size,bins) not in _INTEGRATORS_1D:
                integr = Integrator1d()
                if platformid and deviceid:
                    rc = integr.init(devicetype=devicetype,
                                platformid=platformid,
                                devid=deviceid,
                                useFp64=useFp64)
                else:
                    rc = integr.init(devicetype, useFp64)
                if rc!=0:
                    raise RuntimeError('Failed to initialize OpenCL deviceType %s (%s,%s) 64bits: %s'%(devicetype,platformid,deviceid,useFp64))
                
                if 0!= integr.getConfiguration(size, bins):
                    raise RuntimeError('Failed to configure 1D integrator with Ndata=%s and Nbins=%s'%(size,bins))

                if 0!= integr.configure(<char*> kernel):
                    raise RuntimeError('Failed to compile kernel at %s'%(kernel))

                if 0!= integr.loadTth(cpos0, dpos0, pos0_min, pos0_max):
                    raise RuntimeError("Failed to upload 2th arrays")
#if 0!= integr.setSolidAngle(solid)
#a.setMask(mask)
                else:
                    _INTEGRATORS_1D[(size,bins)]=integr
    integr = _INTEGRATORS_1D[(size,bins)]
    return integr.execute(cdata)
    
