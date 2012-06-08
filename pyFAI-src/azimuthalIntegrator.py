#!/usr/bin/env python
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
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "21/12/2011"
__status__ = "beta"

import os, logging
import numpy
from numpy import degrees
from geometry import Geometry
import fabio
from utils import timeit
logger = logging.getLogger("pyFAI.azimuthalIntegrator")



class AzimuthalIntegrator(Geometry):
    """
    This class is an azimuthal integrator based on P. Boesecke's geometry and 
    histogram algorithm by Manolo S. del Rio and V.A Sole
    
    All geometry calculation are done in the Geometry class
    
    """
    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0, pixel1=1, pixel2=1, splineFile=None):
        """
        @param dist: distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        @param poni1: coordinate of the point of normal incidence along the detector's first dimension, in meter
        @param poni2: coordinate of the point of normal incidence along the detector's second dimension, in meter
        @param rot1: first rotation from sample ref to detector's ref, in radians
        @param rot2: second rotation from sample ref to detector's ref, in radians
        @param rot3: third rotation from sample ref to detector's ref, in radians
        @param pixel1: pixel size of the fist dimension of the detector,  in meter
        @param pixel2: pixel size of the second dimension of the detector,  in meter
        @param splineFile: file containing the geometric distortion of the detector. Overrides the pixel size.  
        """
        Geometry.__init__(self, dist, poni1, poni2, rot1, rot2, rot3, pixel1, pixel2, splineFile)
        self._nbPixCache = {} #key=shape, value: array

        self.maskfile = None    #just a placeholder
        self.background = None  #just a placeholder
        self.flatfield = None   #just a placeholder
        self.darkcurrent = None   #just a placeholder
        self.header = None

        self._backgrounds = {}  #dict for caching
        self._flatfields = {}
        self._darkcurrents = {}

    def makeMask(self, data, mask=None, dummy=None, delta_dummy=None, invertMask=None):
        """
        Combines a mask
        
        For the mask: 1 for good pixels, 0 for bas pixels
        @param data: input array of 
        @param mask: input mask 
        @param dummy: value of dead pixels
        @param delta_dumy: precision of dummy pixels
        @param invertMask: to force inversion of the input mask
        """
        shape = data.shape
        if (mask is None) and (self.maskfile is not None) and os.path.isfile(self.maskfile):
            mask = fabio.open(self.maskfile).data
        if (mask is None) and (self.maskfile is None):
            mask = numpy.ones(shape, dtype=bool)
        else:
            mask_min = mask.min()
            mask_max = mask.max()
            if  mask_min < 0 and mask_max == 0:
                mask = (mask.clip(-1, 0) + 1).astype(bool)
            else:
                mask = mask.astype(bool)
            if mask.sum(dtype=int) > mask.size:
                logger.debug("Mask likely to be inverted as more than half pixel are masked !!!")
                mask = (1 - mask).astype(bool)
        if (mask.shape != shape):
            try:
                mask = mask[:shape[0], :shape[1]]
            except Exception as error:#IGNORE:W0703
                logger.error("Mask provided has wrong shape: expected: %s, got %s, error: %s" % (shape, mask.shape, error))
                mask = numpy.ones(shape, dtype=bool)
        if invertMask:
            mask = (1 - mask)
        if dummy is not None:
            if delta_dummy is None:
                mask *= (data != dummy)
            else:
                mask *= abs(data - dummy) > delta_dummy
        return mask.astype(bool)



    def xrpd_numpy(self, data, nbPt, filename=None, correctSolidAngle=True, tthRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the powder diffraction pattern from a set of data, an image. Numpy implementation

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays
        """
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        tth = self.twoThetaArray(data.shape)[mask]

        if nbPt not in self._nbPixCache:
            ref, b = numpy.histogram(tth, nbPt)
            self._nbPixCache[nbPt] = numpy.maximum(1, ref)
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange])
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask].astype("float64")
        else:
            data = data[mask].astype("float64")

        val, b = numpy.histogram(tth,
                                 bins=nbPt,
                                 weights=data,
                                 range=tthRange)
        tthAxis = degrees(b[1:].astype("float32") + b[:-1].astype("float32")) / 2.0
        I = val / self._nbPixCache[nbPt]
        if filename:
            try:
                with open(filename, "w") as openedfile:
                    openedfile.writelines(["%s\t%s%s" % (t, i, os.linesep) for t, i in zip(tthAxis, I)])
            except IOError:
                logger.error("IOError: unable to write to file %s" % filename)
        return tthAxis, I



    def xrpd_cython(self, data, nbPt, filename=None, correctSolidAngle=True, tthRange=None, mask=None, dummy=None, delta_dummy=None, pixelSize=None):
        """
        Calculate the powder diffraction pattern from a set of data, an image. Cython implementation
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of the 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays
        """

        try:
            import histogram #IGNORE:F0401
        except ImportError as error:#IGNORE:W0703
            logger.error("Import error (%s), falling back on old method !" % error)
            return self.xrpd_numpy(data, nbPt, filename, correctSolidAngle, tthRange)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        tth = self.twoThetaArray(data.shape)[mask]
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange])
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        if dummy is None:
            dummy = 0.0
        tthAxis, I, a, b = histogram.histogram(pos=tth,
                                                   weights=data,
                                                   bins=nbPt,
                                                   bin_range=tthRange,
                                                   pixelSize_in_Pos=pixelSize,
                                                   dummy=dummy)
        tthAxis = numpy.degrees(tthAxis)
        if filename:
            open(filename, "w").writelines(["%s\t%s%s" % (t, i, os.linesep) for t, i in zip(tthAxis, I)])
        return tthAxis, I

    @timeit
    def xrpd_splitBBox(self, data, nbPt, filename=None, correctSolidAngle=True,
                       tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the powder diffraction pattern from a set of data, an image. Cython implementation
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of the 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type chiRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays
        """
        try:
            import splitBBox  #IGNORE:F0401
        except ImportError as error:#IGNORE:W0703
            logger.error("Import error (%s), falling back on old method !" % error)
            return self.xrpd_numpy(data, nbPt, filename, correctSolidAngle, tthRange)
        shape = data.shape
        if chiRange is not None:
            chi = self.chiArray(shape)[mask]
            dchi = self.deltaChi(shape)[mask]
        else:
            chi = None
            dchi = None
        tth = self.twoThetaArray(data.shape)
        dtth = self.delta2Theta(data.shape)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        if dummy is None:
            dummy = 0.0
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange[:2]])
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange[:2]])
        if correctSolidAngle: #outPos, outMerge, outData, outCount
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        tthAxis, I, a, b = splitBBox.histoBBox1d(weights=data,
                                                 pos0=tth[mask],
                                                 delta_pos0=dtth[mask],
                                                 pos1=chi,
                                                 delta_pos1=dchi,
                                                 bins=nbPt,
                                                 pos0Range=tthRange,
                                                 pos1Range=chiRange,
                                                 dummy=dummy)
        tthAxis = numpy.degrees(tthAxis)
        if filename:
            open(filename, "w").writelines(["%s\t%s%s" % (t, i, os.linesep) for t, i in zip(tthAxis, I)])
        return tthAxis, I


    def xrpd_splitPixel(self, data, nbPt, filename=None, correctSolidAngle=True,
                        tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the powder diffraction pattern from a set of data, an image. 
        
        Cython implementation
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in
        @type filename: string
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays

        """
        try:
            import splitPixel#IGNORE:F0401
        except ImportError as error:
            logger.error("Import error %s , falling back on simple histogram !" % error)
            return self.xrpd_cython(data, nbPt, filename, correctSolidAngle, tthRange, mask, dummy, delta_dummy)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        pos = self.cornerArray(data.shape)[mask]
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data [mask]
        if dummy is None:
            dummy = 0.0
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange])
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange])


        tthAxis, I, a, b = splitPixel.fullSplit1D(pos=pos,
                                                  weights=data,
                                                  bins=nbPt,
                                                  pos0Range=tthRange,
                                                  pos1Range=chiRange,
                                                  dummy=dummy)
        tthAxis = numpy.degrees(tthAxis)
        if filename:
            open(filename, "w").writelines(["%s\t%s%s" % (t, i, os.linesep) for t, i in zip(tthAxis, I)])
        return tthAxis, I
    #Default implementation:
    xrpd = xrpd_splitBBox

    @timeit
    def xrpd_OpenCL(self, data, nbPt, filename=None, correctSolidAngle=True,
                       tthRange=None, mask=None, dummy=None, delta_dummy=None,
                        devicetype="all", useFp64=True, platformid=None, deviceid=None):

        """
        Calculate the powder diffraction pattern from a set of data, an image. Cython implementation
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of the 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type chiRange: (float, float), optional, disabled for now
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        OpenCL specific parameters:
        @param deviceType: "cpu" or "gpu" or "all"  or "def"
        @param useFp64: shall histogram be done in double precision (adviced)
        @param platformid: platform number 
        @param deviceid: device number
        @return 2theta, I, weighted histogram, unweighted histogram


        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays
        """
        try:
            import ocl_azim  #IGNORE:F0401
        except ImportError as error:#IGNORE:W0703
            logger.error("Import error (%s), falling back on old method !" % error)
            return self.xrpd_splitBBox(data=data,
                                       nbPt=nbPt,
                                       filename=filename,
                                       correctSolidAngle=correctSolidAngle,
                                       tthRange=tthRange,
                                       mask=mask,
                                       dummy=dummy,
                                       delta_dummy=delta_dummy,
                                       devicetype=devicetype,
                                       useFp64=useFp64,
                                       platformid=platformid,
                                       deviceid=deviceid)
        shape = data.shape
        tth = self.twoThetaArray(data.shape)
        dtth = self.delta2Theta(data.shape)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        if dummy is None:
            dummy = 0.0
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange[:2]])
#        if chiRange is not None:
#            chiRange = tuple([numpy.deg2rad(i) for i in chiRange[:2]])
        if correctSolidAngle: #outPos, outMerge, outData, outCount
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        tthAxis, I, a = ocl_azim.histGPU1d(weights=data,
                                                 pos0=tth[mask],
                                                 delta_pos0=dtth[mask],
                                                 bins=nbPt,
                                                 pos0Range=tthRange,
                                                 dummy=dummy)
        tthAxis = numpy.degrees(tthAxis)
        if filename:
            open(filename, "w").writelines(["%s\t%s%s" % (t, i, os.linesep) for t, i in zip(tthAxis, I)])
        return tthAxis, I


    def xrpd2_numpy(self, data, nbPt2Th, nbPtChi=360, filename=None, correctSolidAngle=True,
                         tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from a set of data, an image
        
        Pure numpy implementation (VERY SLOW !!!)
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of points in the output pattern in the Radial (horizontal) axis (2 theta)
        @type nbPt: integer
        @param nbPtChi: number of points in the output pattern along the Azimuthal (vertical) axis (chi) 
        @type nbPtChi: integer
        @param filename: file to save data in
        @type filename: string
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: azimuthaly regrouped data, 2theta pos and chipos
        @rtype: 3-tuple of ndarrays
        """
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        tth = self.twoThetaArray(data.shape)[mask]
        chi = self.chiArray(data.shape)[mask]
        bins = (nbPtChi, nbPt2Th)
        if bins not in self._nbPixCache:
            ref, binsChi, bins2Th = numpy.histogram2d(chi, tth, bins=list(bins))
            self._nbPixCache[bins] = numpy.maximum(1.0, ref)
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        if tthRange is not None:
            tthRange = [numpy.deg2rad(i) for i in tthRange]
        else:
            tthRange = [numpy.deg2rad(tth.min()), numpy.deg2rad(tth.max())]
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange])
        else:
            chiRange = tuple([numpy.deg2rad(chi.min()), numpy.deg2rad(chi.max())])

        val, binsChi, bins2Th = numpy.histogram2d(chi, tth,
                                                  bins=list(bins),
                                                  weights=data,
                                                  range=[chiRange, tthRange])
        I = val / self._nbPixCache[bins]
        return I, bins2Th, binsChi

    def xrpd2_histogram(self, data, nbPt2Th, nbPtChi=360, filename=None, correctSolidAngle=True,
                              tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from a set of data, an image
        
        Cython implementation: fast but incaccurate
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of points in the output pattern in the Radial (horizontal) axis (2 theta)
        @type nbPt: integer
        @param nbPtChi: number of points in the output pattern along the Azimuthal (vertical) axis (chi) 
        @type nbPtChi: integer
        @param filename: file to save data in
        @type filename: string
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: azimuthaly regrouped data, 2theta pos and chipos
        @rtype: 3-tuple of ndarrays
        """

        try:
            import histogram#IGNORE:F0401
        except ImportError as error:
            logger.error("Import error %s , falling back on numpy histogram !" % error)
            return self.xrpd2_numpy(data=data, nbPt2Th=nbPt2Th, nbPtChi=nbPtChi,
                                    filename=filename, correctSolidAngle=correctSolidAngle,
                                    tthRange=tthRange, chiRange=chiRange, mask=mask, dummy=dummy,
                                    delta_dummy=delta_dummy)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        tth = self.twoThetaArray(data.shape)[mask]
        chi = self.chiArray(data.shape)[mask]
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        if dummy is None:
            dummy = 0.0
        I, binsChi, bins2Th, val, count = histogram.histogram2d(pos0=chi,
                                                                pos1=tth,
                                                                bins=(nbPtChi, nbPt2Th),
                                                                weights=data,
                                                                split=1,
                                                                dummy=dummy)
        bins2Th = numpy.degrees(bins2Th)
        binsChi = numpy.degrees(binsChi)
        if filename:
            self.save2D(filename, I, bins2Th, binsChi)
        return I, bins2Th, binsChi



    def xrpd2_splitBBox(self, data, nbPt2Th, nbPtChi=360, filename=None, correctSolidAngle=True,
                         tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from a set of data, an image
        
        Split pixels according to their coordinate and a bounding box
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of points in the output pattern in the Radial (horizontal) axis (2 theta)
        @type nbPt: integer
        @param nbPtChi: number of points in the output pattern along the Azimuthal (vertical) axis (chi) 
        @type nbPtChi: integer
        @param filename: file to save data in
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of the 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the azimuthal angle. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type chiRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays
        """
        try:
            import splitBBox#IGNORE:F0401
        except ImportError as error:
            logger.error("Import error %s , falling back on simple histogram !" % error)
            return self.xrpd2_histogram(data=data, nbPt2Th=nbPt2Th, nbPtChi=nbPtChi,
                                        filename=filename, correctSolidAngle=correctSolidAngle,
                                        tthRange=tthRange, chiRange=chiRange, mask=mask, dummy=dummy, delta_dummy=delta_dummy)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        tth = self.twoThetaArray(data.shape)[mask]
        chi = self.chiArray(data.shape)[mask]
        dtth = self.delta2Theta(data.shape)[mask]
        dchi = self.deltaChi(data.shape)[mask]
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange])
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange])
        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        if dummy is None:
            dummy = 0.0
        I, bins2Th, binsChi, a, b = splitBBox.histoBBox2d(weights=data,
                                                          pos0=tth,
                                                          delta_pos0=dtth,
                                                          pos1=chi,
                                                          delta_pos1=dchi,
                                                          bins=(nbPt2Th, nbPtChi),
                                                          pos0Range=tthRange,
                                                          pos1Range=chiRange,
                                                          dummy=dummy)
        bins2Th = numpy.degrees(bins2Th)
        binsChi = numpy.degrees(binsChi)
        if filename:
            self.save2D(filename, I, bins2Th, binsChi)
        return I, bins2Th, binsChi

    def xrpd2_splitPixel(self, data, nbPt2Th, nbPtChi=360, filename=None, correctSolidAngle=True,
                         tthRange=None, chiRange=None, mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from a set of data, an image
        
        Split pixels according to their corner positions
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of points in the output pattern in the Radial (horizontal) axis (2 theta)
        @type nbPt: integer
        @param nbPtChi: number of points in the output pattern along the Azimuthal (vertical) axis (chi) 
        @type nbPtChi: integer
        @param filename: file to save data in
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean
        @param tthRange: The lower and upper range of the 2theta. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the azimuthal angle. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type chiRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays
        """
        try:
            import splitPixel#IGNORE:F0401
        except ImportError as error:
            logger.error("Import error %s , falling back on SplitBBox !" % error)
            return self.xrpd2_splitBBox(data=data, nbPt2Th=nbPt2Th, nbPtChi=nbPtChi,
                                        filename=filename, correctSolidAngle=correctSolidAngle,
                                        tthRange=tthRange, chiRange=chiRange, mask=mask, dummy=dummy, delta_dummy=delta_dummy)
        mask = self.makeMask(data, mask, dummy, delta_dummy)
        pos = self.cornerArray(data.shape)[mask]
        if tthRange is not None:
            tthRange = tuple([numpy.deg2rad(i) for i in tthRange])
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange])

        if correctSolidAngle:
            data = (data / self.solidAngleArray(data.shape))[mask]
        else:
            data = data[mask]
        if dummy is None:
            dummy = 0
        I, bins2Th, binsChi, a, b = splitPixel.fullSplit2D(pos=pos,
                                                           weights=data,
                                                           bins=(nbPt2Th, nbPtChi),
                                                           pos0Range=tthRange,
                                                           pos1Range=chiRange,
                                                           dummy=dummy)
        bins2Th = numpy.degrees(bins2Th)
        binsChi = numpy.degrees(binsChi)
        if filename:
            self.save2D(filename, I, bins2Th, binsChi)
        return I, bins2Th, binsChi
    xrpd2 = xrpd2_splitBBox




    def save2D(self, filename, I, dim1, dim2, dim1_unit="2th"):
        header = {"dist":str(self._dist),
                  "poni1": str(self._poni1),
                  "poni2": str(self._poni2),
                  "rot1": str(self._rot1),
                  "rot2": str(self._rot2),
                  "rot3": str(self._rot3),
                  "chi_min":str(dim2.min()),
                  "chi_max":str(dim2.max()),
                  dim1_unit + "_min":str(dim1.min()),
                  dim1_unit + "_max":str(dim1.max()),
                  "pixelX": str(self.pixel2), #this is not a bug ... most people expect dim1 to be X
                  "pixelY": str(self.pixel1), #this is not a bug ... most people expect dim2 to be Y
                }
        if self.splineFile:
            header["spline"] = str(self.splineFile)
        f2d = self.getFit2D()
        for key in f2d:
            header["key"] = f2d[key]
        try:
            fabio.edfimage.edfimage(data=I.astype("float32"), header=header).write(filename)
        except IOError:
            logger.error("IOError while writing %s" % filename)



    def saxs(self, data, nbPt, filename=None, correctSolidAngle=True, variance=None,
             qRange=None, chiRange=None, mask=None, dummy=None,
             delta_dummy=None, method="bbox"):
        """
        Calculate the azimuthal integrated Saxs curve  
        
        Multi algorithm implementation (tries to be bullet proof)
        
        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data to
        @type filename: string
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: boolean     
        @param variance: array containing the variance of the data
        @type variance: ndarray
        @param qRange: The lower and upper range of the sctter vector q. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type qRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle. If not provided, range is simply (data.min(), data.max()).
                        Values outside the range are ignored. 
        @type chiRange: (float, float), optional
        @param mask: array (same siza as image) with 0 for masked pixels, and 1 for valid pixels
        @param  dummy: value for dead/masked pixels 
        @param delta_dummy: precision for dummy value
        @param method: can be "numpy", "cython", "BBox" or "splitpixel" 
        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays
        """
        method = method.lower()
        if variance is not None:
            assert variance.shape == data.shape
        mask = self.makeMask(data, mask)
        shape = data.shape
        data = data.astype("float32")
        q = self.qArray(shape)
        if chiRange is not None:
            chiRange = tuple([numpy.deg2rad(i) for i in chiRange])
            chi = self.chiArray(shape)
        else:
            chi = None
        if self.darkcurrent is not None:
            if os.path.isfile(self.darkcurrent):
                dcf = os.path.abspath(self.darkcurrent)
                if dcf not  in self._darkcurrents:
                    self._darkcurrents[dcf] = fabio.open(dcf).data.astype("float32")
                data -= self._darkcurrents[dcf]
                mask = mask * (data >= 0) #invalidate data < dark current
        if self.flatfield is not None:
            if os.path.isfile(self.flatfield):
                fff = os.path.abspath(self.flatfield)
                if fff not  in self._flatfields:
                    self._flatfields[fff] = fabio.open(fff).data.astype("float32")
                data /= self._flatfields[fff]
                mask = mask * (self._flatfields[fff] > 0) #invalidate flatfield <=0
        if self.background is not None:
            if os.path.isfile(self.background):
                bgf = os.path.abspath(self.background)
                if bgf not  in self._backgrounds:
                    self._backgrounds[bgf] = fabio.open(bgf).data.astype("float32")
                data -= self._backgrounds[bgf]
                mask = mask * (data >= 0) #invalidate data < background

        if correctSolidAngle:
            data = (data / self.solidAngleArray(shape))
        I = None
        sigma = None
        if method.lower() == "splitpixel":
            logger.debug("saxs uses SplitPixel implementation")
            try:
                import splitPixel#IGNORE:F0401
            except ImportError as error:
                logger.error("Import error %s , falling back on splitbbox histogram !" % error)
                method = "bbox"
            else:
                pos = self.cornerQArray(shape)
                data = data[mask]
                pos = pos[mask]
                if variance is not None:
                    variance = variance[mask]
                qAxis, I, a, b = splitPixel.fullSplit1D(pos=pos,
                                                        weights=data,
                                                        bins=nbPt,
                                                        pos0Range=qRange,
                                                        pos1Range=chiRange,
                                                        dummy=dummy,
                                                        delta_dummy=delta_dummy)
                if variance is not None:
                    qa, var1d, a, b = splitPixel.fullSplit1D(pos=pos,
                                                            weights=variance,
                                                            bins=nbPt,
                                                            pos0Range=qRange,
                                                            pos1Range=chiRange,
                                                            dummy=dummy,
                                                            delta_dummy=delta_dummy)
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if method.lower() == "bbox":
            logger.debug("saxs uses BBox implementation")
            try:
                import splitBBox#IGNORE:F0401
            except ImportError as error:
                logger.error("Import error %s , falling back on Cython histogram !" % error)
                method = "cython"
            else:
                mask = self.makeMask(data, mask, dummy, delta_dummy)
                dq = self.deltaQ(shape)[mask]
                q = q[mask]
                if chi is not None:
                    chi = chi[mask]
                    dchi = self.deltaChi(shape)[mask]
                else:
                    dchi = None
                if variance is not None:
                    variance = variance[mask]
                data = data[mask]
                if dummy is None:
                    dummy = 0
                qAxis, I, a, b = splitBBox.histoBBox1d(weights=data,
                                                      pos0=q,
                                                      delta_pos0=dq,
                                                      pos1=chi,
                                                      delta_pos1=dchi,
                                                      bins=nbPt,
                                                      pos0Range=qRange,
                                                      pos1Range=chiRange,
                                                      dummy=dummy)
                if variance is not None:
                    qa, var1d, a, b = splitBBox.histoBBox1d(weights=variance,
                                                      pos0=q,
                                                      delta_pos0=dq,
                                                      pos1=chi,
                                                      delta_pos1=dchi,
                                                      bins=nbPt,
                                                      pos0Range=qRange,
                                                      pos1Range=chiRange,
                                                      dummy=dummy)
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if (I is None) and (method == "cython"):
            logger.debug("saxs uses cython implementation")
            try:
                import histogram#IGNORE:F0401
            except ImportError as error:
                logger.error("Import error %s , falling back on Numpy histogram !", error)
                method = "numpy"
            else:
                mask = self.makeMask(data, mask, dummy, delta_dummy)
                if qRange is not None:
                    qMin, qMax = qRange
                    mask *= (q >= qMin) * (q <= qMax)
                if chiRange is not None:
                    chiMin, chiMax = chiRange
                    chi = self.chiArray(shape)
                    mask *= (chi >= chiMin) * (chi <= chiMax)
                q = q[mask]
                if variance is not None:
                    variance = variance[mask]
                data = data[mask]
                if dummy is None:
                    dummy = 0
                qAxis, I, a, b = histogram.histogram(pos=q,
                                                   weights=data,
                                                   bins=nbPt,
                                                   pixelSize_in_Pos=1,
                                                   dummy=dummy)
                if variance is not None:
                    qa, var1d, a, b = histogram.histogram(pos=q,
                                                   weights=variance,
                                                   bins=nbPt,
                                                   pixelSize_in_Pos=1,
                                                   dummy=dummy)
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if (I is None) :
            logger.debug("saxs uses Numpy implementation")
            mask = self.makeMask(data, mask, dummy, delta_dummy)
            data = data[mask]
            q = q[mask]
            ref, b = numpy.histogram(q, nbPt)
            count = numpy.maximum(1, ref)
            val, b = numpy.histogram(q, nbPt, weights=data)

            if variance is not None:
                variance = variance[mask]
                var1d, b = numpy.histogram(q, nbPt, weights=variance)
                sigma = numpy.sqrt(var1d) / count
            qAxis = (b[1:] + b[:-1]) / 2.0
            I = val / count

        if filename:
            with open(filename, "w") as f:
                f.write(self.makeHeaders())
                f.write("# --> %s%s" % (filename, os.linesep))
                if variance is None:
                    f.write("#%14s %14s %s" % ("q_nm^-1 ", "I ", os.linesep))
                    for t, i in zip(qAxis, I):
                        f.write("%14.6e  %14.6e %s" % (t, i, os.linesep))
                else:
                    f.write("#%14s  %14s  %14s%s" % ("q_nm^-1 ", "I ", "sigma ", os.linesep))
                    for t, i, s in zip(qAxis, I, sigma):
                        f.write("%14.6e  %14.6e  %14.6e %s" % (t, i, s, os.linesep))
        return qAxis, I, sigma

    def makeHeaders(self, hdr="#"):
        """
        @return: a string to be used for headers
        """
        if self.header is None:
            headerLst = ["== pyFAI calibration =="]
            headerLst.append("SplineFile: %s" % self.spline)
            headerLst.append("PixelSize: %.3e, %.3e m" % (self.pixel1, self.pixel2))
            headerLst.append("PONI: %.3e, %.3e m" % (self.poni1, self.poni2))
            headerLst.append("Distance Sample to Detector: %s m" % self.dist)
            headerLst.append("Rotations: %.6f %.6f %.6f rad" % (self.rot1, self.rot2, self.rot3))
            headerLst += ["", "== Fit2d calibration =="]
            f2d = self.getFit2D()
            headerLst.append("Distance Sample-beamCenter: %.3f mm" % f2d["DirectBeamDist"])
            headerLst.append("Center: x=%.3f, y=%.3f pix" % (f2d["BeamCenterX"], f2d["BeamCenterY"]))
            headerLst.append("Tilt: %.3f deg  TiltPlanRot: %.3f deg" % (f2d["Tilt"], f2d["TiltPlanRot"]))
            headerLst.append("")
            if self.wavelength is not None:
                headerLst.append("Wavelength: %s" % self.wavelength)
            if self.maskfile is not None:
                headerLst.append("Mask File: %s" % self.maskfile)
            if self.darkcurrent is not None:
                headerLst.append("DarkCurrent File: %s" % self.darkcurrent)
            if self.flatfield is not None:
                headerLst.append("Flatfield File: %s" % self.flatfield)
            if self.background is not None:
                headerLst.append("Background File: %s" % self.background)
            self.header = os.linesep.join([hdr + " " + i for i in headerLst])
        return self.header
