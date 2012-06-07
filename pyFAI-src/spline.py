#!/usr/bin/env python
# -*- coding: UTF8 -*-
###########################################################################
# Written 2009-12-22 by Jérôme Kieffer 
# Copyright (C) 2009 European Synchrotron Radiation Facility
#                       Grenoble, France
#
#    Principal authors: Jérôme Kieffer  (jerome.kieffer@esrf.fr)
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
########################################################################################

""" This is piece of software aims to manipulate spline files for 
geometric corrections of the 2D detectors  using cubic-spline"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import os, time, sys
import numpy, scipy, Image, fabio
import scipy.optimize
import scipy.interpolate
import scipy.interpolate.fitpack



class Spline:
    """This class is a python representation of the spline file 
    Those file represent cubic splines for 2D detector distortions and makes heavy use of 
    fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
    FITPACK is a collection of FORTRAN programs for curve and surface fitting with splines and tensor product splines.
    See
    http://www.cs.kuleuven.ac.be/cwis/research/nalag/research/topics/fitpack.html
    or
    http://www.netlib.org/dierckx/index.html
    """


    def __init__(self, filename=None):
        """this is the constructor of the Spline class, for"""
        self.splineOrder = 3  #This is the default, so cubic splines
        self.lenStrFloat = 14 # by default one float is 14 char in ascii
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.xDispArray = None
        self.yDispArray = None
        self.xSplineKnotsX = []
        self.xSplineKnotsY = []
        self.xSplineCoeff = []
        self.ySplineKnotsX = []
        self.ySplineKnotsY = []
        self.ySplineCoeff = []
        self.pixelSize = None
        self.grid = None
        self.filename = None
        if filename is not None:
            self.read(filename)



    def __repr__(self):
        txt = "Array size: x= %s - %s\ty= %s - %s" % (self.xmin, self.xmax, self.ymin, self.ymax)
        txt += "\nPixel size = %s microns, Grid spacing = %s" % (self.pixelSize, self.grid)
        txt += "\nX-Displacement spline %i X_knots, %i Y_knots and %i coef: should be (X_knot-1-X_order)*(Y_knot-1-Y_order)" \
                    % (len(self.xSplineKnotsX), len(self.xSplineKnotsY), len(self.xSplineCoeff))
        txt += "\nY-Displacement spline %i X_knots, %i Y_knots and %i coef: should be (X_knot-1-X_order)*(Y_knot-1-Y_order)" \
                    % (len(self.ySplineKnotsX), len(self.ySplineKnotsY), len(self.ySplineCoeff))
        return txt


    def zeros(self, xmin=0.0, ymin=0.0, xmax=2048.0, ymax=2048.0, pixSize=None):
        """defines a spline file with no ( zero ) displacement.
        @type xmin: float
        @type xmax: float
        @type ymax: float
        @type ymin: float
        @param xmin: minimum coordinate in x, usually zero
        @param xmax: maximum coordinate in x (+1) usually 2048
        @param ymin: minimum coordinate in y, usually zero
        @param ymax: maximum coordinate y (+1) usually 2048
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.xDispArray = numpy.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
        self.yDispArray = numpy.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
        if pixSize:
            self.pixelSize = pixSize


    def zeros_like(self, other):
        """defines a spline file with no ( zero ) displacement with the same shape as the other one given.
        @param other: another Spline
        @type other: Spline
        """
        self.zeros(self, other.xmin, other.ymin, other.xmax, other.ymax)


    def read(self, filename):
        """read an ascii spline file from file
        @param filename: name of the file containing the cubic spline distortion file
        @type filename: string
        """
        if not os.path.isfile(filename):
            raise IOError("File does not exist %s" % filename)
        self.filename = filename
        stringSpline = [ i.rstrip() for i in open (filename).readlines() ]
        indexLine = 0
        for oneLine in stringSpline:
            stripedLine = oneLine.strip().upper()
            if stripedLine == "VALID REGION":
                data = stringSpline[ indexLine + 1 ]
                self.xmin = float(data[self.lenStrFloat * 0:self.lenStrFloat * 1])
                self.ymin = float(data[self.lenStrFloat * 1:self.lenStrFloat * 2])
                self.xmax = float(data[self.lenStrFloat * 2:self.lenStrFloat * 3])
                self.ymax = float(data[self.lenStrFloat * 3:self.lenStrFloat * 4])
            elif stripedLine == "GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE":
                data = stringSpline[ indexLine + 1 ]
                self.grid = float(data[:self.lenStrFloat])
                self.pixelSize = (float(data[self.lenStrFloat:self.lenStrFloat * 2]), float(data[self.lenStrFloat * 2:self.lenStrFloat * 3]))
            elif stripedLine == "X-DISTORTION":
                data = stringSpline[ indexLine + 1 ]
                [splineKnotsXLen, splineKnotsYLen] = [ int(i)  for i in data.split() ]
                databloc = []
                for line in  stringSpline[ indexLine + 2 : ]:
                    if len(line) > 0 :
                        for i in range(len(line) / self.lenStrFloat):
                            databloc.append(float(line[i * self.lenStrFloat: (i + 1) * self.lenStrFloat ]))
                    else:
                        break
                self.xSplineKnotsX = databloc[ : splineKnotsXLen ]
                self.xSplineKnotsY = databloc[splineKnotsXLen: splineKnotsXLen + splineKnotsYLen]
                self.xSplineCoeff = databloc[ splineKnotsXLen + splineKnotsYLen: ]
            elif stripedLine == "Y-DISTORTION":
                data = stringSpline[ indexLine + 1 ]
                [splineKnotsXLen, splineKnotsYLen] = [ int(i)  for i in data.split() ]
                databloc = []
                for line in  stringSpline[ indexLine + 2 : ]:
                    if len(line) > 0 :
                        for i in range(len(line) / self.lenStrFloat):
                            databloc.append(float(line[i * self.lenStrFloat: (i + 1) * self.lenStrFloat ]))
                    else:
                        break
                self.ySplineKnotsX = databloc[ : splineKnotsXLen ]
                self.ySplineKnotsY = databloc[splineKnotsXLen: splineKnotsXLen + splineKnotsYLen]
                self.ySplineCoeff = databloc[ splineKnotsXLen + splineKnotsYLen: ]
# Keep this at the end
            indexLine += 1


    def comparison(self, ref, verbose=False):
        """Compares the current spline distortion with a reference
        @param ref: another spline file
        @return: True or False depending if the splines are the same or not """
        self.spline2array()
        ref.spline2array()
        deltax = (self.xDispArray - ref.xDispArray)
        deltay = (self.yDispArray - ref.yDispArray)
        histX = numpy.histogram(deltax.reshape(deltax.size), bins=100)
        histY = numpy.histogram(deltay.reshape(deltay.size), bins=100)
        histXdr = (histX[1][1:] + histX[1][:-1]) / 2.0
        histYdr = (histY[1][1:] + histY[1][:-1]) / 2.0
        histXmax = histXdr [histX[0].argmax()]
        histYmax = histYdr [histY[0].argmax()]
        maxErrX = abs(deltax).max()
        maxErrY = abs(deltay).max()
        curvX = scipy.interpolate.interp1d(histXdr, histX[0] - histX[0].max() / 2.0)
        curvY = scipy.interpolate.interp1d(histYdr, histY[0] - histY[0].max() / 2.0)
        fFWHM_X = scipy.optimize.bisect(curvX , histXmax, histXdr[-1]) - scipy.optimize.bisect(curvX , histXdr[0], histXmax)
        fFWHM_Y = scipy.optimize.bisect(curvY , histYmax, histYdr[-1]) - scipy.optimize.bisect(curvY , histYdr[0], histYmax)
        print ("Analysis of the difference between two splines")
        print ("Maximum error in X= %.3f pixels,\t in Y= %.3f pixels." % (maxErrX, maxErrY))
        print ("Maximum of histogram in X= %.3f pixels,\t in Y= %.3f pixels." % (histXmax, histYmax))
        print ("Mean of histogram in X= %.3f pixels,\t in Y= %.3f pixels." % (deltax.mean(), deltay.mean()))
        print ("FWHM in X= %.3f pixels,\t in Y= %.3f pixels." % (fFWHM_X, fFWHM_Y))

        if verbose:
            import pylab
            pylab.plot(histXdr , histX[0], label="error in X")
            pylab.plot(histYdr, histY[0], label="error in Y")
            pylab.legend()
            pylab.show()
        return (fFWHM_X < 0.05) and (fFWHM_Y < 0.05) and (maxErrX < 0.5) and (maxErrY < 0.5) \
                and (deltax.mean() < 0.01) and(deltay.mean() < 0.01) and (histXmax < 0.01) and (histYmax < 0.01)


    def spline2array(self, timing=False):
        """calculates the displacement matrix using fitpack
         bisplev(x, y, tck, dx = 0, dy = 0)

         Evaluate a bivariate B-spline and its derivatives.
         Return a rank-2 array of spline function values (or spline derivative
         values) at points given by the cross-product of the rank-1 arrays x and y.
         In special cases, return an array or just a float if either x or y or
         both are floats.
        """
        if  (self.xDispArray == None) :
            x_1d_array = numpy.arange(self.xmin, self.xmax + 1)
            y_1d_array = numpy.arange(self.ymin, self.ymax + 1)
            startTime = time.time()
            self.xDispArray = scipy.interpolate.fitpack.bisplev(x_1d_array, y_1d_array, \
                                               [self.xSplineKnotsX, self.xSplineKnotsY, self.xSplineCoeff, self.splineOrder, self.splineOrder ], \
                                               dx=0, dy=0).transpose()
            intermediateTime = time.time()
            self.yDispArray = scipy.interpolate.fitpack.bisplev(x_1d_array, y_1d_array, \
                                               [self.ySplineKnotsX, self.ySplineKnotsY, self.ySplineCoeff, self.splineOrder, self.splineOrder ], \
                                               dx=0, dy=0).transpose()
            if timing:
                print "Timing for: X-Displacement spline evaluation: %.3f sec, Y-Displacement Spline evaluation:  %.3f sec." % \
                        ((intermediateTime - startTime), (time.time() - intermediateTime))

    def splineFuncX(self, x, y):
        """ calculates the displacement matrix using fitpack for the X direction 
        @param x: numpy array repesenting the points in the x direction
        @param y: numpy array repesenting the points in the y direction
        @return: displacement matrix for the X direction
        @rtype: numpy arrays
        """
        if x.ndim == 2:
            if abs(x[1:, :] - x[:-1, :] - numpy.zeros((x.shape[0] - 1, x.shape[1]))).max() < 1e-6:
                x = x[0]
                y = y[:, 0]
            elif abs(x[:, 1:] - x[:, :-1] - numpy.zeros((x.shape[0], x.shape[1] - 1))).max() < 1e-6:
                x = x[:, 0]
                y = y[0]
        xDispArray = scipy.interpolate.fitpack.bisplev(x, y, \
                                               [self.xSplineKnotsX, self.xSplineKnotsY, self.xSplineCoeff, self.splineOrder, self.splineOrder ], \
                                               dx=0, dy=0).transpose()
        return xDispArray

    def splineFuncY(self, x, y):
        """ calculates the displacement matrix using fitpack for the Y direction 
        @param x: numpy array repesenting the points in the x direction
        @param y: numpy array repesenting the points in the y direction
        @return: displacement matrix for the Y direction
        @rtype: numpy array
        """
        if x.ndim == 2:
            if abs(x[1:, :] - x[:-1, :] - numpy.zeros((x.shape[0] - 1, x.shape[1]))).max() < 1e-6:
                x = x[0]
                y = y[:, 0]
            elif abs(x[:, 1:] - x[:, :-1] - numpy.zeros((x.shape[0], x.shape[1] - 1))).max() < 1e-6:
                x = x[:, 0]
                y = y[0]

        yDispArray = scipy.interpolate.fitpack.bisplev(x, y, \
                                               [self.ySplineKnotsX, self.ySplineKnotsY, self.ySplineCoeff, self.splineOrder, self.splineOrder ], \
                                               dx=0, dy=0).transpose()
        return yDispArray


    def array2spline(self, smoothing=1000, timing=False):
        """calculates the spline coefficents from the displacements matrix using fitpack
       """
        self.xmin = 0.0
        self.ymin = 0.0
        self.xmax = float(self.xDispArray.shape[0] - 1)
        self.ymax = float(self.yDispArray.shape[1] - 1)
        if timing:
            startTime = time.time()
        xRectBivariateSpline = scipy.interpolate.fitpack2.RectBivariateSpline(numpy.arange(self.xmax + 1.0), numpy.arange(self.ymax + 1), self.xDispArray.transpose(), s=smoothing)
        if timing:
            intermediateTime = time.time()
        yRectBivariateSpline = scipy.interpolate.fitpack2.RectBivariateSpline(numpy.arange(self.xmax + 1.0), numpy.arange(self.ymax + 1), self.yDispArray.transpose(), s=smoothing)
        if timing:
            print "X-Displ evaluation= %.3f sec, Y-Displ evaluation=  %.3f sec." % (intermediateTime - startTime, time.time() - intermediateTime)
        print len(xRectBivariateSpline.get_coeffs()), "x-coefs", xRectBivariateSpline.get_coeffs()
        print len(yRectBivariateSpline.get_coeffs()), "y-coefs", yRectBivariateSpline.get_coeffs()
        print len(xRectBivariateSpline.get_knots()[0]), len(xRectBivariateSpline.get_knots()[1]), "x-knots", xRectBivariateSpline.get_knots()
        print len(yRectBivariateSpline.get_knots()[0]), len(yRectBivariateSpline.get_knots()[1]), "y-knots", yRectBivariateSpline.get_knots()
        print "Residual x,y", xRectBivariateSpline.get_residual(), yRectBivariateSpline.get_residual()
        self.xSplineKnotsX = xRectBivariateSpline.get_knots()[0]
        self.xSplineKnotsY = xRectBivariateSpline.get_knots()[1]
        self.xSplineCoeff = xRectBivariateSpline.get_coeffs()
        self.ySplineKnotsX = yRectBivariateSpline.get_knots()[0]
        self.ySplineKnotsY = yRectBivariateSpline.get_knots()[1]
        self.ySplineCoeff = yRectBivariateSpline.get_coeffs()


    def writeEDF(self, basename):
        """save the distortion matrices into a couple of files called basename-x.edf and  basename-y.edf
        
        """
        try:
            from fabio.edfimage import edfimage
            #from EdfFile import EdfFile as EDF
        except ImportError:
            print "You will need the Fabio library available from the Fable sourceforge"
            return
        self.spline2array()

        edfDispX = edfimage(data=self.xDispArray.astype("float32"), header={})
        edfDispY = edfimage(data=self.yDispArray.astype("float32"), header={})
        edfDispX.write(basename + "-x.edf", force_type="float32")
        edfDispY.write(basename + "-y.edf", force_type="float32")


    def write(self, filename):
        """save the cubic spline in an ascii file usable with Fit2D or SPD
        @param filename: name of the file containing the cubic spline distortion file
        @type filename: string
        """

        txt = "SPATIAL DISTORTION SPLINE INTERPOLATION COEFFICIENTS\n\n  VALID REGION\n%14.7E%14.7E%14.7E%14.7E\n\n" % (self.xmin, self.ymin, self.xmax, self.ymax)
        txt += "  GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE\n%14.7E%14.7E%14.7E\n\n" % (self.grid, self.pixelSize[0], self.pixelSize[1])
        txt += "  X-DISTORTION\n%6i%6i" % (len(self.xSplineKnotsX), len(self.xSplineKnotsY))
        for i in range(len(self.xSplineKnotsX)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.xSplineKnotsX[i]
        for i in range(len(self.xSplineKnotsY)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.xSplineKnotsY[i]
        for i in range(len(self.xSplineCoeff)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.xSplineCoeff[i]
        txt += "\n\n  Y-DISTORTION\n%6i%6i" % (len(self.ySplineKnotsX), len(self.ySplineKnotsY))
        for i in range(len(self.ySplineKnotsX)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.ySplineKnotsX[i]
        for i in range(len(self.ySplineKnotsY)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.ySplineKnotsY[i]
        for i in range(len(self.ySplineCoeff)):
            if i % 5 == 0:
                txt += "\n"
            txt += "%14.7E" % self.ySplineCoeff[i]
        txt += "\n"
        open(filename, "w").write(txt)


    def tilt(self, center=(0.0, 0.0), tiltAngle=0.0, tiltPlanRot=0.0, distanceSampleDetector=1.0, timing=False):
        """The tilt method apply a virtual tilt on the detector, the point of tilt is given by the center
        
        @param center: position of the point of tilt, this point will not be moved.
        @type center: 2tuple of floats
        @param tiltAngle: the value of the tilt in degrees
        @type tiltAngle: float in the range [-90:+90] degrees
        @param tiltPlanRot: the rotation of the tilt plan with the Ox axis (0 deg for y axis invariant, 90 deg for x axis invariant)  
        @type tiltPlanRot: Float in the range [-180:180]
        @type distanceSampleDetector: float
        @param distanceSampleDetector: the distance from sample to detector in meter (along the beam, so distance from sample to center)
        @return: tilted Spline instance
        @rtype: Spline
        """
        if self.xDispArray is None:
            if self.filename is None:
                self.zeros()
            else:
                self.read()
        print("center=%s, tilt=%s, tiltPlanRot=%s, distanceSampleDetector=%sm, pixelSize=%sµm" % (center, tiltAngle, tiltPlanRot, distanceSampleDetector, self.pixelSize))
        if timing:
            startTime = time.time()
        distance = 1.0e6 * distanceSampleDetector #from meters to microns
        cosb = numpy.cos(numpy.radians(tiltPlanRot))
        sinb = numpy.sin(numpy.radians(tiltPlanRot))
        cosf = numpy.cos(numpy.radians(tiltAngle))
        sinf = numpy.sin(numpy.radians(tiltAngle))

        x = lambda i, j: j - center[0] - 0.5 #x and y are tilted in C/Fortran representation
        y = lambda i, j: i - center[1] - 0.5

        iPos = numpy.fromfunction(x, (int(self.ymax - self.ymin + 1), int(self.xmax - self.xmin + 1)))
        jPos = numpy.fromfunction(y, (int(self.ymax - self.ymin + 1), int(self.xmax - self.xmin + 1)))

        xPos = (iPos + self.xDispArray) * self.pixelSize[0]
        yPos = (jPos + self.yDispArray) * self.pixelSize[1]

        tiltArrayX = distance * (xPos * (cosf * cosb * cosb + sinb * sinb) + yPos * (cosf * cosb * sinb - cosb * sinb)) / \
                        (distance + xPos * sinf * cosb + yPos * sinf * sinb) / self.pixelSize[0] - iPos
        tiltArrayY = distance * (xPos * (cosf * sinb * cosb - cosb * sinb) + yPos * (cosf * sinb * sinb + cosb * cosb)) / \
                        (distance + xPos * sinf * cosb + yPos * sinf * sinb) / self.pixelSize[1] - jPos
        tiltedSpline = Spline()
        tiltedSpline.pixelSize = self.pixelSize
        tiltedSpline.grid = self.grid
        tiltedSpline.xDispArray = tiltArrayX
        tiltedSpline.yDispArray = tiltArrayY
        #tiltedSpline.array2spline(smoothing=1e-6, timing=True)
        if timing:
            print("Time for the generation of the distorted spline: %.3f sec" % (time.time() - startTime))
        return tiltedSpline


#    def setPixelSize(self, pixelSize):
#        """
#        sets the size of the pixel from a 2-tuple of floats expressed in microns.
#        """
#        if len(pixelSize) == 2 :
#            self.pixelSize = pixelSize
#    def getPixelSize(self):
#        """
#        @return: the size of the pixel from a 
#        @rtype: 2-tuple of floats expressed in microns.
#        """
#        return self.pixelSize
#    

    def setPixelSize(self, pixelSize):
        """
        sets the size of the pixel from a 2-tuple of floats expressed in meters.
        @param: pixel size in meter
        @type pixelSize: 2-tuple of float 
        """
        if len(pixelSize) == 2 :
            self.pixelSize = (pixelSize[0] * 1.0e6, pixelSize[1] * 1.0e6)


    def getPixelSize(self):
        """
        @return: the size of the pixel from a 2D detector
        @rtype: 2-tuple of floats expressed in meter.
        """
        return (self.pixelSize[0] * 1.0e-6, self.pixelSize[1] * 1.0e-6)


#    def horizontalFlip(self):
#        """calculate the flipped spline file interverting xmin and xmax
#        @return: another spline file"""
#        other = Spline()
#        other.xmin = self.xmin
#        other.xmax = self.xmax
#        other.ymin = self.ymin
#        other.ymax = self.ymax
#        other.pixelSize = self.pixelSize
#        other.grid = self.grid
#        other.xSplineKnotsX = [ self.xmax + self.xmin - x for x in self.xSplineKnotsX  ]
#        #other.xSplineKnotsX = self.xSplineKnotsX[:]
#        other.xSplineKnotsY = self.xSplineKnotsY[:]
#        other.xSplineCoeff = [ -i for i in self.xSplineCoeff]
#        other.ySplineKnotsX = [ self.xmax + self.xmin - x for x in self.xSplineKnotsX  ]
#        #other.ySplineKnotsX = self.ySplineKnotsX[:]
#        other.ySplineKnotsY = self.ySplineKnotsY[:]
#        other.ySplineCoeff = self.ySplineCoeff[:]
#        return other
#
#def xDispHorFlip(array):
#    """make an horizontal flip of the given X displacement array"""
#    return - numpy.fliplr(array)
#def yDispHorFlip(array):
#    """make an horizontal flip of the given Y displacement array"""
#    return numpy.fliplr(array)
#def xDispVerFlip(array):
#    """make a vertical flip of the given X displacement array"""
#    return numpy.flipud(array)
#def yDispVerFlip(array):
#    """make a vertical flip of the given Y displacement array"""
#    return - numpy.flipud(array)


if __name__ == '__main__':
#    """this is the main program if somebody wants to use this as a library"""
#
#
#
#
##    spline.write("test")
#    new = Spline()
#    new.pixelSize = spline.pixelSize
#    new.grid = spline.grid
#    new.xDispArray = xDispHorFlip(spline.xDispArray[:])
#    new.yDispArray = yDispHorFlip(spline.yDispArray[:])
#    new.array2spline(smoothing=10, timing=False)
#    print "matrix flipped", new
#    flipped = spline.horizontalFlip()
#    print "Spline flipped", flipped
#
#
#    print "---" * 50
#
#
#    print new.comparison(flipped, verbose=True)

#    new.write("new.spline")
#    new.xDispArray = None
#    new.yDispArray = None
#    new.spline2array(timing=True)
#    print new.comparison(spline, verbose=True)
    # Size of the image in pixels:
##########################################################    
#    spline = Spline()
#    spline.zeros(0, 0, 2000, 2000)
#    spline.spline2array()
#    spline.pixelSize = (100, 100)
#    spline.grid = 1

    CENTER = (1000, 1000)
    TILT = 10 #deg
    ROTATION_TILT = 0 #deg
    DISTANCE = 100 #mm
    SPLINE_FILE = "example.spline"
    for keyword in sys.argv[1:]:
        if os.path.isfile(keyword):
            SPLINE_FILE = keyword
        elif keyword.lower().find("center=") in [0, 1, 2]:
            CENTER = map(float, keyword.split("=")[1].split("x"))
        elif keyword.lower().find("dist=") in [0, 1, 2]:
            DISTANCE = float(keyword.split("=")[1])
        elif keyword.lower().find("tilt=") in [0, 1, 2]:
            TILT = float(keyword.split("=")[1])
        elif keyword.lower().find("rot=") in [0, 1, 2]:
            ROTATION_TILT = float(keyword.split("=")[1])


    spline = Spline()
    spline.read(SPLINE_FILE)
    print ("Original Spline: %s" % spline)
    spline.spline2array(timing=True)
    tilted = spline.tilt(CENTER, TILT, ROTATION_TILT, DISTANCE, timing=True)
    #tilted.write("tilted-t%i-p%i-d%i.spline" % (TILT, ROTATION_TILT, DISTANCE))
    tilted.writeEDF("%s-tilted-t%i-p%i-d%i" % (os.path.splitext(SPLINE_FILE)[0], TILT, ROTATION_TILT, DISTANCE))
#    for i in range(0, 14, 2):
#        tilted.array2spline(smoothing=(10.0 ** (-i)), timing=True)
#        tilted.write("tilted-t%i-p%i-d%i-s%i.spline" % (TILT, ROTATION_TILT, DISTANCE, i))
#        fromspline = Spline()
#        fromspline.read("tilted-t%i-p%i-d%i-s%i.spline" % (TILT, ROTATION_TILT, DISTANCE, i))
#        print fromspline.comparison(tilted, verbose=True)
#    spline = Spline()
#    spline.read("tilted-t%i-p%i-d%i.spline" % (TILT, ROTATION_TILT, DISTANCE))
#    spline.spline2array(timing=True)
#    print spline.comparison(tilted, verbose=True)
