#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/03/2017"
__status__ = "development"
__docformat__ = 'restructuredtext'


import os
import logging
import numpy
from collections import OrderedDict
from .massif import Massif
from .control_points import ControlPoints
from .detectors import detector_factory
from .geometry import Geometry
from .geometryRefinement import GeometryRefinement
from .azimuthalIntegrator import AzimuthalIntegrator
from .utils import StringTypes
from .ext.marchingsquares import isocontour
logger = logging.getLogger("pyFAI.goniometer")


class SingleGeometry(object):
    """This class represents a single geometry of a detector position on a 
    goniometer arm
    """
    def __init__(self, label, image=None, metadata=None, position_function=None,
                 control_points=None, calibrant=None, detector=None, geometry=None):
        """Constructor of the SingleGeometry class, used for calibrating a 
        multi-geometry setup with a moving detector
        
        :param label: name of the geometry, a string or anything unmutable
        :param image: image with Debye-Scherrer rings as 2d numpy array
        :param metadata: anything which contains the goniometer position
        :param position_function: a function which takes the metadata as input 
                                 and returns the goniometer arm position
        Optional parameters:
        :param control_points: a pyFAI.control_points.ControlPoints instance
        :param calibrant: a pyFAI.calibrant.Calibrant instance. 
                        Contains the wavelength to be used
         :param detector: a pyFAI.detectors.Detector instance or something like that 
                        Contains the mask to be used
        :param geometry: an azimuthal integrator or a ponifile 
                        (or a dict with the geometry)  
                         
        """
        self.label = label
        self.image = image
        self.metadata = metadata  # may be anything
        self.control_points = control_points
        self.calibrant = calibrant
        if detector is not None:
            self.detector = detector_factory(detector)
        else:
            self.detector = None
        if isinstance(geometry, Geometry):
            dict_geo = geometry.getPyFAI()
        elif isinstance(geometry, StringTypes) and os.path.exists(geometry):
            dict_geo = Geometry.sload(geometry).getPyFAI()
        elif isinstance(geometry, dict):
            dict_geo = geometry
        if self.detector is not None:
            dict_geo["detector"] = self.detector
        self.geometry_refinement = GeometryRefinement(**dict_geo)
        if self.detector is None:
            self.detector = self.geometry_refinement.detector
        self.position_function = position_function
        self.massif = None

    def get_position(self):
        """This method  is in charge of calculating the motor position from metadata/label/..."""
        return self.position_function(self.metadata)

    def extract_cp(self, max_rings=None, pts_per_deg=1.0):
        """Performs an automatic keypoint extraction and update the geometry refinement part

        :param max_ring: extract at most N rings from the image
        :param pts_per_deg: number of control points per azimuthal degree (increase for better precision)
        """
        if self.massif is None:
            self.massif = Massif(self.image)

        tth = numpy.array([i for i in self.calibrant.get_2th() if i is not None])
        tth = numpy.unique(tth)
        tth_min = numpy.zeros_like(tth)
        tth_max = numpy.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 4.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth
        shape = self.image.shape
        ttha = self.geometry_refinement.twoThetaArray(shape)
        chia = self.geometry_refinement.chiArray(shape)
        rings = 0
        cp = ControlPoints(calibrant=self.calibrant)
        if max_rings is None:
            max_rings = tth.size
        for i in range(tth.size):
            if rings >= max_rings:
                break
            mask = numpy.logical_and(ttha >= tth_min[i], ttha < tth_max[i])
            if self.detector.mask is not None:
                mask = numpy.logical_and(mask, numpy.logical_not(self.geometry_refinement.detector.mask))
            size = mask.sum(dtype=int)
            if (size > 0):
                rings += 1
                sub_data = self.image.ravel()[numpy.where(mask.ravel())]
                mean = sub_data.mean(dtype=numpy.float64)
                std = sub_data.std(dtype=numpy.float64)
                upper_limit = mean + std
                mask2 = numpy.logical_and(self.image > upper_limit, mask)
                size2 = mask2.sum(dtype=int)
                if size2 < 1000:
                    upper_limit = mean
                    mask2 = numpy.logical_and(self.image > upper_limit, mask)
                    size2 = mask2.sum()
                # length of the arc:
                points = isocontour(ttha, tth[i]).round().astype(int)
                seeds = set((i[1], i[0]) for i in points if mask2[i[1], i[0]])
                # max number of points: 360 points for a full circle
                azimuthal = chia[points[:, 1].clip(0, shape[0]), points[:, 0].clip(0, shape[1])]
                nb_deg_azim = numpy.unique(numpy.rad2deg(azimuthal).round()).size
                keep = int(nb_deg_azim * pts_per_deg)
                if keep == 0:
                    continue
                dist_min = len(seeds) / 2.0 / keep
                # why 3.0, why not ?

                logger.info("Extracting datapoint for ring %s (2theta = %.2f deg); " +
                            "searching for %i pts out of %i with I>%.1f, dmin=%.1f",
                            i, numpy.degrees(tth[i]), keep, size2, upper_limit, dist_min)
                res = self.massif.peaks_from_area(mask2, Imin=0, keep=keep, dmin=dist_min, seed=seeds, ring=i)
                cp.append(res, i)
        self.control_points = cp
        self.geometry_refinement.data = numpy.asarray(cp.getList(), dtype=numpy.float64)
        return cp

    def display(self):
        """
        Display the image with the control points and the iso-contour overlaid. 
        
        @return: the figure to be showed
        """
        # should already be set-up ...
        from pylab import figure, legend

        if self.image is None:
            return
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(numpy.arcsinh(self.image), origin="lower")
        if self.control_points is not None:
            cp = self.control_points
            for lbl in cp.get_labels():
                pt = numpy.array(cp.get(lbl=lbl).points)
                ax.scatter(pt[:, 1], pt[:, 0], label=lbl)
            legend()
        if self.geometry_refinement is not None and self.calibrant is not None:
            ai = self.geometry_refinement
            tth = self.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(ttha, levels=tth, cmap="autumn", linewidths=2, linestyles="dashed")
        return fig

    def get_ai(self):
        """Create a new azimuthal integrator to be used.

        @return: Azimuthal Integrator instance
        """
        return AzimuthalIntegrator(detector=self.detector,
                                   **self.geometry_refinement.getPyFAI())

class Goniometer(object):
    """This class represents the goniometer modelisation
    """

class GoniometerRefinement(Goniometer):
    """This class allow the translation of a goniometer geometry into a pyFAI 
    geometry using a set of parameter to refine. 
    """
    def __init__(self, param, position_function, translation_function,
                 bounds=None):
        """Constructor of the GoniometerRefinement class
        
        :param param: vector of parameter to refine for defining the detector 
                        position on the goniometer
        :parma position_function: a function taking metadata and extracting the 
                                  goniometer position
        :param translation_function: function taking the parameters of the 
                                    goniometer and the gonopmeter position and return the
                                    6 parameters [dist, poni1, poni2, rot1, rot2, rot3] 
        :param bounds: 
        """
        self.single_geometries = OrderedDict()  # a dict of labels: SingleGeometry
        self.multiparam = param
        self.bounds = bounds
        self.position_function = position_function
        self.translation_function = translation_function

    def new_geometry(self, label, image=None, metadata=None, controlpoints=None, calibrant=None, geometryrefinement=None):
        self.single_geometries[label] = SingleGeometry(label, image, metadata, controlpoints, calibrant, geometryrefinement)

    def __repr__(self):
        return "MultiGeometryRefinement with %i geometries labeled: %s" % \
                (len(self.single_geometries), " ".join(self.single_geometries.keys()))

    def translate(self, multiparam, motor_pos):
        """translate a set of param of the multigeometry into a paramter for SingleGeometry
        This is where the goniometer definition is
        """
        return numpy.concatenate((multiparam[0:3] , [-numpy.radians(multiparam[5] * motor_pos + multiparam[4]), multiparam[3], numpy.pi / 2.0]))
        # return numpy.concatenate((multiparam[0:4], [numpy.radians(motor_pos * multiparam[5]+ multiparam[4]), 0]))

    def residu2(self, param):
        sumsquare = 0
        for key, single in self.single_geometries.items():
            motor_pos = single.get_position()
            single_param = self.translate(param, motor_pos)
            if single.geometryrefinement is not None:
                sumsquare += single.geometryrefinement.chi2(single_param) / single.geometryrefinement.data.shape[0]
        return sumsquare

    def chi2(self, param=None):
        if param is not None:
            return self.residu2(param)
        else:
            return self.residu2(self.multiparam)

    def refine2(self, maxiter=1000):
        self.multiparam = numpy.asarray(self.multiparam, dtype=numpy.float64)
        newparam = fmin_slsqp(self.residu2, self.multiparam, iter=maxiter,
                              iprint=2, bounds=self.bounds,
                              acc=1.0e-12)
        print(newparam)
        print("Constrained Least square", self.chi2(), "--> ", self.chi2(newparam))
        if self.chi2(newparam) < self.chi2():
            i = abs(self.multiparam - newparam).argmax()
            print("maxdelta on: ", i, self.multiparam[i], "-->", newparam[i])
            self.multiparam = newparam
        return self.multiparam
    def get_ai(self, motor_pos):
        """Creates an azimuthal integrator from the motor position"""
        r = self.translate(self.multiparam, motor_pos)

    def get_mg(self, lbl, motor_pos):

        gr = self.single_geometries[lbl].geometryrefinement
        ai = AzimuthalIntegrator(detector=gr.detector, wavelength=gr.wavelength)
        ai.dist = r[0]
        ai.poni1 = r[1]
        ai.poni2 = r[2]
        ai.rot1 = r[3]
        ai.rot2 = r[4]
        ai.rot3 = r[5]
        return ai
