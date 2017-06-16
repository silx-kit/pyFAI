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

"""Everything you need to calibrate a detector mounted on a goniometer or any
translation table
"""

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/06/2017"
__status__ = "development"
__docformat__ = 'restructuredtext'


import os
import logging
import json
import numpy
from collections import OrderedDict, namedtuple
from scipy.optimize import minimize
from .massif import Massif
from .control_points import ControlPoints
from .detectors import detector_factory, Detector
from .geometry import Geometry
from .geometryRefinement import GeometryRefinement
from .azimuthalIntegrator import AzimuthalIntegrator
from .utils import StringTypes
from .multi_geometry import MultiGeometry
from .ext.marchingsquares import isocontour
logger = logging.getLogger("pyFAI.goniometer")

try:
    import numexpr
except ImportError:
    numexpr = None

# Parameter set used in PyFAI:
PoniParam = namedtuple("PoniParam", ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"])


class GeometryTransformation(object):
    """This class, once instanciated, behaves like a function (via the __call__
    method). It is responsible for taking any input geometry and translate it
    into a set of parameters compatible with pyFAI, i.e. a tuple with:
    (dist, poni1, poni2, rot1, rot2, rot3)

    This function uses numexpr for formula evaluation.
    """
    def __init__(self, dist_expr, poni1_expr, poni2_expr,
                 rot1_expr, rot2_expr, rot3_expr,
                 param_names, pos_names=None, constants=None,
                 content=None):
        """Constructor of the class

        :param dist_expr: formula (as string) providing with the dist
        :param poni1_expr: formula (as string) providing with the poni1
        :param poni2_expr: formula (as string) providing with the poni2
        :param rot1_expr: formula (as string) providing with the rot1
        :param rot2_expr: formula (as string) providing with the rot2
        :param rot3_expr: formula (as string) providing with the rot3
        :param param_names: list of names of the parameters used in the model
        :param pos_names: list of motor names for gonio with >1 degree of freedom
        :param constants: a dictionary with some constants the user may want to use
        :param content: Should be None or the name of the class (may be used
            in the future to dispatch to multiple derivative classes)
        """
        if content is not None:
            # Ensures we use the constructor of the right class
            assert content in (self.__class__.__name__, "GeometryTranlation")
        if numexpr is None:
            raise RuntimeError("Geometry translation requires the *numexpr* package")
        self.dist_expr = dist_expr
        self.poni1_expr = poni1_expr
        self.poni2_expr = poni2_expr
        self.rot1_expr = rot1_expr
        self.rot2_expr = rot2_expr
        self.rot3_expr = rot3_expr

        self.variables = {"pi": numpy.pi}
        if constants is not None:
            self.variables.update(constants)

        self.param_names = tuple(param_names)
        if pos_names is not None:
            self.pos_names = tuple(pos_names)
        else:
            self.pos_names = ("pos",)
        for key in self.param_names + self.pos_names:
            if key in self.variables:
                raise RuntimeError("The keyword %s is already defined, please chose another variable name")
            self.variables[key] = numpy.NaN

        self.codes = [numexpr.NumExpr(expr) for expr in (self.dist_expr, self.poni1_expr, self.poni2_expr,
                                                         self.rot1_expr, self.rot2_expr, self.rot3_expr)]

    def __call__(self, param, pos):
        """This makes the class instance behave like a function,
        actually a function that translates the n-parameter of the detector
        positioning on the goniometer and the m-parameters.

        :param param: parameter of the fit
        :param pos: position of the goniometer (representation from the
            goniometer)
        :return: 6-tuple with (dist, poni1, poni2, rot1, rot2, rot3) as needed
            for pyFAI.
        """
        res = []
        variables = self.variables.copy()
        for name, value in zip(self.param_names, param):
            variables[name] = value
        if len(self.pos_names) == 1:
            variables[self.pos_names[0]] = pos
        else:
            for name, value in zip(self.pos_names, pos):
                variables[name] = value
        for code in self.codes:
            signa = [variables.get(name, numpy.NaN) for name in code.input_names]
            res.append(float(code(*signa)))
            # could ne done in a single liner but harder to understand !
        return PoniParam(*res)

    def __repr__(self):
        res = ["GeometryTransformation with param: %s and pos: %s" % (self.param_names, self.pos_names),
               "    dist= %s" % self.dist_expr,
               "    poni1= %s" % self.poni1_expr,
               "    poni2= %s" % self.poni2_expr,
               "    rot1= %s" % self.rot1_expr,
               "    rot2= %s" % self.rot2_expr,
               "    rot3= %s" % self.rot3_expr]
        return os.linesep.join(res)

    def to_dict(self):
        """Export the instance representation for serialization as a dictionary
        """
        res = OrderedDict([("content", self.__class__.__name__),
                           ("param_names", self.param_names),
                           ("pos_names", self.pos_names),
                           ("dist_expr", self.dist_expr),
                           ("poni1_expr", self.poni1_expr),
                           ("poni2_expr", self.poni2_expr),
                           ("rot1_expr", self.rot1_expr),
                           ("rot2_expr", self.rot2_expr),
                           ("rot3_expr", self.rot3_expr),
                           ])
        constants = OrderedDict()
        for key, val in self.variables.items():
            if key in self.param_names:
                continue
            if self.pos_names and key in self.pos_names:
                continue
            constants[key] = val
        res["constants"] = constants
        return res

GeometryTranslation = GeometryTransformation


class Goniometer(object):
    """This class represents the goniometer model. Unlike this name suggests,
    it may include translation in addition to rotations
    """

    file_version = "Goniometer calibration v1.0"

    def __init__(self, param, trans_function, detector="Detector",
                 wavelength=None, param_names=None, pos_names=None):
        """Constructor of the Goniometer class.

        :param param: vector of parameter to refine for defining the detector
                        position on the goniometer
        :param trans_function: function taking the parameters of the
                        goniometer and the goniometer position and return the
                        6 parameters [dist, poni1, poni2, rot1, rot2, rot3]
        :param detector: detector mounted on the moving arm
        :param wavelength: the wavelength used for the experiment
        :param param_names: list of names to "label" the param vector.
        :param pos_names: list of names to "label" the position vector of
            the gonio.
        """

        self.param = param
        self.trans_function = trans_function
        self.detector = detector_factory(detector)
        self.wavelength = wavelength
        if param_names is None and "param_names" in dir(trans_function):
            param_names = trans_function.param_names
        if param_names is not None:
            if isinstance(param, dict):
                self.param = [param.get(i, 0) for i in param_names]
            self.nt_param = namedtuple("GonioParam", param_names)
        else:
            self.nt_param = lambda *x: tuple(x)
        if pos_names is None and "pos_names" in dir(trans_function):
            pos_names = trans_function.pos_names
        self.nt_pos = namedtuple("GonioPos", pos_names) if pos_names else lambda *x: tuple(x)

    def __repr__(self):
        return "Goniometer with param %s    %s with %s" % (self.nt_param(*self.param), os.linesep, self.detector)

    def get_ai(self, position):
        """Creates an azimuthal integrator from the motor position

        :param position: the goniometer position, a float for a 1 axis
            goniometer
        :return: A freshly build AzimuthalIntegrator
        """
        res = self.trans_function(self.param, position)
        ai = AzimuthalIntegrator(detector=self.detector, wavelength=self.wavelength)
        ai.dist, ai.poni1, ai.poni2, ai.rot1, ai.rot2, ai.rot3 = res
        return ai

    def get_mg(self, positions):
        """Creates a MultiGeometry integrator from a list of goniometer
        positions.

        :param positions: A list of goniometer positions
        :return: A freshly build multi-geometry
        """
        ais = [self.get_ai(pos) for pos in positions]
        mg = MultiGeometry(ais)
        return mg

    def to_dict(self):
        """Export the goniometer configuration to a dictionary

        :return: Ordered dictionary
        """
        dico = OrderedDict([("content", self.file_version)])
        dico.update(self.detector.getPyFAI())
        if self.wavelength:
            dico["wavelength"] = self.wavelength
        dico["param"] = tuple(self.param)
        if "_fields" in dir(self.nt_param):
            dico["param_names"] = self.nt_param._fields
        if "_fields" in dir(self.nt_pos):
            dico["pos_names"] = self.nt_pos._fields
        if "to_dict" in dir(self.trans_function):
            dico["trans_function"] = self.trans_function.to_dict()
        else:
            logger.warning("trans_function is not serializable")
        return dico

    def save(self, filename):
        """Save the goniometer configuration to file

        :param filename: name of the file to save configuration to
        """
        dico = self.to_dict()
        try:
            with open(filename, "w") as f:
                f.write(json.dumps(dico, indent=2))
        except IOError:
            logger.error("IOError while writing to file %s", filename)
    write = save

    @classmethod
    def sload(cls, filename):
        with open(filename) as f:
            dico = json.load(f)
        assert dico["content"] == cls.file_version, "JSON file contains a goniometer calibration"
        assert "trans_function" in dico, "No translation function defined in JSON file"
        detector = Detector.from_dict(dico)
        tansfun = dico.get("trans_function", {})
        if "content" in tansfun:
            content = tansfun.pop("content")
            assert content in ("GeometryTranslation", "GeometryTransformation")
            # May be adapted for other classes of GeometryTransformation functions
        funct = GeometryTransformation(**tansfun)
        gonio = cls(dico.get("param", []), funct, detector, dico.get("wavelength"))
        return gonio


class SingleGeometry(object):
    """This class represents a single geometry of a detector position on a
    goniometer arm
    """
    def __init__(self, label, image=None, metadata=None, pos_function=None,
                 control_points=None, calibrant=None, detector=None, geometry=None):
        """Constructor of the SingleGeometry class, used for calibrating a
        multi-geometry setup with a moving detector.

        :param label: name of the geometry, a string or anything unmutable
        :param image: image with Debye-Scherrer rings as 2d numpy array
        :param metadata: anything which contains the goniometer position
        :param pos_function: a function which takes the metadata as input
                                 and returns the goniometer arm position
        :param control_points: a pyFAI.control_points.ControlPoints instance
            (optional parameter)
        :param calibrant: a pyFAI.calibrant.Calibrant instance.
                        Contains the wavelength to be used (optional parameter)
        :param detector: a pyFAI.detectors.Detector instance or something like
                        that Contains the mask to be used (optional parameter)
        :param geometry: an azimuthal integrator or a ponifile
                        (or a dict with the geometry) (optional parameter)
        """
        self.label = label
        self.image = image
        self.metadata = metadata  # may be anything
        self.calibrant = calibrant
        if control_points is None or isinstance(control_points, ControlPoints):
            self.control_points = control_points
        else:
            # Probaly a NPT file
            self.control_points = ControlPoints(control_points, calibrant=calibrant)

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
        if self.control_points is not None:
            dict_geo["data"] = self.control_points.getList()
        if self.calibrant is not None:
            dict_geo["calibrant"] = self.calibrant
        self.geometry_refinement = GeometryRefinement(**dict_geo)
        if self.detector is None:
            self.detector = self.geometry_refinement.detector
        self.pos_function = pos_function
        self.massif = None

    def get_position(self):
        """This method  is in charge of calculating the motor position from metadata/label/..."""
        return self.pos_function(self.metadata)

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

    def get_ai(self):
        """Create a new azimuthal integrator to be used.

        :return: Azimuthal Integrator instance
        """
        return AzimuthalIntegrator(detector=self.detector,
                                   **self.geometry_refinement.getPyFAI())


class GoniometerRefinement(Goniometer):
    """This class allow the translation of a goniometer geometry into a pyFAI
    geometry using a set of parameter to refine.
    """
    def __init__(self, param, pos_function, trans_function,
                 detector="Detector", wavelength=None, param_names=None, pos_names=None,
                 bounds=None):
        """Constructor of the GoniometerRefinement class

        :param param: vector of parameter to refine for defining the detector
                            position on the goniometer
        :param pos_function: a function taking metadata and extracting the
                            goniometer position
        :param trans_function: function taking the parameters of the
                            goniometer and the gonopmeter position and return the
                            6 parameters [dist, poni1, poni2, rot1, rot2, rot3]
        :param detector: detector mounted on the moving arm
        :param wavelength: the wavelength used for the experiment
        :param param_names: list of names to "label" the param vector.
        :param pos_names: list of names to "label" the position vector of the
                            gonio.
        :param bounds: list of 2-tuple with the lower and upper bound of each function
        """
        Goniometer.__init__(self, param, trans_function,
                            detector=detector, wavelength=wavelength,
                            param_names=param_names, pos_names=pos_names)
        self.single_geometries = OrderedDict()  # a dict of labels: SingleGeometry
        if bounds is None:
            self.bounds = [(None, None)] * len(self.param)
        else:
            if isinstance(bounds, dict) and "_fields" in dir(self.nt_param):
                self.bounds = [bounds.get(i, (None, None))
                               for i in self.nt_param._fields]
            else:
                self.bounds = list(bounds)
        self.pos_function = pos_function

    def new_geometry(self, label, image=None, metadata=None, control_points=None,
                     calibrant=None, geometry=None):
        """Add a new geometry for calibration

        :param label: usually a string
        :param image: 2D numpy array with the Debye scherrer rings
        :param metadata: some metadata
        :param control_points: an instance of ControlPoints
        :param calibrant: the calibrant used for calibrating
        :param geometry: poni or AzimuthalIntegrator instance.
        """
        if geometry is None:
            geometry = self.get_ai(self.pos_function(metadata))
        sg = SingleGeometry(label=label,
                            image=image,
                            metadata=metadata,
                            control_points=control_points,
                            calibrant=calibrant,
                            detector=self.detector,
                            pos_function=self.pos_function,
                            geometry=geometry)
        self.single_geometries[label] = sg
        return sg

    def __repr__(self):
        return "%s with %i geometries labeled: %s" % \
                (self.__class__.__name__, len(self.single_geometries),
                 ", ".join(self.single_geometries.keys()) + ".")

    def residu2(self, param):
        "Actually performs the calulation of the average of the error squared"
        sumsquare = 0.0
        npt = 0
        for single in self.single_geometries.values():
            motor_pos = single.get_position()
            single_param = self.trans_function(param, motor_pos)
            if single.geometry_refinement is not None and len(single.geometry_refinement.data) > 1:
                sumsquare += single.geometry_refinement.chi2(single_param)
                npt += single.geometry_refinement.data.shape[0]
        return sumsquare / max(npt, 1)

    def chi2(self, param=None):
        """Calculate the average of the square of the error for a given parameter set
        """
        if param is not None:
            return self.residu2(param)
        else:
            return self.residu2(self.param)

    def refine2(self, method="slsqp", **options):
        """Geometry refinement tool

        See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html

        :param method: name of the minimizer
        :param options: options for the minimizer
        """
        former_error = self.chi2()
        print("Cost function before refinement: %s" % former_error)
        param = numpy.asarray(self.param, dtype=numpy.float64)
        res = minimize(self.residu2, param, method=method,
                       bounds=self.bounds, tol=1e-12,
                       options=options)
        print(res)
        newparam = res.x
        new_error = res.fun
        print("Cost function after refinement: %s" % new_error)
        print(self.nt_param(*newparam))

        # print("Constrained Least square %s --> %s" % (former_error, new_error))
        if new_error < former_error:
            i = abs(param - newparam).argmax()
            if "_fields" in dir(self.nt_param):
                name = self.nt_param._fields[i]
                print("maxdelta on: %s (%i) %s --> %s" % (name, i, self.param[i], newparam[i]))
            else:
                print("maxdelta on: %i %s --> %s" % (i, self.param[i], newparam[i]))
            self.param = newparam
        return self.param

    def set_bounds(self, name, mini=None, maxi=None):
        """Redefines the bounds for the refinement

        :param name: name of the parameter or index in the parameter set
        :param mini: minimum value
        :param maxi: maximum value
        """
        if isinstance(name, StringTypes) and "_fields" in dir(self.nt_param):
            idx = self.nt_param._fields.index(name)
        else:
            idx = int(name)
        self.bounds[idx] = (mini, maxi)
