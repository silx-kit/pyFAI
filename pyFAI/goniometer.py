#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "22/08/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'


import os
import logging
import json
import numpy
from collections import OrderedDict, namedtuple
from scipy.optimize import minimize
from silx.image import marchingsquares
from .massif import Massif
from .control_points import ControlPoints
from .detectors import detector_factory, Detector
from .geometry import Geometry
from .geometryRefinement import GeometryRefinement
from .azimuthalIntegrator import AzimuthalIntegrator
from .utils import StringTypes
from .multi_geometry import MultiGeometry
from .units import CONST_hc, CONST_q

logger = logging.getLogger(__name__)

try:
    import numexpr
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    numexpr = None

# Parameter set used in PyFAI:
PoniParam = namedtuple("PoniParam", ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"])


class BaseTransformation(object):
    """This class, once instanciated, behaves like a function (via the __call__
    method). It is responsible for taking any input geometry and translate it
    into a set of parameters compatible with pyFAI, i.e. a tuple with:
    (dist, poni1, poni2, rot1, rot2, rot3)

    This class relies on a user provided function which does the work.
    """
    def __init__(self, funct, param_names, pos_names=None):
        """Constructor of the class

        :param funct: function which takes as parameter the param_names and the pos_name
        :param param_names: list of names of the parameters used in the model
        :param pos_names: list of motor names for gonio with >1 degree of freedom
        """
        self.callable = funct
        self.variables = {}
        self.param_names = tuple(param_names)
        if pos_names is not None:
            self.pos_names = tuple(pos_names)
        else:
            self.pos_names = ("pos",)
        for key in self.param_names + self.pos_names:
            if key in self.variables:
                raise RuntimeError("The keyword %s is already defined, please chose another variable name")
            self.variables[key] = numpy.NaN
        self.codes = []

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
        variables = self.variables.copy()
        for name, value in zip(self.param_names, param):
            variables[name] = value
        if len(self.pos_names) == 1:
            variables[self.pos_names[0]] = pos
        else:
            for name, value in zip(self.pos_names, pos):
                variables[name] = value

        res = self.callable(**variables)
        return PoniParam(*res)

    def __repr__(self):
        return "BaseTransformation with param: %s and pos: %s" % (self.param_names, self.pos_names)

    def to_dict(self):
        """Export the instance representation for serialization as a dictionary
        """
        raise RuntimeError("BaseTransformation is not serializable")


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
            assert content in (self.__class__.__name__, "GeometryTransformation")
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


class ExtendedTransformation(object):
    """This class behaves like GeometryTransformation and extends transformation
    to the wavelength parameter.

    This function uses numexpr for formula evaluation.
    """
    def __init__(self, dist_expr=None, poni1_expr=None, poni2_expr=None,
                 rot1_expr=None, rot2_expr=None, rot3_expr=None, wavelength_expr=None,
                 param_names=None, pos_names=None, constants=None,
                 content=None):
        """Constructor of the class

        :param dist_expr: formula (as string) providing with the dist
        :param poni1_expr: formula (as string) providing with the poni1
        :param poni2_expr: formula (as string) providing with the poni2
        :param rot1_expr: formula (as string) providing with the rot1
        :param rot2_expr: formula (as string) providing with the rot2
        :param rot3_expr: formula (as string) providing with the rot3
        :param wavelength_expr: formula (as a string) to calculate wavelength used in angstrom
        :param param_names: list of names of the parameters used in the model
        :param pos_names: list of motor names for gonio with >1 degree of freedom
        :param constants: a dictionary with some constants the user may want to use
        :param content: Should be None or the name of the class (may be used
            in the future to dispatch to multiple derivative classes)
        """
        if content is not None:
            # Ensures we use the constructor of the right class
            assert content in (self.__class__.__name__, "ExtendedTransformation")
        if numexpr is None:
            raise RuntimeError("This Transformation requires the *numexpr* package")
        self.expressions = OrderedDict()

        if dist_expr is not None:
            self.expressions["dist"] = dist_expr
        if poni1_expr is not None:
            self.expressions["poni1"] = poni1_expr
        if poni2_expr is not None:
            self.expressions["poni2"] = poni2_expr
        if rot1_expr is not None:
            self.expressions["rot1"] = rot1_expr
        if rot2_expr is not None:
            self.expressions["rot2"] = rot2_expr
        if rot3_expr is not None:
            self.expressions["rot3"] = rot3_expr
        if wavelength_expr is not None:
            self.expressions["wavelength"] = wavelength_expr
        self.ParamNT = namedtuple("ParamNT", list(self.expressions.keys()))
        self.variables = {"pi": numpy.pi,
                          "hc": CONST_hc,
                          "q": CONST_q}
        if constants is not None:
            self.variables.update(constants)
        self.param_names = tuple(param_names) if param_names is not None else tuple()
        if pos_names is not None:
            self.pos_names = tuple(pos_names)
        else:
            self.pos_names = ("pos",)
        for key in self.param_names + self.pos_names:
            if key in self.variables:
                raise RuntimeError("The keyword %s is already defined, please chose another variable name")
            self.variables[key] = numpy.NaN

        self.codes = OrderedDict(((name, numexpr.NumExpr(expr)) for name, expr in self.expressions.items()))

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
        res = {}
        variables = self.variables.copy()
        for name, value in zip(self.param_names, param):
            variables[name] = value
        if len(self.pos_names) == 1:
            variables[self.pos_names[0]] = pos
        else:
            for name, value in zip(self.pos_names, pos):
                variables[name] = value
        for name, code in self.codes.items():
            signa = [variables.get(name, numpy.NaN) for name in code.input_names]
            res[name] = (float(code(*signa)))
            # could ne done in a single liner but harder to understand !
        return self.ParamNT(**res)

    def __repr__(self):
        res = ["%s with param: %s and pos: %s" % (self.__class__.__name__, self.param_names, self.pos_names), ]
        for name, expr in self.expressions.items():
            res.append("    %s= %s" % (name, expr))
        return os.linesep.join(res)

    def to_dict(self):
        """Export the instance representation for serialization as a dictionary
        """
        res = OrderedDict([("content", self.__class__.__name__),
                           ("param_names", self.param_names),
                           ("pos_names", self.pos_names),
                           ])
        for name, expr in self.expressions.items():
            res[name + "_expr"] = expr
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

    _file_version_1_1 = "Goniometer calibration v1.1"

    file_version = "Goniometer calibration v2"

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
        params = {"detector": self.detector,
                  "wavelength": self.wavelength}
        for name, value in zip(res._fields, res):
            params[name] = value
        return AzimuthalIntegrator(**params)

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

        dico["detector"] = self.detector.name
        dico["detector_config"] = self.detector.get_config()

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
    def _get_detector_from_dict(cls, dico):
        file_version = dico["content"]
        if file_version == cls._file_version_1_1:
            # v1.1
            # Try to extract useful keys
            detector = Detector.factory(dico["detector"])
            # This is not accurate, some keys could be missing
            keys = detector.get_config().keys()
            config = {}
            for k in keys:
                if k in dico:
                    config[k] = dico[k]
                    del dico[k]
            detector = Detector.factory(dico["detector"], config)
        else:
            # v2
            detector = Detector.factory(dico["detector"], dico.get("detector_config", None))
        return detector

    @classmethod
    def sload(cls, filename):
        """Class method for instanciating a Goniometer object from a JSON file

        :param filename: name of the JSON file
        :return: Goniometer object
        """

        with open(filename) as f:
            dico = json.load(f)
        assert "trans_function" in dico, "No translation function defined in JSON file"
        file_version = dico["content"]
        assert file_version in [cls.file_version, cls._file_version_1_1], "JSON file contains a goniometer calibration"
        detector = cls._get_detector_from_dict(dico)
        tansfun = dico.get("trans_function", {})
        if "content" in tansfun:
            content = tansfun.pop("content")
            # May be adapted for other classes of GeometryTransformation functions
            if content in ("GeometryTranslation", "GeometryTransformation"):
                funct = GeometryTransformation(**tansfun)
            elif content == "ExtendedTranformation":
                funct = ExtendedTransformation(**tansfun)
            else:
                raise RuntimeError("content= %s, not in in (GeometryTranslation, GeometryTransformation, ExtendedTranformation)")
        else:  # assume GeometryTransformation
            funct = GeometryTransformation(**tansfun)

        gonio = cls(param=dico.get("param", []),
                    trans_function=funct,
                    detector=detector,
                    wavelength=dico.get("wavelength"))
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
        if "max_shape" in dict_geo:
            # not used in constructor
            dict_geo.pop("max_shape")
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

        ms = marchingsquares.MarchingSquaresMergeImpl(ttha,
                                                      mask=self.geometry_refinement.detector.mask,
                                                      use_minmax_cache=True)
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
                points = ms.find_pixels(tth[i])
                seeds = set((i[0], i[1]) for i in points if mask2[i[0], i[1]])
                # max number of points: 360 points for a full circle
                azimuthal = chia[points[:, 0].clip(0, shape[0]), points[:, 1].clip(0, shape[1])]
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
        geo = self.geometry_refinement.getPyFAI()
        geo["detector"] = self.detector
        return AzimuthalIntegrator(**geo)


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
                            6/7 parameters [dist, poni1, poni2, rot1, rot2, rot3, wavelength]
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
        self.fit_wavelength = "wavelength" in self.trans_function.codes

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
        name = self.__class__.__name__
        count = len(self.single_geometries)
        geometry_list = ", ".join(self.single_geometries.keys())
        return "%s with %i geometries labeled: %s." % (name, count, geometry_list)

    def residu2(self, param):
        "Actually performs the calulation of the average of the error squared"
        sumsquare = 0.0
        npt = 0
        for single in self.single_geometries.values():
            motor_pos = single.get_position()
            single_param = self.trans_function(param, motor_pos)._asdict()
            pyFAI_param = [single_param.get(name, 0.0)
                           for name in ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]]
            pyFAI_param.append(single_param.get("wavelength", self.wavelength) * 1e10)
            if (single.geometry_refinement is not None) and (len(single.geometry_refinement.data) >= 1):
                sumsquare += single.geometry_refinement.chi2_wavelength(pyFAI_param)
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
        if method.lower() in ["simplex", "nelder-mead"]:
            method = "Nelder-Mead"
            bounds = None
        else:
            bounds = self.bounds
        former_error = self.chi2()
        print("Cost function before refinement: %s" % former_error)
        param = numpy.asarray(self.param, dtype=numpy.float64)
        print(param)
        res = minimize(self.residu2, param, method=method,
                       bounds=bounds, tol=1e-12,
                       options=options)
        print(res)
        newparam = res.x
        new_error = res.fun
        print("Cost function after refinement: %s" % new_error)
        print(self.nt_param(*newparam))

        # print("Constrained Least square %s --> %s" % (former_error, new_error))
        if new_error < former_error:
            # print(param, newparam)

            i = abs(param - newparam).argmax()
            if "_fields" in dir(self.nt_param):
                name = self.nt_param._fields[i]
                print("maxdelta on: %s (%i) %s --> %s" % (name, i, self.param[i], newparam[i]))
            else:
                print("maxdelta on: %i %s --> %s" % (i, self.param[i], newparam[i]))
            self.param = newparam
            # update wavelength after successful optimization: not easy
            # if self.fit_wavelength:
            #     self.wavelength = self.
        elif self.fit_wavelength:
            print("Restore wavelength and former parameters")
            former_wavelength = self.wavelength
            for sg in self.single_geometries.values():
                sg.calibrant.setWavelength_change2th(former_wavelength)
            print(self.nt_param(*self.param))
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

    @classmethod
    def sload(cls, filename, pos_function=None):
        """Class method for instanciating a Goniometer object from a JSON file

        :param filename: name of the JSON file
        :param pos_function: a function taking metadata and extracting the
                    goniometer position
        :return: Goniometer object
        """

        with open(filename) as f:
            dico = json.load(f)
        assert dico["content"] == cls.file_version, "JSON file contains a goniometer calibration"
        assert "trans_function" in dico, "No translation function defined in JSON file"
        detector = cls._get_detector_from_dict(dico)
        tansfun = dico.get("trans_function", {})
        if "content" in tansfun:
            content = tansfun.pop("content")
            # May be adapted for other classes of GeometryTransformation functions
            if content in ("GeometryTranslation", "GeometryTransformation"):
                funct = GeometryTransformation(**tansfun)
            elif content == "ExtendedTranformation":
                funct = ExtendedTransformation(**tansfun)
            else:
                raise RuntimeError("content= %s, not in in (GeometryTranslation, GeometryTransformation, ExtendedTranformation)")
        else:  # assume GeometryTransformation
            funct = GeometryTransformation(**tansfun)

        gonio = cls(param=dico.get("param", []),
                    trans_function=funct,
                    pos_function=pos_function,
                    detector=detector,
                    wavelength=dico.get("wavelength"))
        return gonio
