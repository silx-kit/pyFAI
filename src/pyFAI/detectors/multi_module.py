# !/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2026 European Synchrotron Radiation Facility, Grenoble, France
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
#

"""Multi-module detectors:

This module contains some helper function to define a detector from several modules
and later-on refine this module position from powder diffraction data
as demonstrated in https://doi.org/10.3390/cryst12020255
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/09/2026"
__status__ = "development"

import copy
import logging
from dataclasses import dataclass
from math import cos, pi, sin

import numpy
from scipy import ndimage, optimize

from ..control_points import ControlPoints
from ..ext import _geometry
from ..io.ponifile import PoniFile
from ..third_party.classproperties import classproperty
from ._common import Detector

logger = logging.getLogger(__name__)

module_d = numpy.dtype(
    [
        ("d0", numpy.float64),
        ("d1", numpy.float64),
        ("ring", numpy.int32),
        ("module", numpy.int32),
    ]
)


# Those are the optimizable parameters ... 2 translations and one rotation.
@dataclass
class ModuleParam:
    d0: float = 0.0
    d1: float = 0.0
    rot: float = 0.0

    def set(self, iterable):
        self.d0, self.d1, self.rot = iterable[:3]

    def get(self):
        return (self.d0, self.d1, self.rot)

    @classproperty
    def nb_param(cls):
        return len(cls.__dataclass_fields__)


@dataclass
class PoniParam:
    dist: float = 0.0
    poni1: float = 0.0
    poni2: float = 0.0
    rot1: float = 0.0
    rot2: float = 0.0
    # rot3:float=0.0
    # wavelength:float=0.0

    @classproperty
    def nb_param(self):
        return len(self.__dataclass_fields__)


class SingleModule:
    def __init__(self, detector, mask, index=None, fixed=False):
        self.parent_detector = detector
        self.parent_index = index
        if (index is not None) and index <= mask.max():
            self.mask = mask == index
        else:
            self.mask = mask
        self.fixed = fixed
        self.param = ModuleParam()
        self.center = None
        self.bounding_box = None
        self.calc_bounding_box()

    def __repr__(self):
        return (
            f"Module centered at ({self.center[0, 0]:.1f}, {self.center[1, 0]:.1f})"
            + (", fixed." if self.fixed else ".")
        )

    def calc_bounding_box(self):
        d0, d1 = numpy.where(self.mask)
        d0m = d0.min()
        d0M = d0.max()
        d1m = d1.min()
        d1M = d1.max()
        self.center = numpy.atleast_2d([0.5 * (d0M + d0m + 1), 0.5 * (d1M + d1m + 1)]).T
        self.bounding_box = (slice(d0m, d0M + 1), slice(d1m, d1M + 1))
        return self.bounding_box

    def calc_displacement_map(self, d1=None, d2=None, param=None):
        if d1 is None and d2 is None:
            full_detector = True
            p1, p2, _ = self.parent_detector.calc_cartesian_positions()
            d1 = p1 / self.parent_detector.pixel1
            d2 = p2 / self.parent_detector.pixel2
            mp1 = d1[self.mask]
            mp2 = d2[self.mask]
        else:
            full_detector = False
            mp1 = d1
            mp2 = d2

        param = param or self.param

        mpc = numpy.vstack((mp1.ravel(), mp2.ravel()))
        if not self.fixed:
            self.center
            mpc -= self.center
            rot = param.rot
            c, s = cos(rot), sin(rot)
            rotm = numpy.array([[c, -s], [s, c]])
            mpc = (
                numpy.dot(rotm, mpc)
                + self.center
                + numpy.atleast_2d([param.d0, param.d1]).T
            )
        if full_detector:
            mshape = mp1.shape
            p1[self.mask] = mpc[0].reshape(mshape)
            p2[self.mask] = mpc[1].reshape(mshape)
        else:
            p1, p2 = mpc
        return p1, p2

    def calc_position(self, d1=None, d2=None, param=None):
        d1, d2 = self.calc_displacement_map(d1, d2, param)
        return d1 * self.parent_detector.pixel1, d2 * self.parent_detector.pixel2


class MultiModule:
    """Split a detector in several modules"""

    def __init__(self):
        self.modules = {}  # this is contains all of modules
        self.lmask = None
        self.detector = None
        self.nb_modules = 0

    def __repr__(self):
        return f"MultiModule with {self.nb_modules} modules:\n" + "\n".join(
            f"  {i:2d}: {j}" for i, j in self.modules.items()
        )

    def build_labels(self):
        self.lmask, self.nb_modules = ndimage.label(
            numpy.logical_not(self.detector.mask)
        )

    @classmethod
    def from_detector(cls, detector):
        """Alternative constructor

        :param detector: ensure the mask is definied"""
        self = cls()
        if detector.mask is None:
            raise RuntimeError("`detector` must provide an actual mask")
        self.detector = detector
        self.build_labels()
        for module_id in range(1, self.nb_modules + 1):
            self.modules[module_id] = SingleModule(detector, self.lmask, index=module_id, fixed=False)
        return self

    @property
    def shape(self):
        return self.detector.shape

    def calc_displacement_map(self):
        p1, p2, _ = self.detector.calc_cartesian_positions()
        p1 /= self.detector.pixel1
        p2 /= self.detector.pixel2

        for module_id in range(1, self.nb_modules + 1):
            m = self.modules[module_id]
            mp1, mp2 = m.calc_displacement_map()
            p1[m.mask] = mp1[m.mask]
            p2[m.mask] = mp2[m.mask]

        return p1, p2

    @property
    def free_modules(self):
        return sum(not m.fixed for m in self.modules.values())

    def to_detector(self, param=None):
        """Build a detector where every pixel is displaced according to the module positions

        The returned detector has its `_pixel_corners` attribute defined, i.e. it holds the
        actual position of the 4 corners of every pixel, module displacement included.
        Such a detector can be used directly for azimuthal integration, saved as a NeXus
        file with `detector.save(filename)` and read back with
        `pyFAI.detector_factory(filename)`.

        The result is a generic `Detector` and not a copy of the parent one: some detector
        classes (like `Eiger`) overwrite `calc_cartesian_positions` and would silently
        ignore the pixel corners. Pixel size, shape, orientation, mask and sensor are
        inherited from the parent detector.

        Since the module displacement is modeled as a rotation and a translation *within*
        the plane of the detector, the out-of-plane coordinate of the pixel corners is left
        unchanged (it is null for a flat detector).

        :param param: optional vector with the refined parameters of the modules, as
                      returned by `MultiModuleRefinement.refine`. It contains 3 values
                      (d0, d1, rot) per free module; any trailing value, like the
                      poni-parameters, is ignored. When None, the parameters currently
                      stored in each module are used.
        :return: a Detector instance with the `_pixel_corners` attribute defined
        """
        parent = self.detector
        if param is not None:
            expected = ModuleParam.nb_param * self.free_modules
            if len(param) < expected:
                raise ValueError(f"`param` should provide at least {expected} values "
                                 f"({ModuleParam.nb_param} per free module), got {len(param)}")
        pixel1 = parent.pixel1
        pixel2 = parent.pixel2
        # 4D array with, for every pixel, the (z, dim1, dim2) position of its 4 corners
        corners = parent.get_pixel_corners(correct_binning=False).astype(numpy.float64)
        param_idx = 0
        for module_id in sorted(self.modules):
            module = self.modules[module_id]
            if module.fixed:
                # `calc_displacement_map` is the identity for those, skip them
                continue
            if param is None:
                sub_param = None
            else:
                sub_param = ModuleParam(*param[ModuleParam.nb_param * param_idx:
                                               ModuleParam.nb_param * (param_idx + 1)])
            param_idx += 1
            # fancy indexing provides a copy with the shape (nb_pixel, 4, 3)
            sub = corners[module.mask]
            shape = sub.shape[:-1]
            d1, d2 = module.calc_displacement_map(d1=sub[..., 1] / pixel1,
                                                  d2=sub[..., 2] / pixel2,
                                                  param=sub_param)
            sub[..., 1] = d1.reshape(shape) * pixel1
            sub[..., 2] = d2.reshape(shape) * pixel2
            corners[module.mask] = sub
        detector = Detector(pixel1=pixel1, pixel2=pixel2,
                            max_shape=parent.max_shape,
                            orientation=parent.orientation,
                            sensor=copy.deepcopy(parent.sensor))
        detector.mask = parent.mask
        detector.set_pixel_corners(corners)
        return detector


class MultiModuleRefinement(MultiModule):

    LEAST_SQUARES = ("lm", "trf", "dogbox")
    "Optimizers from scipy.optimize.least_squares, they work on the vector of residuals"

    def __init__(self):
        super().__init__()
        self.modulated_points = {}  # key: npt filename, value record array with coordinates, ring & module
        self.calibrants = {}  # contains the different calibrant objects for each control-point file
        self._q_theo = {}
        self.ponis = {}  # geometry of every control-point file, updated by `set_param`
        self.param = None  # current parameter vector, updated by `set_param` and `refine`

    def calc_cp_positions(self, param=None, key=None, center=True):
        """Calculate the physical position for control points of a given registered calibrant"""
        mcp = self.modulated_points[key]
        p1 = mcp.d0.copy()
        p2 = mcp.d1.copy()
        param_idx = 0
        center = 0.5 if center else 0
        for module_id in range(1, self.nb_modules + 1):
            m = self.modules[module_id]
            mask = mcp.module == module_id
            valid = mcp[mask]
            sub_param = (
                None
                if param is None or m.fixed
                else ModuleParam(*param[3 * param_idx : 3 * (param_idx + 1)])
            )
            param_idx += 0 if m.fixed else 1
            mp1, mp2 = m.calc_position(
                d1=valid.d0 + center, d2=valid.d1 + center, param=sub_param
            )
            p1[mask] = mp1
            p2[mask] = mp2
        return p1, p2

    def print_control_points_per_module(self, filename):
        if filename not in self.modulated_points:
            print(f"No control-point file named {filename}. Did you load it ?")
        else:
            print(f"`{filename}`: {self.calibrants.get(filename)}")
            modulated_cp = self.modulated_points[filename]
            w1 = len("Module")
            w2 = len("Points")
            print("+"+"-"*w1+"+"+"-"*w2+"+")
            print("|Module|Points|")
            print("+"+"-"*w1+"+"+"-"*w2+"+")
            for module_id in range(1, self.nb_modules + 1):
                print(f"|{module_id:6d}|{(modulated_cp.module == module_id).sum():6d}|")
            print("+"+"-"*w1+"+"+"-"*w2+"+")

    def load_control_points(self, filename, poni=None, verbose=False):
        """
        :param filename: file with control points
        :param poni: file with the (uncorrected) detector position. When missing, an empty
                     `PoniFile` is registered so that every set of control points always
                     comes with exactly one geometry: the size of the parameter vector is
                     therefore unambiguous. Its parameters start at 0 and have to be
                     refined, which is unlikely to converge from such a starting point.
        :param verbose: set to True to print out the number of control points per module
        """
        cp = ControlPoints(filename)
        self.calibrants[filename] = cp.calibrant
        if poni:
            self.ponis[filename] = PoniFile(poni)
        else:
            logger.warning("No poni-file provided for %s: the geometry starts from scratch",
                           filename)
            self.ponis[filename] = PoniFile()
        # build modulated list of control points
        d0 = []
        d1 = []
        ring = []
        modules = []
        for i in cp.getList():
            d0.append(i[0])
            d1.append(i[1])
            ring.append(i[2])
            modules.append(0)
        modulated_cp = numpy.rec.fromarrays((d0, d1, ring, modules), dtype=module_d)
        linear = numpy.round(modulated_cp.d0).astype(numpy.int32) * self.shape[
            -1
        ] + numpy.round(modulated_cp.d1).astype(numpy.int32)
        modulated_cp.module = self.lmask.ravel()[linear]
        self.modulated_points[filename] = modulated_cp
        if verbose:
            self.print_control_points_per_module(filename)

    def init_q_theo(self, force=False):
        if force or not self._q_theo:
            self._q_theo = {
                key: 20.0
                * pi
                / numpy.array(calibrant.dspacing)[self.modulated_points[key].ring]
                for key, calibrant in self.calibrants.items()
            }

    def residu(self, param=None):
        """Calculate the delta_q value between the expected ring position and the actual one

        :param param: parameter vector, defaults to the current state of the refinement
        :return: vector with the deviation in q (nm^-1) for every control point
        """
        if not self._q_theo:
            self.init_q_theo()
        if param is None:
            param = self.init_param()
        module_param = param[
            : ModuleParam.nb_param * sum(not m.fixed for m in self.modules.values())
        ]
        delta = []
        for idx, (key, calibrant) in enumerate(self.calibrants.items()):
            # print(key)
            tmp_e = self._q_theo[
                key
            ]  # This is the theoritical q_value for the given ring (in nm^-1)
            # print("exp", len(tmp_e), tmp_e)
            dp1, dp2 = self.calc_cp_positions(param=module_param, key=key)
            # print("dp", len(dp1), len(dp2))

            start_idx = (
                ModuleParam.nb_param * self.free_modules + idx * PoniParam.nb_param
            )
            end_idx = start_idx + PoniParam.nb_param
            poni_param = PoniParam(*param[start_idx:end_idx])
            # print(poni_param)
            tmp_c = _geometry.calc_q(
                poni_param.dist,
                poni_param.rot1,
                poni_param.rot2,
                0.0,
                dp1 - poni_param.poni1,
                dp2 - poni_param.poni2,
                calibrant.wavelength,
            )
            # print("residu", tmp_e, tmp_c)
            delta.append(tmp_c - tmp_e)
        return numpy.concatenate(delta)

    @property
    def nb_param(self):
        """Number of parameters for the refinement

        This is 3 values per free module plus 5 per registered geometry. `load_control_points`
        guarantees one geometry per set of control points, hence `self.ponis`,
        `self.calibrants` and `self.modulated_points` always share the same keys.
        """
        free = sum(not m.fixed for m in self.modules.values())
        return free * ModuleParam.nb_param + PoniParam.nb_param * len(self.ponis)

    def init_param(self):
        """Generate the numpy array with all parameters

        The values are read from the current state of the refinement: the displacement
        stored in each free module and the geometry of every registered poni.
        """
        param = numpy.zeros(self.nb_param)
        idx = 0
        for m in self.modules.values():
            if m.fixed:
                continue
            for i, n in enumerate(ModuleParam.__dataclass_fields__, start=idx):
                param[i] = m.param.__getattribute__(n)
            idx += ModuleParam.nb_param
        for p in self.ponis.values():
            for i, n in enumerate(PoniParam.__dataclass_fields__, start=idx):
                value = p.__getattribute__(n)
                # an empty PoniFile has all its parameters set to None
                param[i] = 0.0 if value is None else value
            idx += PoniParam.nb_param
        return param

    def set_param(self, param):
        """Store a parameter vector as the current state of the refinement

        The displacement of every free module is copied into its `ModuleParam` and the
        geometry of every image into its `PoniFile`, so that `init_param` reads the very
        same vector back and so that `to_detector`, `residu`, `cost` or `print_param` use
        it without any argument.

        :param param: vector with all the parameters, as provided by `init_param`
        :return: the parameter vector actually stored
        """
        param = numpy.ascontiguousarray(param, dtype=numpy.float64)
        if param.size != self.nb_param:
            raise ValueError(f"`param` should provide {self.nb_param} values, got {param.size}")
        idx = 0
        for m in self.modules.values():
            if m.fixed:
                continue
            m.param.set(param[idx: idx + ModuleParam.nb_param])
            idx += ModuleParam.nb_param
        for key, poni in self.ponis.items():
            # a PoniFile is not mutable: derive a new one from the refined values
            nb = PoniParam.nb_param
            values = dict(zip(PoniParam.__dataclass_fields__, param[idx: idx + nb]))
            self.ponis[key] = poni.with_params(**values)
            idx += nb
        self.param = param
        return param

    @property
    def param_names(self):
        """Name of every parameter, in the same order as the vector of `init_param`"""
        names = []
        for module_id, module in self.modules.items():
            if module.fixed:
                continue
            names += [f"module{module_id}.{n}" for n in ModuleParam.__dataclass_fields__]
        for key in self.ponis:
            names += [f"{key}.{n}" for n in PoniParam.__dataclass_fields__]
        return names

    def print_param(self, param=None, sigma=None):
        """Display the parameter vector, module per module and geometry per geometry

        :param param: vector with all the parameters, as provided by `init_param`.
                      Defaults to the current state of the refinement.
        :param sigma: optional vector with the standard deviation of every parameter, as
                      provided by `calc_uncertainties`. Displayed after a ± sign.
        """
        param = self.init_param() if param is None else param
        idx = 0
        for i, m in self.modules.items():
            if m.fixed:
                print(f"module #{i:2d}: Fixed")
            else:
                res = f"module #{i:2d}:"
                for i, n in enumerate(ModuleParam.__dataclass_fields__, start=idx):
                    res += (f" {n:5s}= {param[i]}," if sigma is None else
                            f" {n:5s}= {param[i]:9.6f} ± {sigma[i]:8.6f},")
                idx += ModuleParam.nb_param
                print(res)
        for p in self.ponis:
            res = f"{p}:"
            for i, n in enumerate(PoniParam.__dataclass_fields__, start=idx):
                res += (f" {n:5s}= {param[i]:6f}," if sigma is None else
                        f" {n:5s}= {param[i]:9.6f} ± {sigma[i]:8.6f},")
            print(res)
            idx += PoniParam.nb_param

    def cost(self, param=None):
        """Sum of the squared residuals, i.e. the cost function which is minimized

        :param param: parameter vector, defaults to the current state of the refinement
        """
        delta = self.residu(param)
        return numpy.dot(delta, delta)

    def chi2(self, param=None):
        """Average of the squared deviation in q, per control point

        Unlike `cost`, this value does not depend on the number of control points, hence
        it can be compared between datasets. This is the quantity `refine` uses to decide
        whether a refinement is an improvement or not.

        :param param: parameter vector, defaults to the current state of the refinement
        :return: cost function divided by the number of control points
        """
        delta = self.residu(param)
        return numpy.dot(delta, delta) / max(delta.size, 1)

    def refine(self, param=None, method="SLSQP", **kwargs):
        """Refine the position of the modules and the geometry of every image

        Two families of optimizers are available:

        * least-squares optimizers ("lm", "trf" and "dogbox") work on the vector of
          residuals and take advantage of its derivatives. They are much faster and more
          reliable on this kind of problem, where the number of parameters is large. In
          addition, the jacobian they provide allows `calc_uncertainties` to estimate the
          precision of the fit. "lm" (Levenberg-Marquardt) is the algorithm used in the
          publication this module is based on.
        * scalar minimizers from `scipy.optimize.minimize` ("SLSQP", "simplex", ...) only
          see the cost function, i.e. the sum of the squared residuals.

        Like `GeometryRefinement.refine3`, the chi² is evaluated before and after the
        optimization and the result is stored with `set_param` **only if it is an
        improvement**: a refinement can therefore not degrade the current state.

        :param param: vector with the initial guess of the parameters, see `init_param`.
                      Defaults to the current state of the refinement.
        :param method: name of the optimizer, "simplex" is an alias for "Nelder-Mead"
        :param kwargs: any extra keyword argument, passed to the scipy optimizer
        :return: the OptimizeResult object from scipy, with 3 extra keys: `chi2_before`
                 and `chi2_after`, the average squared deviation per control point, and
                 `applied`, whether the result was stored as the new state.
                 Nota: `result.fun` contains the vector of residuals with least-squares
                 optimizers and the value of the cost function with the other ones.
        """
        param = self.init_param() if param is None else param
        chi2_before = self.chi2(param)
        method = "Nelder-Mead" if method.lower() == "simplex" else method
        if method.lower() in self.LEAST_SQUARES:
            result = optimize.least_squares(self.residu, param, method=method.lower(), **kwargs)
        else:
            result = optimize.minimize(self.cost, param, method=method, **kwargs)
        chi2_after = self.chi2(result.x)
        result.chi2_before = chi2_before
        result.chi2_after = chi2_after
        result.applied = chi2_after < chi2_before
        if result.applied:
            names = self.param_names
            idx = numpy.argmax(abs(numpy.asarray(result.x) - param))
            logger.info("Least square %s --> %s", chi2_before, chi2_after)
            logger.info("maxdelta on %s: %s --> %s", names[idx], param[idx], result.x[idx])
            self.set_param(result.x)
        elif chi2_after > chi2_before:
            logger.warning("Refinement rejected: chi² would degrade from %s to %s",
                           chi2_before, chi2_after)
        else:
            logger.info("Refinement did not improve chi² (%s): already converged", chi2_before)
        return result

    def calc_uncertainties(self, result):
        """Estimate the standard deviation of the refined parameters

        The covariance matrix is obtained by inverting $J^tJ$, where $J$ is the jacobian of
        the residuals at the solution, and by scaling it with the variance of the residuals.
        This is the very calculation performed by `scipy.optimize.curve_fit`.

        :param result: OptimizeResult returned by `refine` with a least-squares method
        :return: array with the standard deviation of every parameter, in the same order
                 as `result.x`
        """
        jac = getattr(result, "jac", None)
        if jac is None or jac.ndim != 2:
            raise RuntimeError("`result` should come from a least-squares optimizer, "
                               "i.e. from refine(..., method='lm')")
        npt, nparam = jac.shape
        # Moore-Penrose inverse of J^t.J, discarding the null singular values
        _, sval, vt = numpy.linalg.svd(jac, full_matrices=False)
        threshold = numpy.finfo(float).eps * max(jac.shape) * sval[0]
        vt = vt[sval > threshold]
        sval = sval[sval > threshold]
        covariance = (vt.T / sval ** 2) @ vt
        if npt > nparam:
            covariance = covariance * 2.0 * result.cost / (npt - nparam)
        else:
            logger.warning("Less residuals than parameters: uncertainties are meaningless")
        return numpy.sqrt(numpy.diag(covariance))
