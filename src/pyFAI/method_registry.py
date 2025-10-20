#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Contains a registry of all integrator available
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/10/2025"
__status__ = "development"

import inspect
import copy
from logging import getLogger
from collections import OrderedDict, namedtuple
logger = getLogger(__name__)
ClassFunction = namedtuple("ClassFunction", ["klass", "function"])


class _Nothing(object):
    """Used to identify an unset attribute that we could nullify."""
    pass


class Method(namedtuple("_", ["dim", "split", "algo", "impl", "target"])):

    def degraded(self):
        """Returns a degraded version of this method.

        :rtype: Method"
        """
        if self.impl == "opencl":
            result = Method(self.dim, self.split, self.algo, "cython", None)
        elif self.algo == "lut":
            result = Method(self.dim, self.split, "histogram", self.impl, self.target)
        elif self.algo == "csr":
            result = Method(self.dim, self.split, "histogram", self.impl, self.target)
        elif self.split == "full":
            result = Method(self.dim, "pseudo", self.algo, self.impl, self.target)
#         elif self.split == "full":
#             result = Method(self.dim, "pseudo", self.algo, self.impl, self.target)
        elif self.split == "pseudo":
            result = Method(self.dim, "bbox", self.algo, self.impl, self.target)
        elif self.split == "bbox":
            result = Method(self.dim, "no", self.algo, self.impl, self.target)
        elif self.impl == "cython":
            result = Method(self.dim, self.split, self.algo, "python", None)
        else:
            # Totally fail safe ?
            result = Method(self.dim, "no", "histogram", "python", None)
        return result

    def fixed(self, dim=_Nothing, split=_Nothing, algo=_Nothing, impl=_Nothing, target=_Nothing):
        """
        Returns a method containing this Method data except requested attributes
        set.

        :rtype: Method
        """
        if dim is _Nothing:
            dim = self.dim
        if split is _Nothing:
            split = self.split
        if algo is _Nothing:
            algo = self.algo
        if impl is _Nothing:
            impl = self.impl
        if target is _Nothing:
            target = self.target
        return Method(dim, split, algo, impl, target)

    @staticmethod
    def parsed(string):
        """Returns a Method from string.

        :param str string: A string identifying a method. Like "python", "ocl",
            "ocl_gpu", "ocl_0,0"
        :rtype: Method"
        """
        algo = "*"
        impl = "*"
        split = "*"
        string = string.lower()

        if "lut" in string:
            algo = "lut"
        elif "csr" in string:
            algo = "csr"

        target = None

        if string in ["numpy", "python"]:
            impl = "python"
        elif string == "cython":
            impl = "cython"
        elif "ocl" in string:
            impl = "opencl"
            elements = string.split("_")
            if len(elements) >= 2:
                target_string = elements[-1]
                if target_string == "cpu":
                    target = "cpu"
                elif target_string == "gpu":
                    target = "gpu"
                elif target_string in ["*", "any", "all"]:
                    target = None
                elif "," in target_string:
                    try:
                        values = target_string.split(",")
                        target = int(values[0]), int(values[1])
                    except ValueError:
                        pass

        if "bbox" in string:
            split = "bbox"
        elif "full" in string:
            split = "full"
        elif "no" in string:
            split = "no"

        return Method(None, split, algo, impl, target)


class IntegrationMethod:
    "Keeps track of all integration methods"
    _registry = OrderedDict()

    AVAILABLE_SPLITS = ("no", "bbox", "pseudo", "full")
    AVAILABLE_ALGOS = ("histogram", "lut", "csr", "csc")
    AVAILABLE_IMPLS = ("python", "cython", "opencl")

    @classmethod
    def list_available(cls):
        """return a list of pretty printed integration method available"""
        return [i.__repr__() for i in cls._registry.values()]

    @classmethod
    def select_one_available(cls, method, dim=None, default=None, degradable=False):
        """Select one available method from the requested method.

        :param [str,Method,IntegrationMethod] method: The requested method
        :param [None,int] dim: If specified, override the dim of the method
        :param [None,IntegrationMethod] default: If no method found, return this value
        :param bool degradable: If true, it the request do not have available method,
            it will return method close to it.
        :rtype: [IntegrationMethod,None]
        """
        if method is None:
            return default
        if isinstance(method, IntegrationMethod):
            if dim and method.dimension == dim:
                return method
            else:
                method = method.method
        if isinstance(method, str):  # not elif, prevents change of dim in sigma-clip-legacy
            method = cls.parse(method, dim)
            method = method.method
        elif dim is not None:
            if len(method) == 3:
                split, algo, impl = method
                target = None
            elif len(method) == 4:
                split, algo, impl, target = method
            else:
                _dim, split, algo, impl, target = method
            algo = algo.lower()
            if algo.startswith("histo"):
                algo = "histogram"
            method = Method(dim, split.lower(), algo, impl.lower(), target)
        methods = cls.select_method(method=method, degradable=degradable)
        if len(methods) == 0:
            return default
        return methods[0]

    @classmethod
    def select_method(cls, dim=None, split=None, algo=None, impl=None,
                      target=None, target_type=None, degradable=True, method=None):
        """Retrieve all algorithm which are fitting the requirement:

        :param int dim:
        :param str split:

        :return: list of compatible methods or None
        """
        if method is not None:
            dim, split, algo, impl, target = method[:5]
            if isinstance(target, (list, tuple)):
                target, target_type = tuple(target), None
            else:
                target, target_type = None, method.target
            return cls.select_method(dim, split, algo, impl,
                                     target, target_type,
                                     degradable=degradable)

        any_values = set(["any", "all", "*"])
        if dim in any_values:
            methods = []
            for d in [1, 2]:
                methods += cls.select_method(dim=d,
                                             split=split, algo=algo, impl=impl,
                                             target=target, target_type=target_type,
                                             degradable=degradable, method=method)
            return methods

        dim = int(dim) if dim else 0
        algo = algo.lower() if algo is not None else "*"
        impl = impl.lower() if impl is not None else "*"
        split = split.lower() if split is not None else "*"
        target_type = target_type.lower() if target_type else "*"
        if target_type in any_values:
            target_type = "*"
        if isinstance(target, list):
            target = tuple(target)
        method_nt = Method(dim, split, algo, impl, target)
        if method_nt in cls._registry:
            return [cls._registry[method_nt]]
        # Validate on pixel splitting, implementation and algorithm
        if dim:
            candidates = [i for i in cls._registry.keys() if i[0] == dim]
        else:
            candidates = cls._registry.keys()
        if split != "*":
            candidates = [i for i in candidates if i[1] == split]
        if algo != "*":
            candidates = [i for i in candidates if i[2] == algo]
        if impl != "*":
            candidates = [i for i in candidates if i[3] == impl]
        if target:
            candidates = [i for i in candidates if i[4] == target]
        if target_type != "*":
            candidates = [i for i in candidates
                          if cls._registry[i].target_type == target_type]

        res = [cls._registry[i] for i in candidates]
        if degradable:
            while not res:
                new_method = method_nt.degraded()
                if new_method == method_nt:
                    break
                logger.info("Degrading method from %s -> %s", method_nt, new_method)
                method_nt = new_method
                res = cls.select_method(method=new_method)
        return res

    @staticmethod
    def parse_old_method(old_method):
        """
        :rtype: Method
        """
        return Method.parsed(old_method)

    @classmethod
    def select_old_method(cls, dim, old_method):
        """Retrieve all algorithms which are fitting the requirements from
        old_method. Valid input are "numpy", "cython", "bbox" or "splitpixel",
        "lut", "csr", "nosplit_csr", "full_csr", "lut_ocl" and "csr_ocl".
        """
        results = []
        for v in cls._registry.values():
            if (v.dimension == dim) and (v.old_method_name == old_method):
                results.append(v)
        if results:
            return results
        dim = int(dim)
        method = cls.parse_old_method(old_method)
        _, split, algo, impl, target = method
        if target in ["cpu", "gpu", None]:
            target_type = target
            target = None
        else:
            target_type = None

        return cls.select_method(dim, split, algo, impl, target_type=target_type, target=target)

    @classmethod
    def is_available(cls, dim=None, split=None, algo=None, impl=None, method_nt=None):
        """
        Check if the method is currently available

        :param dim: 1 or 2D integration
        :param split: pixel splitting options "no", "BBox", "pseudo", "full"
        :param algo: "histogram" for direct integration, LUT or CSR for sparse
        :param impl: "python", "cython" or "opencl" to describe the implementation
        :param method_nt: a Method namedtuple with (split, algo, impl)
        :return: True if such integrator exists
        """
        if method_nt is None:
            algo = algo.lower() if algo is not None else ""
            impl = impl.lower() if impl is not None else ""
            split = split.lower() if split is not None else ""
            if impl == "opencl":
                # indexes start at 0 hence ...
                target = (0, 0)
            else:
                target = None
            method_nt = Method(dim, split, algo, impl, target)

        return method_nt in cls._registry

    @classmethod
    def parse(cls, smth, dim=1):
        """Parse the string/list/tuple/dict for the content

        :param smth: something
        :param int dim: dimensionality of integrator
        :return: one method fitting the requirement or the default method
        """
        res = []
        weighted = True
        if isinstance(smth, cls):
            return smth
        if isinstance(smth, Method) and cls.is_available(method_nt=smth):
            return cls._registry[smth]
        if isinstance(smth, str):
            comacount = smth.count(",")
            if comacount <= 1:
                res = cls.select_old_method(dim, smth)
            else:
                res = cls.select_method(dim, *[i.split()[0] for i in (smth.split(","))])
        elif isinstance(smth, (list, tuple, dict)):
            target = None

            if isinstance(smth, dict):
                split = smth.get("split") or smth.get("pixel_splitting")
                algo = smth.get("algo") or smth.get("algorithm")
                impl = smth.get("impl") or smth.get("implementation")
            else:
                if len(smth) >= 3:
                    split, algo, impl = smth[:3]
                if len(smth) >= 4 and impl == "opencl":
                    target = smth[3]
                if len(smth) == 5:
                    weighted = bool(smth[4])
            split = split.lower() if split is not None else "*"
            algo = algo.lower() if algo is not None else "*"
            impl = impl.lower() if impl is not None else "*"
            res = cls.select_method(dim, split, algo, impl, target)

        if res:
            result = res[0]
        else:
            result = None
        # TODO after other #1998 have been merged
        # else:
        #   if dim == 1:
            #     result = default 1D
            # else
            #     result = default 2D
        if weighted:
            return result
        else:
            return result.unweighted

    def __init__(self, dim, split, algo, impl,
                 target=None, target_name=None, target_type=None,
                 class_funct_legacy=None, class_funct_ng=None,
                 old_method=None, extra=None):
        r"""Constructor of the class, registering the methods.

        ⚠ DO NOT INSTANCIATE THIS CLASS ... IT MAY INTERFER WITH PYFAI ⚠


        :param dim: 1 or 2 integration engine
        :param split: pixel splitting options "no", "BBox", "pseudo", "full"
        :param algo: "histogram" for direct integration, LUT or CSR for sparse
        :param impl: "python", "cython" or "opencl" to describe the implementation
        :param target: the OpenCL device as 2-tuple of indices
        :param target_name: Full name of the OpenCL device
        :param class_funct_legacy: class used and function to be used for legacy integrator
        :param class_funct_ng: class used and function to be used for new generation integrator
        :param old_method: former method name (legacy)
        :param extra: extra 
        """
        self.__dimension = int(dim)
        self.__algorithm = str(algo)
        self.__pixel_splitting = str(split)
        self.__implementation = str(impl)
        self.__target = target
        self.__target_name = target_name or str(target)
        self.__target_type = target_type
        if class_funct_legacy:
            self.__class_funct_legacy = ClassFunction(*class_funct_legacy)
        else:
            self.__class_funct_legacy = None
        if class_funct_ng:
            self.__class_funct_ng = ClassFunction(*class_funct_ng)
        else:
            self.__class_funct_ng = None
        self.__old_method_name = old_method
        self.extra = extra
        self._weighted_average = True  # this one is mutable
        self.__method = Method(self.dimension, self.split_lower, self.algo_lower, self.impl_lower, target)
        self.__manage_variance = self._does_manage_variance()
        # finally register
        self._register()

    def __repr__(self):
        if self.target:
            string = ", ".join((str(self.dimension) + "d int", self.pixel_splitting + " split", self.algorithm, self.implementation, self.target_name))
        else:
            string = ", ".join((str(self.dimension) + "d int", self.pixel_splitting + " split", self.algorithm, self.implementation))
        return "IntegrationMethod(%s)" % string

    def __hash__(self):
        """Make it independent from weighted"""
        return self.__method.__hash__()

    def __eq__(self, other):
        """Make it independent from weighted"""
        if isinstance(other, self.__class__):
            return self.__method == other.method
        elif isinstance(other, Method):
            return self.__method == other
        else:
            return False

    def _does_manage_variance(self):
        "Checks if the method handles alone the error_model"
        manage_variance = False
        if self.class_funct_ng and self.class_funct_ng.function:
            function = self.class_funct_ng.function
            sig = inspect.signature(function)
            manage_variance = ("poissonian" in sig.parameters) or ("error_model" in sig.parameters)
        return manage_variance

    def _register(self):
        """basic checks before registering the method"""
        if self.split_lower not in self.AVAILABLE_SPLITS:
            raise RuntimeError("Unknown splitting scheme")
        if self.algo_lower not in self.AVAILABLE_ALGOS:
            raise RuntimeError("Unknown algorithm")
        if self.impl_lower not in self.AVAILABLE_IMPLS:
            raise RuntimeError("Unknown implementation")
        self.__class__._registry[self.method] = self

    @property
    def algo_is_sparse(self):
        return self.algo_lower in self.AVAILABLE_ALGOS[1:]

    @property
    def dimension(self):
        return self.__dimension

    @property
    def dim(self):
        return self.__dimension

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def algo(self):
        return self.__algorithm

    @property
    def pixel_splitting(self):
        return self.__pixel_splitting

    @property
    def split(self):
        return self.__pixel_splitting

    @property
    def implementation(self):
        return self.__implementation

    @property
    def impl(self):
        return self.__implementation

    @property
    def algo_lower(self):
        return self.__algorithm.lower()

    @property
    def split_lower(self):
        return self.__pixel_splitting.lower()

    @property
    def impl_lower(self):
        return self.implementation.lower()

    @property
    def target(self):
        return self.__target

    @property
    def target_name(self):
        return self.__target_name

    @property
    def target_type(self):
        return self.__target_type

    @property
    def class_funct_legacy(self):
        return self.__class_funct_legacy

    @property
    def class_funct_ng(self):
        return self.__class_funct_ng

    @property
    def old_method_name(self):
        return self.__old_method_name

    @property
    def method(self):
        return self.__method

    @property
    def manage_variance(self):
        return self.__manage_variance

    @property
    def weighted_average(self):
        return self._weighted_average

    @weighted_average.setter
    def weighted_average(self, value):
        self._weighted_average = bool(value)

    @property
    def weighted(self):
        "return the weighted version"
        cpy = copy.copy(self)
        cpy._weighted_average = True
        return cpy

    @property
    def unweighted(self):
        "return the unweighted version"
        cpy = copy.copy(self)
        cpy._weighted_average = False
        return cpy
