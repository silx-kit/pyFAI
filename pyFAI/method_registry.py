#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/12/2018"
__status__ = "development"

from logging import getLogger
logger = getLogger(__name__)
from collections import OrderedDict, namedtuple
Method = namedtuple("Method", ["dim", "split", "algo", "impl", "target"])
ClassFunction = namedtuple("ClassFunction", ["klass", "function"])


def _degraded(split, algo, impl):
    "provide a degraded version of the input"
    if impl == "opencl":
        return split, algo, "cython"
    elif algo == "lut":
        return split, "histogram", impl
    elif algo == "csr":
        return split, "histogram", impl
    elif split == "full":
        return "pseudo", algo, impl
    elif split == "full":
        return "pseudo", algo, impl
    elif split == "pseudo":
        return "bbox", algo, impl
    elif split == "bbox":
        return "no", algo, impl
    elif impl == "cython":
        return split, algo, "python"
    else:
        # Totally fail safe ?
        return "no", "histogram", "python"


class IntegrationMethod:
    "Keeps track of all integration methods"
    _registry = OrderedDict()

    @classmethod
    def list_available(cls):
        """return a list of pretty printed integration method available"""
        return [i.__repr__() for i in cls._registry.values()]

    @classmethod
    def select_method(cls, dim=None, split=None, algo=None, impl=None,
                      target=None, target_type=None):
        """Retrieve all algorithm which are fitting the requirement
        """
        dim = int(dim) if dim else 0
        algo = algo.lower() if algo is not None else "*"
        impl = impl.lower() if impl is not None else "*"
        split = split.lower() if split is not None else "*"
        target_type = target_type.lower() if target_type else "*"
        if target_type in ("any", "all"):
            target_type = "*"
        method_nt = Method(dim, algo, impl, split, target)
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
        while not res:
            newsplit, newalgo, newimpl = _degraded(split, algo, impl)
            logger.info("Degrading method from (%s,%s,%s) -> (%s,%s,%s)",
                        split, algo, impl, newsplit, newalgo, newimpl)
            split, algo, impl = newsplit, newalgo, newimpl
            res = cls.select_method(dim, split, algo, impl)
        return res

    @classmethod
    def select_old_method(cls, dim, old_method):
        """Retrieve all algorithm which are fitting the requirement from old_method
        valid 
        "numpy", "cython", "bbox" or "splitpixel", "lut", "csr", "nosplit_csr", "full_csr", "lut_ocl" and "csr_ocl"
        """
        results = []
        for v in cls._registry.values():
            if (v.dimension == dim) and (v.old_method_name == old_method):
                results.append(v)
        if results:
            return results
        dim = int(dim)
        algo = "*"
        impl = "*"
        split = "*"
        old_method = old_method.lower()
        if "lut" in old_method:
            algo = "lut"
        elif "csr" in old_method:
            algo = "csr"

        if "ocl" in old_method:
            impl = "opencl"

        if "bbox" in old_method:
            split = "bbox"
        elif "full" in old_method:
            split = "full"
        elif "no" in old_method:
            split = "no"
        return cls.select_method(dim, split, algo, impl)

    @classmethod
    def is_available(cls, dim, split=None, algo=None, impl=None, method_nt=None):
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
            method_nt = Method(dim, split, algo, impl, target)

        return method_nt in cls._registry

    @classmethod
    def parse(cls, smth, dim=1):
        """Parse the string for the content
         
        TODO: parser does not allow to select device
        """
        res = []
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
        if res:
            return res[0]

    def __init__(self, dim, split, algo, impl,
                 target=None, target_name=None, target_type=None,
                 class_funct=None, old_method=None, extra=None):
        """Constructor of the class, only registers the 
        :param dim: 1 or 2 integration engine
        :param split: pixel splitting options "no", "BBox", "pseudo", "full"
        :param algo: "histogram" for direct integration, LUT or CSR for sparse
        :param impl: "python", "cython" or "opencl" to describe the implementation
        :param target: the OpenCL device as 2-tuple of indices
        :param target_name: Full name of the OpenCL device
        :param class_funct: class used and function to be used
        :param old_method: former method name (legacy)
        :param extra: extra informations
        """
        self.dimension = int(dim)
        self.algorithm = str(algo)
        self.algo_lower = self.algorithm.lower()
        self.pixel_splitting = str(split)
        self.split_lower = self.pixel_splitting.lower()
        self.implementation = str(impl)
        self.impl_lower = self.implementation.lower()
        self.target = target
        self.target_name = target_name or str(target)
        self.target_type = target_type
        if class_funct:
            self.class_funct = ClassFunction(*class_funct)
        self.old_method_name = old_method
        self.extra = extra
        self.method = Method(self.dimension, self.split_lower, self.algo_lower, self.impl_lower, target)
        # basic checks ....
        assert self.split_lower in ("no", "bbox", "pseudo", "full")
        assert self.algo_lower in ("histogram", "lut", "csr")
        assert self.impl_lower in ("python", "cython", "opencl")
        self.__class__._registry[self.method] = self

    def __repr__(self):
        if self.target:
            return ", ".join((str(self.dimension) + "d int", self.pixel_splitting + " split", self.algorithm, self.implementation, self.target_name))
        else:
            return ", ".join((str(self.dimension) + "d int", self.pixel_splitting + " split", self.algorithm, self.implementation))
