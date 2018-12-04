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
__date__ = "04/12/2018"
__status__ = "development"

from collections import OrderedDict, namedtuple
Method = namedtuple("Method", ["dim", "algo", "impl", "split", "target"])


class IntegrationMethod:
    "Keeps track of all integration methods"
    _registry = OrderedDict()

    @classmethod
    def available(cls):
        """return a list of pretty printed integration method available"""
        return [i.__repr__() for i in cls._registry.values()]

    @classmethod
    def select_method(cls, dim, algo=None, impl=None, split=None):
        """Retrieve all algorithm which are fitting the requirement
        """
        # TODO
        pass

    @classmethod
    def select_old_method(cls, dim, old_method):
        """Retrieve all algorithm which are fitting the requirement from old_method
        """
        # TODO
        pass

    def __init__(self, dim, algo, impl, split, target=None, target_name=None,
                 class_=None, function=None, old_method=None, extra=None):
        """Constructor of the class, only registers the 
        :param dim: 1 or 2 integration engine
        :param algo: "histogram" for direct integration, "sparse" for LUT or CSR
        :param impl: "numpy", "scipy", "cython" or "opencl" to describe the implementation
        :param split: pixel splitting options "no", "BBox", "pseudo", "full"
        :param target: the OpenCL device as 2-tuple of indices
        :param target_name: Full name of the OpenCL device
        :param class_: class used to instanciate
        :param function: function to be called
        :param old_method: former method name (legacy)
        :param extra: extra informations
        """
        self.dimension = dim
        self.algorithm = algo
        self.pixel_splitting = split
        self.implementation = impl
        self.target = target
        self.target_name = target_name or str(target)
        self.class_ = class_
        self.function = function
        self.old_method_name = old_method
        self.extra = extra
        self.method = Method(dim, algo.lower(), impl.lower(), split.lower(), target)
        self.__class__._registry[self.method] = self

    def __repr__(self):
        if self.target:
            return ", ".join((str(self.dimension) + "d", self.implementation, self.pixel_splitting + " split", self.algorithm, self.target_name))
        else:
            return ", ".join((str(self.dimension) + "d", self.implementation, self.pixel_splitting + " split", self.algorithm))
