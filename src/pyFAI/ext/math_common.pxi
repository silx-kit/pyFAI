# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""Common cdef functions for math power calculation with integrers

"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "06/03/2023"
__status__ = "stable"
__license__ = "MIT"

from .shared_types cimport int8_t, uint8_t, int16_t, uint16_t, \
                           int32_t, uint32_t, int64_t, uint64_t,\
                           float32_t, float64_t, floating, any_int_t, any_t


cdef inline any_t pow2(any_t x) noexcept nogil:
    return x*x


cdef inline any_t pown(any_t x, unsigned int n) noexcept nogil:
    cdef any_t result = 1
    if n==0:
        return result
    elif n==1:
        return x
    elif n==2:
        return x*x
    while n:
        result *= x
        n-=1
    return result
