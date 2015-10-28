#!/usr/bin/env python
# coding: utf-8
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, division, print_function

__doc__ = """small program to transform a metrology CSV file into a detector specification file"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/10/2015"

import os
import sys
import numpy
import pyFAI
import h5py
import logging
import six
logger = logging.getLogger("xpad")
from pyFAI import bilinear


def parse(fname):
    """
    @param fname: name of the CSV definition file
    @return numpy array containing x,y,z coordinates
    """
    res = []
    with open(fname) as fd:
        started = False
        for line in fd:
            if "PlanPads" in line:
                started = True
            if started and line.startswith("P.No.:"):
                one_line = []
                for block in line.split(";"):
                    kv = block.split(":")
                    if (len(kv) == 2):
                        try:
                            one_line.append(float(kv[1]))
                        except Exception as err:
                            logger.error(str(err))
                res.append(one_line)
    return numpy.array(res)


def one_module(p1, p2, dx=80, dy=120, px=1.3e-4, py=1.3e-4):
    """
    @param p1: actual coordinate of the point close to the origin
    @param p2:  actual coordinate of the point close to the end of first line
    @param dx: number of pixel in a line
    @param dy: number of pixel in a column
    @param px: pixel size in x
    @param py: pixel size in y
    @return 2x (dy+1)x(dx+1) array of corner position
    """
    xyz1 = p1[1:4] / 1000.0  # in meter
    xyz2 = p2[1:4] / 1000.0  # in meter
    x = xyz2 - xyz1
    x /= numpy.linalg.norm(x)
    z = numpy.array([0., 0., 1.])
    y = numpy.cross(z, x)
    m = numpy.vstack((x * px, y * py, z * px)).T
    vol_xyz = numpy.zeros((dy + 1, dx + 1, 3))
    vol_xyz[:, :, 0] = numpy.outer(numpy.arange(0, dy + 1), numpy.ones(dx + 1))
    vol_xyz[:, :, 1] = numpy.outer(numpy.ones(dy + 1), numpy.arange(dx + 1))
    n = numpy.dot(vol_xyz, m.T) + xyz1
    return numpy.ascontiguousarray(n[:, :, 0]), numpy.ascontiguousarray(n[:, :, 1]), numpy.ascontiguousarray(n[:, :, 2])


def display(data):
    """
    Display the plan
    @param data: 3d array with coordinates
    """
    from matplotlib import pyplot
    fig = pyplot.figure()
    fig.show()
    xy = fig.add_subplot(2, 2, 1)
    xy.plot(data[:, 1], data[:, 2], "o")
    for txt in data:
        xy.annotate(str(int(txt[0])), (txt[1], txt[2]))
    xy.set_title("xy")
    xz = fig.add_subplot(2, 2, 2)
    xz.plot(data[:, 1], data[:, 3], "o")
    xz.set_title("xz")
    yz = fig.add_subplot(2, 2, 3)
    yz.plot(data[:, 2], data[:, 3], "o")
    yz.set_title("yz")
    z = fig.add_subplot(2, 2, 4)
    z.plot(data[:, 0], data[:, 3],)
    z.set_title("z")
    fig.show()
    six.moves.input()


def build_detector(data):
    """
    """
    det = pyFAI.detectors.Xpad_flat()
    det._pixel_corners = numpy.zeros((det.shape[0], det.shape[1], 4, 3))
    det.uniform_pixel = False
    for j in range(8):
        for i in range(7):
            k = j * 7 + i
            module = bilinear.convert_corner_2D_to_4D(3, *one_module(data[2 * k], data[2 * k + 1]))
            det._pixel_corners[(j * 120):(j + 1) * 120, i * 80:(i + 1) * 80, :, :] = module
    det.save("filename.h5")
    return det

if __name__ == "__main__":
    data = parse(sys.argv[1])
    print(data)
    print(data.shape)
    build_detector(data)
    display(data)
