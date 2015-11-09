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
__date__ = "09/11/2015"

import os
import sys
import numpy
import pyFAI
import h5py
import logging
import six
logger = logging.getLogger("xpad")
from pyFAI import bilinear

pix = 130e-6
dx = 80
dy = 120
nx = 7
ny = 8


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


def one_module(p1, p2, flat=False):
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
    if flat:
        xyz1[2] = 0
        xyz2[2] = 0
    x = xyz2 - xyz1
    x /= numpy.linalg.norm(x)
    z = numpy.array([0., 0., 1.])
    y = numpy.cross(z, x)
    z = numpy.cross(x, y)
    m = pix * numpy.vstack((x, y, z))
    vol_xyz = numpy.zeros((dy + 1, dx + 1, 3))
    vol_xyz[:, :, 1] = numpy.outer(numpy.arange(0, dy + 1), numpy.ones(dx + 1))
    vol_xyz[:, :, 0] = numpy.outer(numpy.ones(dy + 1), numpy.arange(dx + 1))
    n = numpy.dot(vol_xyz, m) + xyz1
    return numpy.ascontiguousarray(n[:, :, 1]), numpy.ascontiguousarray(n[:, :, 0]), numpy.ascontiguousarray(n[:, :, 2])


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


def build_detector(data, filename="filename.h5", flat=False):
    """
    """
    det = pyFAI.detectors.Xpad_flat()
    det._pixel_corners = numpy.zeros((det.shape[0], det.shape[1], 4, 3), dtype="float32")
    det.uniform_pixel = False
    det.IS_FLAT = flat
    det._pixel1 = pix
    det._pixel2 = pix
    det.mask = None
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            module = bilinear.convert_corner_2D_to_4D(3, *one_module(data[2 * k], data[2 * k + 1], flat))
            det._pixel_corners[(j * dy):(j + 1) * dy, i * dx:(i + 1) * dx, :, :] = module
    det.save(filename)
    return det


def validate(det, ref="d007_new.h5"):
    """
    """
    if os.path.exists(ref):
        refc = pyFAI.detectors.NexusDetector(ref).get_pixel_corners()
    else:
        refc = None
    from matplotlib import pyplot

    newc = det.get_pixel_corners()
    fig = pyplot.figure()

    p0z = fig.add_subplot(2, 3, 1)
    p0z.plot(newc[:, 0, 0, 0], label="new")
    p0z.set_title("dim1_z")

    p0y = fig.add_subplot(2, 3, 2)
    p0y.plot(newc[:, 0, 0, 1], label="new")
    p0y.set_title("dim1_y")

    p0x = fig.add_subplot(2, 3, 3)
    p0x.plot(newc[:, 0, 0, 2], label="new")
    p0x.set_title("dim1_x")

    p1z = fig.add_subplot(2, 3, 4)
    p1z.plot(newc[0, :, 0, 0], label="new")
    p1z.set_title("dim2_z")

    p1y = fig.add_subplot(2, 3, 5)
    p1y.plot(newc[0, :, 0, 1], label="new")
    p1y.set_title("dim2_y")

    p1x = fig.add_subplot(2, 3, 6)
    p1x.plot(newc[0, :, 0, 2], label="new")
    p1x.set_title("dim2_x")

    if refc is not None:
        p0z.plot(refc[:, 0, 0, 0], label="ref")
        p0y.plot(refc[:, 0, 0, 1], label="ref")
        p0x.plot(refc[:, 0, 0, 2], label="ref")
        p1z.plot(refc[0, :, 0, 0], label="ref")
        p1y.plot(refc[0, :, 0, 1], label="ref")
        p1x.plot(refc[0, :, 0, 2], label="ref")
    p0z.legend()
    p0y.legend()
    p0x.legend()
    p1y.legend()
    p1x.legend()
    p1z.legend()

    fig.show()
    six.moves.input()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-f", "--flat", dest="flat", default=False, action="store_true",
                        help="enforce the detector to be flat")
    parser.add_argument("args", metavar='FILE', type=str, nargs='1',
                         help="Metrology file to be processed (.csv)")
    args = parser.parse_args()
    data = parse(args.args)
    print(data)
    print(data.shape)
    det = build_detector(data, os.path.splitext(sys.argv[1])[0] + ".h5", flat=args.flat)

    validate(det)
    display(data)
