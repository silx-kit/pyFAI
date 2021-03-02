# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""This modules contains a function to fit without refinement an ellipse
on a set of points ....
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/02/2021"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
from collections import namedtuple
import numpy
from math import sqrt
logger = logging.getLogger(__name__)

Alignment = namedtuple("Alignment", "points RMSD rotation center_ref center_set matrix")


class Kabsch:
    """This class aligns a set of point on a reference grid and calculate 
    the optimal rotation, translation.
    It offers 2 methods to correct (move to the reference position) or uncorrect other sets of points
    """

    @staticmethod
    def kabsch(reference, points):
        "Static implementation of the algorithm"
        R = numpy.ascontiguousarray(reference, dtype=numpy.float64)
        P = numpy.ascontiguousarray(points, dtype=numpy.float64)
        assert R.shape == P.shape
        size, ndim = R.shape
        centroid_P = P.mean(axis=0)
        centroid_R = R.mean(axis=0)

        centered_P = P - centroid_P
        centered_R = R - centroid_R

        C = numpy.dot(centered_P.T, centered_R)

        V, S, W = numpy.linalg.svd(C)
        if numpy.linalg.det(V) * numpy.linalg.det(W) < 0.0:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        # Create Rotation matrix U
        U = numpy.dot(V, W)
        rotated_P = numpy.dot(centered_P, U)
        aligned_P = rotated_P + centroid_R
        rmsd = sqrt(((centered_R - rotated_P) ** 2).sum() / size)

        angle = 180 * numpy.arctan2(U[1, 0], U[0, 0]) / numpy.pi

        return Alignment(aligned_P, rmsd, angle, centroid_R, centroid_P, U)

    def __init__(self, reference, points):
        """Constructor of the class:
        calculates the transformation to match the list of point to the reference points
        
        :param reference: 2d array like with `n` lines of `d` coordinates. The reference positions 
        :param points: 2d array like with `n` lines of `d` coordinates. The point to match to the reference
        
        The transformation provided is the rigid transformation: 
        `P·U + V` where U is the rotation matix and V the the translation 
        
        `d` is usally 2 for 2D detectors but 3 is very common. More shouldn't be an issue
        """
        R = numpy.ascontiguousarray(reference, dtype=numpy.float64)
        P = numpy.ascontiguousarray(points, dtype=numpy.float64)
        assert R.ndim == 2
        self.ndim = R.shape[1]

        k = self.kabsch(R, P)

        self.rmsd = k.RMSD
        self.rotation = k.matrix
        self.translation = numpy.atleast_2d(R.mean(axis=0)) - numpy.atleast_2d(P.mean(axis=0)).dot(self.rotation)

    def __repr__(self):
        return f"Rigid transformation of angle {self.angle:.3f}° and translation {self.translation}, RMSD={self.rmsd:.6f}"

    @property
    def angle(self):
        "Rotation value in degrees"
        self.rotation
        return -numpy.rad2deg(numpy.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

    def correct(self, points):
        "Rotate the set of points to be aligned with the reference"
        P = numpy.ascontiguousarray(points, dtype=numpy.float64)
        assert P.ndim == 2
        assert P.shape[1] == self.ndim
        return P.dot(self.rotation) + self.translation

    def uncorrect(self, points):
        "Rotate the reference points to the other set"
        P = numpy.ascontiguousarray(points, dtype=numpy.float64)
        assert P.ndim == 2
        assert P.shape[1] == self.ndim
        return (P - self.translation).dot(self.rotation.T)

    @classmethod
    def test(cls, reference, points, verbose=True):
        "Perform a basic self consitancy test"
        k = cls(reference, points)
        if verbose:
            print(k)
            print("correct", k.correct(points) - reference, "RMSD=", sqrt(((k.correct(points) - reference) ** 2).sum() / len(reference)))
            print("uncorct", points - k.uncorrect(reference), "RMSD=", sqrt(((points - k.uncorrect(reference)) ** 2).sum() / len(reference)))
        else:
            assert numpy.isclose(sqrt(((k.correct(points) - reference) ** 2).sum() / len(reference)), k.rmsd)
            assert numpy.isclose(sqrt(((points - k.uncorrect(reference)) ** 2).sum() / len(reference)), k.rmsd)
        return k


if __name__ == "__main__":
    print("Perform some tests")
    print("Test translation")
    Kabsch.test([[1, 2], [2, 3], [1, 4]], [[2, 3], [3, 4], [2, 5]], 0)
    print("Test rotation")
    P = numpy.array([[2, 3], [3, 4], [2, 5]])
    Q = -P
    Kabsch.test(P, Q, 0)
    print("test advanced")
    P = numpy.array([[1, 1], [2, 0], [3, 1], [2, 2]])
    Q = numpy.array([[-1, 1], [0, 2], [-1, 3], [-2, 2]])
    Kabsch.test(P, Q, 0)
    print("test advanced2")
    P = numpy.array([[1, 1], [2, 0], [3, 1], [2, 2]])
    Q = numpy.array([[2, 0], [3, 1], [2, 2], [1, 1]])
    Kabsch.test(P, Q, 0)
