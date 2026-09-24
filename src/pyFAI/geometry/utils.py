# !/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Helper functions around the geometry, mostly conversions between the
different detector orientation conventions.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/08/2026"
__status__ = "production"
__docformat__ = 'restructuredtext'

__all__ = ["FLIPPED_AXES", "CORNER_OF_THE_DETECTOR", "detector_corner",
           "convert_orientation"]

FLIPPED_AXES = {1: (True, True),
                2: (True, False),
                3: (False, False),
                4: (False, True)}
"""Which axis is mirrored with respect to the native orientation (3),
as a 2-tuple (slow/dim1, fast/dim2)."""

CORNER_OF_THE_DETECTOR = {(False, False): (0, 0, 0),
                          (True, False): (-1, 0, 1),
                          (False, True): (0, -1, 3),
                          (True, True): (-1, -1, 2)}
"""Which corner of the detector a mirror maps the origin onto, as a 3-tuple
(pixel index along dim1, pixel index along dim2, vertex index), keyed by the
2-tuple of mirrored axes (slow/dim1, fast/dim2).

The vertex numbering of ``get_pixel_corners`` is A=0 at (i, j), B=1 at
(i+1, j), C=2 at (i+1, j+1) and D=3 at (i, j+1), so both the pixel and the
vertex follow the mirrored axes.
"""


def detector_corner(detector, flip1=False, flip2=False):
    """Coordinates of one corner of the detector, in meter and in the frame
    ``poni1``/``poni2`` are expressed in.

    ``poni1`` and ``poni2`` are distances measured from the corner of pixel
    (0, 0), which is the corner this returns when no axis is mirrored. Mirroring
    an axis moves that reference to the opposite corner of the sensor, given by
    :data:`CORNER_OF_THE_DETECTOR`.

    A specific vertex of a specific pixel has to be picked, rather than an
    extremum over a pixel or over the whole array: on a spline-corrected
    detector the pixels are distorted quadrilaterals, so nothing guarantees that
    the vertex which bounds the sensor is also the extreme one along a given
    axis.

    Neither coordinate can be assumed either:

    * the far corner is *not* at ``shape * pixel_size`` as soon as the pixels
      are not all the same size. Modular detectors have wider pixels along the
      module borders: ``Xpad_flat`` reaches 25 mm further than the naive
      product along the slow axis.
    * the corner of pixel (0, 0) is *not* at (0, 0) on a spline-corrected
      detector such as the FReLoN, where it is displaced like any other point
      of the sensor.

    :param detector: detector instance, must have a shape
    :param flip1: True if the slow dimension is mirrored
    :param flip2: True if the fast dimension is mirrored
    :return: 2-tuple of coordinates, along the slow and the fast dimension
    """
    index1, index2, vertex = CORNER_OF_THE_DETECTOR[(bool(flip1), bool(flip2))]
    position = detector.get_pixel_corners(correct_binning=True)[index1, index2, vertex]
    return float(position[1]), float(position[2])


def convert_orientation(parameters, detector, from_orientation, to_orientation):
    """Re-express a geometry so that the *same pixel data* is described with
    another detector orientation.

    Mirroring an axis moves the PONI to the opposite side of the detector and
    reverses the sense of the rotation about the *other* in-plane axis:

    * slow axis mirrored: ``poni1 -> origin1 + opposite1 - poni1`` and ``rot2 -> -rot2``
    * fast axis mirrored: ``poni2 -> origin2 + opposite2 - poni2`` and ``rot1 -> -rot1``

    where ``origin`` is the corner of pixel (0, 0), the point ``poni1``/``poni2``
    are measured from, and ``opposite`` the corner the mirror maps it onto. Both
    come from :func:`detector_corner` and neither is ``shape * pixel_size`` nor
    0 in general. ``dist`` and the wavelength are invariant. ``rot3`` is left
    untouched: a setup which is invariant under ``rot3`` does not constrain it.

    Typical use: reading a calibration performed by a program which flips the
    image, such as *Dioptas* (orientation 2), to process the unflipped data.

    Nota: this converts a geometry meant to describe *unchanged* data. When the
    image itself is flipped as well, the two effects cancel out and the
    parameters are unchanged.

    Nota: mirroring puts the origin of the frame exactly on the corner used, so
    on a spline-corrected detector -- the only kind whose corner of pixel (0, 0)
    does not sit at (0, 0) -- converting there and back does not restore the
    PONI exactly. The small difference is legitimate: the mirrored detector is
    not the same object, only its own corner is at the origin of its own frame.

    :param parameters: dict with at least dist, poni1, poni2, rot1 and rot2
    :param detector: detector instance, used for its pixel corners
    :param from_orientation: orientation the parameters are expressed in, 1 to 4
    :param to_orientation: wanted orientation, 1 to 4
    :return: new dict of parameters, the input is left untouched
    """
    for orientation in (from_orientation, to_orientation):
        if orientation not in FLIPPED_AXES:
            raise ValueError(f"Unsupported orientation {orientation}, expected 1 to 4")
    converted = dict(parameters)
    flip1 = FLIPPED_AXES[from_orientation][0] != FLIPPED_AXES[to_orientation][0]
    flip2 = FLIPPED_AXES[from_orientation][1] != FLIPPED_AXES[to_orientation][1]
    if flip1 or flip2:
        origin = detector_corner(detector)
        opposite = detector_corner(detector, flip1, flip2)
    if flip1:
        converted["poni1"] = origin[0] + opposite[0] - parameters["poni1"]
        converted["rot2"] = -parameters["rot2"]
    if flip2:
        converted["poni2"] = origin[1] + opposite[1] - parameters["poni2"]
        converted["rot1"] = -parameters["rot1"]
    return converted
