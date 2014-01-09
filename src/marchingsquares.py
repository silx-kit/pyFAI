#!/usr/bin/python

import sys
import base64
PY3 = sys.version_info[0] == 3
if PY3:
    base64decode = base64.decodebytes
else:
    base64decode = base64.decodestring
import numpy
import marchingsquares_


# Map cell-index to zero or more edge indices
# This first element specifies the number of edge-pairs in the list
# 1 | 2
# ------   -> to see below
# 8 | 4
CONTOUR_CASES = numpy.array([
                        [0, 0, 0, 0, 0], # Case 0: nothing
                        [1, 0, 3, 0, 0], # Case 1
                        [1, 0, 1, 0, 0], # Case 2
                        [1, 1, 3, 0, 0], # Case 3

                        [1, 1, 2, 0, 0], # Case 4
                        [2, 0, 1, 2, 3], # Case 5 > ambiguous
                        [1, 0, 2, 0, 0], # Case 6
                        [1, 2, 3, 0, 0], # Case 7

                        [1, 2, 3, 0, 0], # Case 8
                        [1, 0, 2, 0, 0], # Case 9
                        [2, 0, 3, 1, 2], # Case 10 > ambiguous
                        [1, 1, 2, 0, 0], # Case 11

                        [1, 1, 3, 0, 0], # Case 12
                        [1, 0, 1, 0, 0], # Case 13
                        [1, 0, 3, 0, 0], # Case 14
                        [0, 0, 0, 0, 0], # Case 15
                        ], 'int8')

# Map an edge-index to two relative pixel positions. The edge index
# represents a point that lies somewhere in between these pixels.
# Linear interpolation should be used to determine where it is exactly.
#   0
# 3   1   ->  0x
#   2         xx
# These arrays are used in both isocontour and isosurface algorithm
EDGETORELATIVEPOSX = numpy.array([ [0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0] ], 'int8')
EDGETORELATIVEPOSY = numpy.array([ [0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1] ], 'int8')
EDGETORELATIVEPOSZ = numpy.array([ [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1] ], 'int8')


def isocontour(im, isovalue=None):
    """ isocontour(im, isovalue=None)

    Calculate the iso contours for the given 2D image. If isovalue
    is not given or None, a value between the min and max of the image
    is used.

    Returns a pointset in which each two subsequent points form a line
    piece. This van be best visualized using "vv.plot(result, ls='+')".

    """

    # Check image
    if not isinstance(im, numpy.ndarray) or (im.ndim != 2):
        raise ValueError('im should be a 2D numpy array.')

    # Make sure its 32 bit float
    # todo: also allow bool and uint8 ?
    # if im.dtype != numpy.float32:
    im = numpy.ascontiguousarray(im, numpy.float32)

    # Get isovalue
    if isovalue is None:
        isovalue = 0.5 * (im.min() + im.max())
    isovalue = float(isovalue) # Will raise error if not float-like value given

    # Do the magic!
    data = marchingsquares_.marching_squares(im, isovalue,
                    CONTOUR_CASES, EDGETORELATIVEPOSX, EDGETORELATIVEPOSY)

    # Return as pointset
    return data



if __name__ == "__main__":
    import numpy
    x, y = numpy.ogrid[-10:10:0.1, -10:10:0.1]
    r = numpy.sqrt(x * x + y * y)
    print (isocontour(r, 5))
    print len(isocontour(r, 5))
