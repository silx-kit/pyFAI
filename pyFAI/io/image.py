# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2019 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module function to read images.
"""

import os.path
import fabio
import silx.io


def read_image_data(image_path):
    """
    Returns a numpy.array image from a file name or a URL.

    :param str image_path: Path of the image file
    :rtype: numpy.ndarray
    :raises IOError: if the data is not reachable
    :raises TypeError: if the data is not an image (wrong size, wrong dimension)
    """
    if fabio is None:
        raise RuntimeError("FabIO is missing")
    if os.path.exists(image_path):
        with fabio.open(image_path) as image:
            data = image.data
    elif image_path.startswith("silx:") or image_path.startswith("fabio:"):
        data = silx.io.get_data(image_path)
    elif "::" in image_path:
        # Could be a fabio path
        with fabio.open(image_path) as image:
            data = image.data
    else:
        raise IOError("Data from path '%s' is not supported or missing" % image_path)

    if len(data.shape) != 2:
        raise TypeError("Path %s identify a %dd-array, but a 2d is array is expected" % (image_path, len(data.shape)))
    if data.dtype.kind not in "fui":
        raise TypeError("Path %s identify an %s-kind array, but a numerical kind is expected" % (image_path, data.dtype.kind))

    return data
