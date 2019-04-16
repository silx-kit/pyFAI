#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2003-2018 European Synchrotron Radiation Facility, Grenoble,
#             France
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

"""

Utilities, mainly for image treatment

"""

__authors__ = ["Jérôme Kieffer", "Valentin Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/02/2019"
__status__ = "production"

import logging
import numpy
import fabio
import weakref
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize.optimize import fmin
from scipy.optimize.optimize import fminbound

from .third_party import six
from .utils import stringutil
from .utils import header_utils

from ._version import calc_hexversion
if ("hexversion" not in dir(fabio)) or (fabio.hexversion < calc_hexversion(0, 4, 0, "dev", 5)):
    # Short cut fabio.factory do not exists on older versions
    fabio.factory = fabio.fabioimage.FabioImage.factory

logger = logging.getLogger(__name__)


class ImageReductionFilter(object):
    """
    Generic filter applied in a set of images.
    """

    def init(self, max_images=None):
        """
        Initialize the filter before using it.

        :param int max_images: Max images supported by the filter
        """
        pass

    def add_image(self, image):
        """
        Add an image to the filter.

        :param numpy.ndarray image: image to add
        """
        raise NotImplementedError()

    def get_parameters(self):
        """Return a dictionary containing filter parameters

        :rtype: dict
        """
        return {"cutoff": None, "quantiles": None}

    def get_result(self):
        """
        Get the result of the filter.

        :return: result filter
        """
        raise NotImplementedError()


class ImageAccumulatorFilter(ImageReductionFilter):
    """
    Filter applied in a set of images in which it is possible
    to reduce data step by step into a single merged image.
    """

    def init(self, max_images=None):
        self._count = 0
        self._accumulated_image = None

    def add_image(self, image):
        """
        Add an image to the filter.

        :param numpy.ndarray image: image to add
        """
        self._accumulated_image = self._accumulate(self._accumulated_image, image)
        self._count += 1

    def _accumulate(self, accumulated_image, added_image):
        """
        Add an image to the filter.

        :param numpy.ndarray accumulated_image: image use to accumulate
            information
        :param numpy.ndarray added_image: image to add
        """
        raise NotImplementedError()

    def get_result(self):
        """
        Get the result of the filter.

        :return: result filter
        :rtype: numpy.ndarray
        """
        result = self._accumulated_image
        # release the allocated memory
        self._accumulated_image = None
        return result


class MaxAveraging(ImageAccumulatorFilter):
    name = "max"

    def _accumulate(self, accumulated_image, added_image):
        if accumulated_image is None:
            return added_image
        return numpy.maximum(accumulated_image, added_image)


class MinAveraging(ImageAccumulatorFilter):
    name = "min"

    def _accumulate(self, accumulated_image, added_image):
        if accumulated_image is None:
            return added_image
        return numpy.minimum(accumulated_image, added_image)


class SumAveraging(ImageAccumulatorFilter):
    name = "sum"

    def _accumulate(self, accumulated_image, added_image):
        if accumulated_image is None:
            return added_image
        return accumulated_image + added_image


class MeanAveraging(SumAveraging):
    name = "mean"

    def get_result(self):
        result = super(MeanAveraging, self).get_result()
        return result / numpy.float32(self._count)


class ImageStackFilter(ImageReductionFilter):
    """
    Filter creating a stack from all images and computing everything at the
    end.
    """
    def init(self, max_images=None):
        self._stack = None
        self._max_stack_size = max_images
        self._count = 0

    def add_image(self, image):
        """
        Add an image to the filter.

        :param numpy.ndarray image: image to add
        """
        if self._stack is None:
            shape = self._max_stack_size, image.shape[0], image.shape[1]
            self._stack = numpy.zeros(shape, dtype=numpy.float32)
        self._stack[self._count] = image
        self._count += 1

    def _compute_stack_reduction(self, stack):
        """Called after initialization of the stack and return the reduction
        result."""
        raise NotImplementedError()

    def get_result(self):
        if self._stack is None:
            raise Exception("No data to reduce")

        shape = self._count, self._stack.shape[1], self._stack.shape[2]
        self._stack.resize(shape)
        result = self._compute_stack_reduction(self._stack)
        # release the allocated memory
        self._stack = None
        return result


class AverageDarkFilter(ImageStackFilter):
    """
    Filter based on the algorithm of average_dark

    TODO: Must be split according to each filter_name, and removed
    """
    def __init__(self, filter_name, cut_off, quantiles):
        super(AverageDarkFilter, self).__init__()
        self._filter_name = filter_name
        self._cut_off = cut_off
        self._quantiles = quantiles

    @property
    def name(self):
        return self._filter_name

    def get_parameters(self):
        """Return a dictionary containing filter parameters"""
        return {"cutoff": self._cut_off, "quantiles": self._quantiles}

    def _compute_stack_reduction(self, stack):
        """
        Compute the stack reduction.

        :param numpy.ndarray stack: stack to reduce
        :return: result filter
        :rtype: numpy.ndarray
        """
        return average_dark(stack,
                            self._filter_name,
                            self._cut_off,
                            self._quantiles)


_FILTERS = [
    MaxAveraging,
    MinAveraging,
    MeanAveraging,
    SumAveraging,
]

_FILTER_NAME_MAPPING = {}
for _f in _FILTERS:
    _FILTER_NAME_MAPPING[_f.name] = _f

_AVERAGE_DARK_FILTERS = set(["min", "max", "sum", "mean", "std", "quantiles", "median"])


def is_algorithm_name_exists(filter_name):
    """Return true if the name is a name of a filter algorithm"""
    if filter_name in _FILTER_NAME_MAPPING:
        return True
    elif filter_name in _AVERAGE_DARK_FILTERS:
        return True
    return False


class AlgorithmCreationError(RuntimeError):
    """Exception returned if creation of an ImageReductionFilter is not
    possible"""
    pass


def create_algorithm(filter_name, cut_off=None, quantiles=None):
    """Factory to create algorithm according to parameters

    :param cutoff: keep all data where (I-center)/std < cutoff
    :type cutoff:  float or None
    :param quantiles: 2-tuple of floats average out data between the two
        quantiles
    :type quantiles:  tuple(float, float) or None
    :return: An algorithm
    :rtype: ImageReductionFilter
    :raise AlgorithmCreationError: If it is not possible to create the
        algorithm
    """
    if filter_name in _FILTER_NAME_MAPPING and cut_off is None:
        # use less memory
        filter_class = _FILTER_NAME_MAPPING[filter_name]
        algorithm = filter_class()
    elif filter_name in _AVERAGE_DARK_FILTERS:
        # must create a big array with all the data
        if filter_name == "quantiles" and quantiles is None:
            raise AlgorithmCreationError("Quantiles algorithm expect quantiles parameters")
        algorithm = AverageDarkFilter(filter_name, cut_off, quantiles)
    else:
        raise AlgorithmCreationError("No algorithm available for the expected parameters")

    return algorithm


def bounding_box(img):
    """
    Tries to guess the bounding box around a valid massif

    :param img: 2D array like
    :return: 4-typle (d0_min, d1_min, d0_max, d1_max)
    """
    img = img.astype(numpy.int)
    img0 = (img.sum(axis=1) > 0).astype(numpy.int)
    img1 = (img.sum(axis=0) > 0).astype(numpy.int)
    dimg0 = img0[1:] - img0[:-1]
    min0 = dimg0.argmax()
    max0 = dimg0.argmin() + 1
    dimg1 = img1[1:] - img1[:-1]
    min1 = dimg1.argmax()
    max1 = dimg1.argmin() + 1
    if max0 == 1:
        max0 = img0.size
    if max1 == 1:
        max1 = img1.size
    return (min0, min1, max0, max1)


def remove_saturated_pixel(ds, threshold=0.1, minimum=None, maximum=None):
    """
    Remove saturated fixes from an array inplace.

    :param ds: a dataset as ndarray
    :param float threshold: what is the upper limit?
        all pixel > max*(1-threshold) are discareded.
    :param float minimum: minumum valid value (or True for auto-guess)
    :param float maximum: maximum valid value
    :return: the input dataset
    """
    shape = ds.shape
    if ds.dtype == numpy.uint16:
        maxt = (1.0 - threshold) * 65535.0
    elif ds.dtype == numpy.int16:
        maxt = (1.0 - threshold) * 32767.0
    elif ds.dtype == numpy.uint8:
        maxt = (1.0 - threshold) * 255.0
    elif ds.dtype == numpy.int8:
        maxt = (1.0 - threshold) * 127.0
    else:
        if maximum is None:
            maxt = (1.0 - threshold) * ds.max()
        else:
            maxt = maximum
    if maximum is not None:
        maxt = min(maxt, maximum)
    invalid = (ds > maxt)
    if minimum:
        if minimum is True:
            # automatic guess of the best minimum TODO: use the HWHM to guess the minumum...
            data_min = ds.min()
            x, y = numpy.histogram(numpy.log(ds - data_min + 1.0), bins=100)
            f = interp1d((y[1:] + y[:-1]) / 2.0, -x, bounds_error=False, fill_value=-x.min())
            max_low = fmin(f, y[1], disp=0)
            max_hi = fmin(f, y[-1], disp=0)
            if max_hi > max_low:
                f = interp1d((y[1:] + y[:-1]) / 2.0, x, bounds_error=False)
                min_center = fminbound(f, max_low, max_hi)
            else:
                min_center = max_hi
            minimum = float(numpy.exp(y[((min_center / y) > 1).sum() - 1])) - 1.0 + data_min
            logger.debug("removeSaturatedPixel: best minimum guessed is %s", minimum)
        ds[ds < minimum] = minimum
        ds -= minimum  # - 1.0

    if invalid.sum(dtype=int) == 0:
        logger.debug("No saturated area where found")
        return ds
    gi = ndimage.morphology.binary_dilation(invalid)
    lgi, nc = ndimage.label(gi)
    if nc > 100:
        logger.warning("More than 100 saturated zones were found on this image !!!!")
    for zone in range(nc + 1):
        dzone = (lgi == zone)
        if dzone.sum(dtype=int) > ds.size // 2:
            continue
        min0, min1, max0, max1 = bounding_box(dzone)
        ksize = min(max0 - min0, max1 - min1)
        subset = ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)]
        while subset.max() > maxt:
            subset = ndimage.median_filter(subset, ksize)
        ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)] = subset
    return ds


def average_dark(lstimg, center_method="mean", cutoff=None, quantiles=(0.5, 0.5)):
    """
    Averages a serie of dark (or flat) images.
    Centers the result on the mean or the median ...
    but averages all frames within  cutoff*std

    :param lstimg: list of 2D images or a 3D stack
    :param str center_method: is the center calculated by a "mean", "median",
        "quantile", "std"
    :param cutoff: keep all data where (I-center)/std < cutoff
    :type cutoff:  float or None
    :param quantiles: 2-tuple of floats average out data between the two
        quantiles
    :type quantiles:  tuple(float, float) or None
    :return: 2D image averaged
    """
    if "ndim" in dir(lstimg) and lstimg.ndim == 3:
        stack = lstimg.astype(numpy.float32)
        shape = stack.shape[1:]
        length = stack.shape[0]
    else:
        shape = lstimg[0].shape
        length = len(lstimg)
        if length == 1:
            return lstimg[0].astype(numpy.float32)
        stack = numpy.zeros((length, shape[0], shape[1]), dtype=numpy.float32)
        for i, img in enumerate(lstimg):
            stack[i] = img
    if center_method in dir(stack):
        center = stack.__getattribute__(center_method)(axis=0)
    elif center_method == "median":
        logger.info("Filtering data (median)")
        center = numpy.median(stack, axis=0)
    elif center_method.startswith("quantil"):
        logger.info("Filtering data (quantiles: %s)", quantiles)
        sorted_ = numpy.sort(stack, axis=0)
        lower = max(0, int(numpy.floor(min(quantiles) * length)))
        upper = min(length, int(numpy.ceil(max(quantiles) * length)))
        if (upper == lower):
            if upper < length:
                upper += 1
            elif lower > 0:
                lower -= 1
            else:
                logger.warning("Empty selection for quantil %s, would keep points from %s to %s", quantiles, lower, upper)
        center = sorted_[lower:upper].mean(axis=0)
    else:
        raise RuntimeError("Cannot understand method: %s in average_dark" % center_method)
    if cutoff is None or cutoff <= 0:
        output = center
    else:
        std = stack.std(axis=0)
        strides = 0, std.strides[0], std.strides[1]
        std.shape = 1, shape[0], shape[1]
        std.strides = strides
        center.shape = 1, shape[0], shape[1]
        center.strides = strides
        mask = ((abs(stack - center) / std) > cutoff)
        stack[numpy.where(mask)] = 0.0
        summed = stack.sum(axis=0)
        output = summed / numpy.float32(numpy.maximum(1, (length - mask.sum(axis=0))))
    return output


def _normalize_image_stack(image_stack):
    """
    Convert input data to a list of 2D numpy arrays or a stack
    of numpy array (3D array).

    :param image_stack: slice of images
    :type image_stack: list or numpy.ndarray
    :return: A stack of image (list of 2D array or a single 3D array)
    :rtype: list or numpy.ndarray
    """
    if image_stack is None:
        return None

    if isinstance(image_stack, numpy.ndarray) and image_stack.ndim == 3:
        # numpy image stack (single 3D image)
        return image_stack

    if isinstance(image_stack, list):
        # list of numpy images (multi 2D images)
        result = []
        for image in image_stack:
            if isinstance(image, six.string_types):
                data = fabio.open(image).data
            elif isinstance(image, numpy.ndarray) and image.ndim == 2:
                data = image
            else:
                raise Exception("Unsupported image type '%s' in image_stack" % type(image))
            result.append(data)
        return result

    raise Exception("Unsupported type '%s' for image_stack" % type(image_stack))


class AverageWriter():
    """Interface for using writer in `Average` process."""

    def write_header(self, merged_files, nb_frames, monitor_name):
        """Write the header of the average

        :param list merged_files: List of files used to generate this output
        :param int nb_frames: Number of frames used
        :param str monitor_name: Name of the monitor used. Can be None.
        """
        raise NotImplementedError()

    def write_reduction(self, algorithm, data):
        """Write one reduction

        :param ImageReductionFilter algorithm: Algorithm used
        :param object data: Data of this reduction
        """
        raise NotImplementedError()

    def close(self):
        """Close the writer. Must not be used anymore."""
        raise NotImplementedError()


class MultiFilesAverageWriter(AverageWriter):
    """Write reductions into multi files. File headers are duplicated."""

    def __init__(self, file_name_pattern, file_format, dry_run=False):
        """
        :param str file_name_pattern: File name pattern for the output files.
            If it contains "{method_name}", it is updated for each
            reduction writing with the name of the reduction.
        :param str file_format: File format used. It is the default
            extension file.
        :param bool dry_run: If dry_run, the file is created on memory but not
            saved on the file system at the end
        """
        self._file_name_pattern = file_name_pattern
        self._global_header = {}
        self._fabio_images = weakref.WeakKeyDictionary()
        self._dry_run = dry_run

        # in case "edf.gz"
        if "." in file_format:
            file_format = file_format.split(".")[0]

        self._fabio_class = fabio.factory(file_format + "image")

    def write_header(self, merged_files, nb_frames, monitor_name):
        self._global_header["nfiles"] = len(merged_files)
        self._global_header["nframes"] = nb_frames
        if monitor_name is not None:
            self._global_header["monitor_name"] = monitor_name

        pattern = "merged_file_%%0%ii" % len(str(len(merged_files)))
        for i, f in enumerate(merged_files):
            name = pattern % i
            self._global_header[name] = f.filename

    def _get_file_name(self, reduction_name):
        keys = {"method_name": reduction_name}
        return stringutil.safe_format(self._file_name_pattern, keys)

    def write_reduction(self, algorithm, data):
        file_name = self._get_file_name(algorithm.name)
        # overwrite the method
        header = fabio.fabioimage.OrderedDict()
        header["method"] = algorithm.name
        for name, value in self._global_header.items():
            header[name] = str(value)
        filter_parameters = algorithm.get_parameters()
        for name, value in filter_parameters.items():
            header[name] = str(value)
        image = self._fabio_class.__class__(data=data, header=header)
        if not self._dry_run:
            image.write(file_name)
            logger.info("Wrote %s", file_name)
        self._fabio_images[algorithm] = image

    def get_fabio_image(self, algorithm):
        """Get the constructed fabio image

        :rtype: fabio.fabioimage.FabioImage
        """
        return self._fabio_images[algorithm]

    def close(self):
        """Close the writer. Must not be used anymore."""
        self._header = None


def common_prefix(string_list):
    """Return the common prefix of a list of strings

    TODO: move it into utils package

    :param list(str) string_list: List of strings
    :rtype: str
    """
    prefix = ""
    for ch in zip(string_list):
        c = ch[0]
        good = True
        for i in ch:
            if i != c:
                good = False
                break
        if good:
            prefix += c
        else:
            break
    return prefix


class AverageObserver(object):

    def image_loaded(self, fabio_image, image_index, images_count):
        """Called when an input image is loaded"""
        pass

    def process_started(self):
        """Called when the full processing is started"""
        pass

    def algorithm_started(self, algorithm):
        """Called when an algorithm is started"""
        pass

    def frame_processed(self, algorithm, frame_index, frames_count):
        """Called after providing a frame to an algorithm"""
        pass

    def result_processing(self, algorithm):
        """Called before the result of an algorithm is computed"""
        pass

    def algorithm_finished(self, algorithm):
        """Called when an algorithm is finished"""
        pass

    def process_finished(self):
        """Called when the full process is finished"""
        pass


class Average(object):
    """Process images to generate an average using different algorithms."""

    def __init__(self):
        """Constructor"""
        self._dark = None
        self._raw_flat = None
        self._flat = None
        self._monitor_key = None
        self._threshold = None
        self._minimum = None
        self._maximum = None
        self._fabio_images = []
        self._writer = None
        self._algorithms = []
        self._nb_frames = 0
        self._correct_flat_from_dark = False
        self._results = weakref.WeakKeyDictionary()
        self._observer = None

    def set_observer(self, observer):
        """Set an observer to the average process.

        :param AverageObserver observer: An observer
        """
        self._observer = observer

    def set_dark(self, dark_list):
        """Defines images used as dark.

        :param list dark_list: List of dark used
        """
        if dark_list is None:
            self._dark = None
            return
        darks = _normalize_image_stack(dark_list)
        self._dark = average_dark(darks, center_method="mean", cutoff=4)

    def set_flat(self, flat_list):
        """Defines images used as flat.

        :param list flat_list: List of dark used
        """
        if flat_list is None:
            self._raw_flat = None
            return
        flats = _normalize_image_stack(flat_list)
        self._raw_flat = average_dark(flats, center_method="mean", cutoff=4)

    def set_correct_flat_from_dark(self, correct_flat_from_dark):
        """Defines if the dark must be applied on the flat.

        :param bool correct_flat_from_dark: If true, the dark is applied.
        """
        self._correct_flat_from_dark = correct_flat_from_dark

    def get_counter_frames(self):
        """Returns the number of frames used for the process.

        :rtype: int
        """
        return self._nb_frames

    def get_fabio_images(self):
        """Returns source images as fabio images.

        :rtype: list(fabio.fabioimage.FabioImage)"""
        return self._fabio_images

    def set_images(self, image_list):
        """Defines the set set of source images to used to process an average.

        :param list image_list: List of filename, numpy arrays, fabio images
            used as source for the computation.
        """
        self._fabio_images = []
        self._nb_frames = 0
        if len(image_list) > 100:
            # if too many files are opened, it may crash. The har limit is 1024
            copy_data = True
        else:
            copy_data = False
        for image_index, image in enumerate(image_list):
            if isinstance(image, six.string_types):
                logger.info("Reading %s", image)
                fabio_image = fabio.open(image)
                if copy_data and fabio_image.nframes == 1:
                    # copy the data so that we can close the file right now.
                    fimg = fabio_image.convert(fabio_image.__class__)
                    fimg.filename = image
                    fabio_image.close()
                    fabio_image = fimg
            elif isinstance(image, fabio.fabioimage.fabioimage):
                fabio_image = image
            else:
                if fabio.hexversion < 262148:
                    logger.error("Old version of fabio detected, upgrade to 0.4 or newer")

                # Assume this is a numpy array like
                if not isinstance(image, numpy.ndarray):
                    raise RuntimeError("Not good type for input, got %s, expected numpy array" % type(image))
                fabio_image = fabio.numpyimage.NumpyImage(data=image)

            if self._observer:
                self._observer.image_loaded(fabio_image, image_index, len(image_list))
            self._fabio_images.append(fabio_image)
            self._nb_frames += fabio_image.nframes

    def set_monitor_name(self, monitor_name):
        """Defines the monitor name used to correct images before processing
        the average. This monitor must be part of the file header, else the
        image is skipped.

        :param str monitor_name: Name of the monitor available on the header
            file
        """

        self._monitor_key = monitor_name

    def set_pixel_filter(self, threshold, minimum, maximum):
        """Defines the filter applied on each pixels of the images before
        processing the average.

        :param threshold: what is the upper limit?
            all pixel > max*(1-threshold) are discareded.
        :param minimum: minimum valid value or True
        :param maximum: maximum valid value
        """
        self._threshold = threshold
        self._minimum = minimum
        self._maximum = maximum

    def set_writer(self, writer):
        """Defines the object write which will be used to store the result.

        :param AverageWriter writer: The writer to use."""
        self._writer = writer

    def add_algorithm(self, algorithm):
        """Defines another algorithm which will be computed on the source.

        :param ImageReductionFilter algorithm: An averaging algorithm.
        """
        self._algorithms.append(algorithm)

    def _get_corrected_image(self, fabio_image, image):
        """Returns an image corrected by pixel filter, saturation, flat, dark,
        and monitor correction. The internal computation is done in float
        64bits. The result is provided as float 32 bits.

        :param fabio.fabioimage.FabioImage fabio_image: Object containing the
            header of the data to process
        :param numpy.ndarray image: Data to process
        :rtype: numpy.ndarray
        """
        corrected_image = numpy.ascontiguousarray(image, numpy.float64)
        if self._threshold or self._minimum or self._maximum:
            corrected_image = remove_saturated_pixel(corrected_image, self._threshold, self._minimum, self._maximum)
        if self._dark is not None:
            corrected_image -= self._dark
        if self._flat is not None:
            corrected_image /= self._flat
        if self._monitor_key is not None:
            try:
                monitor = header_utils.get_monitor_value(fabio_image, self._monitor_key)
                corrected_image /= monitor
            except header_utils.MonitorNotFound as e:
                logger.warning("Monitor not found in filename '%s', data skipped. Cause: %s", fabio_image.filename, str(e))
                return None
        return numpy.ascontiguousarray(corrected_image, numpy.float32)

    def _get_image_reduction(self, algorithm):
        """Returns the result of an averaging algorithm using all over
        parameters defined in this object.

        :param ImageReductionFilter algorithm: Averaging algorithm
        :rtype: numpy.ndarray
        """
        algorithm.init(max_images=self._nb_frames)
        frame_index = 0
        for fabio_image in self._fabio_images:
            for frame in range(fabio_image.nframes):
                if fabio_image.nframes == 1:
                    data = fabio_image.data
                else:
                    data = fabio_image.getframe(frame).data
                logger.debug("Intensity range for %s#%i is %s --> %s", fabio_image.filename, frame, data.min(), data.max())

                corrected_image = self._get_corrected_image(fabio_image, data)
                if corrected_image is not None:
                    algorithm.add_image(corrected_image)
                if self._observer:
                    self._observer.frame_processed(algorithm, frame_index, self._nb_frames)
                frame_index += 1
        if self._observer:
            self._observer.result_processing(algorithm)
        return algorithm.get_result()

    def _update_flat(self):
        """
        Update the flat according to the last process parameters

        :rtype: numpy.ndarray
        """
        if self._raw_flat is not None:
            flat = numpy.array(self._raw_flat)
            if self._correct_flat_from_dark:
                if self._dark is not None:
                    flat -= self._dark
                else:
                    logger.debug("No dark. Flat correction using dark skipped")
            flat[numpy.where(flat <= 0)] = 1.0
        else:
            flat = None
        self._flat = flat

    def process(self):
        """Process source images to all defined averaging algorithms defined
        using defined parameters. To access to the results you have to define
        a writer (`AverageWriter`). To follow the process forward you have to
        define an observer (`AverageObserver`).
        """
        self._update_flat()
        writer = self._writer

        if self._observer:
            self._observer.process_started()

        if writer is not None:
            writer.write_header(self._fabio_images, self._nb_frames, self._monitor_key)

        for algorithm in self._algorithms:
            if self._observer:
                self._observer.algorithm_started(algorithm)
            image_reduction = self._get_image_reduction(algorithm)
            logger.debug("Intensity range in merged dataset : %s --> %s", image_reduction.min(), image_reduction.max())
            if writer is not None:
                writer.write_reduction(algorithm, image_reduction)
            self._results[algorithm] = image_reduction
            if self._observer:
                self._observer.algorithm_finished(algorithm)

        if self._observer:
            self._observer.process_finished()

        if writer is not None:
            writer.close()

    def get_image_reduction(self, algorithm):
        """Returns the result of an algorithm. The `process` must be already
        done.

        :param ImageReductionFilter algorithm: An averaging algorithm
        :rtype: numpy.ndarray
        """
        return self._results[algorithm]


def average_images(listImages, output=None, threshold=0.1, minimum=None,
                   maximum=None, darks=None, flats=None, filter_="mean",
                   correct_flat_from_dark=False, cutoff=None, quantiles=None,
                   fformat="edf", monitor_key=None):
    """
    Takes a list of filenames and create an average frame discarding all
        saturated pixels.

    :param listImages: list of string representing the filenames
    :param output: name of the optional output file
    :param threshold: what is the upper limit? all pixel > max*(1-threshold)
        are discareded.
    :param minimum: minimum valid value or True
    :param maximum: maximum valid value
    :param darks: list of dark current images for subtraction
    :param flats: list of flat field images for division
    :param filter_: can be "min", "max", "median", "mean", "sum", "quantiles"
        (default='mean')
    :param correct_flat_from_dark: shall the flat be re-corrected ?
    :param cutoff: keep all data where (I-center)/std < cutoff
    :param quantiles: 2-tuple containing the lower and upper quantile (0<q<1)
        to average out.
    :param fformat: file format of the output image, default: edf
    :param monitor_key str: Key containing the monitor. Can be none.
    :return: filename with the data or the data ndarray in case format=None
    """

    # input sanitization
    if not is_algorithm_name_exists(filter_):
        logger.warning("Filter %s not understood. switch to mean filter", filter_)
        filter_ = "mean"

    if quantiles is not None and filter_ != "quantiles":
        logger.warning("Set method to quantiles as quantiles parameters is defined.")
        filter_ = "quantiles"

    average = Average()
    average.set_images(listImages)
    average.set_dark(darks)
    average.set_flat(flats)
    average.set_correct_flat_from_dark(correct_flat_from_dark)
    average.set_monitor_name(monitor_key)
    average.set_pixel_filter(threshold, minimum, maximum)

    algorithm = create_algorithm(filter_, cutoff, quantiles)
    average.add_algorithm(algorithm)

    # define writer
    if fformat is not None:
        if fformat.startswith("."):
            fformat = fformat.lstrip(".")
        if output is None:
            prefix = common_prefix([i.filename for i in average.get_fabio_images()])
            output = "filt%02i-%s.%s" % (average.get_counter_frames(), prefix, fformat)
            output = "{method_name}" + output

    if output is not None:
        writer = MultiFilesAverageWriter(output, fformat)
        average.set_writer(writer)
    else:
        writer = None

    average.process()

    if writer is not None:
        fabio_image = writer.get_fabio_image(algorithm)
        return fabio_image.filename
    else:
        return average.get_image_reduction(algorithm)
