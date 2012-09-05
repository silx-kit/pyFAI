import numpy
import fabio #, h5py
from scipy import ndimage
from scipy.interpolate import interp1d
from math import  ceil
import logging, sys
import relabel as relabelCython
from scipy.optimize.optimize import fmin, fminbound
logger = logging.getLogger("pyFAI.utils")
import time
timelog = logging.getLogger("pyFAI.timeit")
from scipy.signal           import gaussian
if sys.platform != "win32":
    WindowsError = RuntimeError

def timeit(func):
    def wrapper(*arg, **kw):
        '''This is the docstring of timeit:
        a decorator that logs the execution time'''
        t1 = time.time()
        res = func(*arg, **kw)
        timelog.warning("%s took %.3fs" % (func.func_name, time.time() - t1))
        return res
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

try:
    import fftw3
except (ImportError, WindowsError) as e:
    logging.warn("Exception %s: FFTw3 not available. Falling back on Scipy" % e)
    from scipy.ndimage.filters import gaussian_filter
else:
    def gaussian_filter(input, sigma, mode="reflect", cval=0.0):
        """
        2-dimensional Gaussian filter implemented with FFTw

        @param input:    input array to filter
        @type input: array-like
        @param sigma: standard deviation for Gaussian kernel.
            The standard deviations of the Gaussian filter are given for each axis as a sequence,
            or as a single number, in which case it is equal for all axes.
        @type sigma: scalar or sequence of scalars
        @param mode: {'reflect','constant','nearest','mirror', 'wrap'}, optional
            The ``mode`` parameter determines how the array borders are
            handled, where ``cval`` is the value when mode is equal to
            'constant'. Default is 'reflect'
        @param cval: scalar, optional
            Value to fill past edges of input if ``mode`` is 'constant'. Default is 0.0
"""

        try:
#            orig_shape = input.shape
            if mode != "wrap":
                input = expand(input, sigma, mode, cval)
            s0, s1 = input.shape
            if isinstance(sigma, (list, tuple)):
                k0 = int(ceil(float(sigma[0])))
                k1 = int(ceil(float(sigma[1])))
            else:
                k0 = k1 = int(ceil(float(sigma)))

            sum_init = input.astype(numpy.float32).sum()
            fftOut = numpy.zeros((s0, s1), dtype=complex)
            fftIn = numpy.zeros((s0, s1), dtype=complex)
            fft = fftw3.Plan(fftIn, fftOut, direction='forward')
            ifft = fftw3.Plan(fftOut, fftIn, direction='backward')

            g0 = gaussian(s0, k0)
            g1 = gaussian(s1, k1)
            g0 = numpy.concatenate((g0[s0 // 2:], g0[:s0 // 2]))
            g1 = numpy.concatenate((g1[s1 // 2:], g1[:s1 // 2]))
            g2 = numpy.outer(g0, g1)
            g2fft = numpy.zeros((s0, s1), dtype=complex)
            fftIn[:, :] = g2.astype(complex)
            fft()
            g2fft[:, :] = fftOut.conjugate()

            fftIn[:, :] = input.astype(complex)
            fft()

            fftOut *= g2fft
            ifft()
            out = fftIn.real.astype(numpy.float32)
            sum_out = out.sum()
            res = out * sum_init / sum_out
            if mode == "wrap":
                return res
            else:
                return res[k0:-k0, k1:-k1]
        except MemoryError:
            logging.error("MemoryError in FFTw3 part. Falling back on Scipy")
            import scipy.ndimage.filters
            return scipy.ndimage.filters.gaussian_filter(input, sigma, mode=(mode or "reflect"))




def expand(input, sigma, mode="constant", cval=0.0):

    """Expand array a with its reflection on boundaries

    @param a: 2D array
    @param sigma: float or 2-tuple of floats
    @param mode:"constant","nearest" or "reflect"
    @param cval: filling value used for constant, 0.0 by default
    """
    s0, s1 = input.shape
    dtype = input.dtype
    if isinstance(sigma, (list, tuple)):
        k0 = int(ceil(float(sigma[0])))
        k1 = int(ceil(float(sigma[1])))
    else:
        k0 = k1 = int(ceil(float(sigma)))
    if k0 > s0 or k1 > s1:
        raise RuntimeError("Makes little sense to apply a kernel (%i,%i)larger than the image (%i,%i)" % (k0, k1, s0, s1))
    output = numpy.zeros((s0 + 2 * k0, s1 + 2 * k1), dtype=dtype) + float(cval)
    output[k0:k0 + s0, k1:k1 + s1] = input
    if mode in  ["reflect", "mirror"]:
    #4 corners
        output[s0 + k0:, s1 + k1:] = input[-1:-k0 - 1:-1, -1:-k1 - 1:-1]
        output[:k0, :k1] = input[k0 - 1::-1, k1 - 1::-1]
        output[:k0, s1 + k1:] = input[k0 - 1::-1, s1 - 1: s1 - k1 - 1:-1]
        output[s0 + k0:, :k1] = input[s0 - 1: s0 - k0 - 1:-1, k1 - 1::-1]
    #4 sides
        output[k0:k0 + s0, :k1] = input[:s0, k1 - 1::-1]
        output[:k0, k1:k1 + s1] = input[k0 - 1::-1, :s1]
        output[-k0:, k1:s1 + k1] = input[:s0 - k0 - 1:-1, :]
        output[k0:s0 + k0, -k1:] = input[:, :s1 - k1 - 1:-1]
    elif mode == "nearest":
    #4 corners
        output[s0 + k0:, s1 + k1:] = input[-1, -1]
        output[:k0, :k1] = input[0, 0]
        output[:k0, s1 + k1:] = input[0, -1]
        output[s0 + k0:, :k1] = input[-1, 0]
    #4 sides
        output[k0:k0 + s0, :k1] = numpy.outer(input[:, 0], numpy.ones(k1))
        output[:k0, k1:k1 + s1] = numpy.outer(numpy.ones(k0), input[0, :])
        output[-k0:, k1:s1 + k1] = numpy.outer(numpy.ones(k0), input[-1, :])
        output[k0:s0 + k0, -k1:] = numpy.outer(input[:, -1], numpy.ones(k1))
    return output



def relabel(label, data, blured, max_size=None):
    """
    Relabel limits the number of region in the label array.
    They are ranked relatively to their max(I0)-max(blur(I0)

    @param label: a label array coming out of scipy.ndimage.measurement.label
    @param data: an array containing the raw data
    @param blured: an array containing the blured data
    @param max_size: the max number of label wanted
    @return array like label
    """
    max_label = label.max()
    a, b, c, d = relabelCython.countThem(label, data, blured)
    count = d
    sortCount = count.argsort()
    invSortCount = sortCount[-1::-1]
    invCutInvSortCount = numpy.zeros(max_label + 1, dtype=int)
    for i, j in enumerate(list(invSortCount[:max_size])):
        invCutInvSortCount[j] = i
    f = lambda i:invCutInvSortCount[i]
    return f(label)


def averageImages(listImages, output=None, threshold=0.1, minimum=None, maximum=None, darks=None, flats=None):
    """
    Takes a list of filenames and create an average frame discarding all saturated pixels.

    @param listImages: list of string representing the filenames
    @param output: name of the optional output file
    @param threshold: what is the upper limit? all pixel > max*(1-threshold) are discareded.
    @param minimum: minimum valid value or True
    @param maximum: maximum valid value
    @param darks: list of dark current images for subtraction
    @param flats: list of flat field images for division
    """
    ld = len(listImages)
    sumImg = None
    dark = None
    flat = None

    for fn in listImages:
        logger.info("Reading %s" % fn)
        ds = fabio.open(fn).data
        logger.debug("Intensity range for %s is %s --> %s", fn, ds.min(), ds.max())
        shape = ds.shape
        if sumImg is None:
            sumImg = numpy.zeros((shape[0], shape[1]), dtype=numpy.float64)
        if dark is None:
            dark = numpy.zeros((shape[0], shape[1]), dtype=numpy.float32)
            if darks:
                for f in darks:
                    dark += fabio.open(f).data
                dark /= max(1, len(darks))
        if flat is None:
            print flats
            if flats:
                flat = numpy.zeros((shape[0], shape[1]), dtype=numpy.float32)
                for f in flats:
                    flat += fabio.open(f).data
                flat /= max(1, len(flats))
                flat -= dark
                flat[flats < 1] = 1.0
            else:
                flat = numpy.ones((shape[0], shape[1]), dtype=numpy.float32)

        sumImg += (removeSaturatedPixel(ds.astype(numpy.float32), threshold, minimum, maximum) - dark) / flat
    datared = (sumImg / float(ld)).astype(numpy.float32)
    if output is None:
        prefix = ""
        for ch in zip(*listImages):
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
        output = ("merge%02i-" % ld) + prefix + ".edf"
    logger.debug("Intensity range in merged dataset : %s --> %s", datared.min(), datared.max())
    fabio.edfimage.edfimage(data=datared,
                            header={"merged": ", ".join(listImages)}).write(output)
    return output


def boundingBox(img):
    """
    Tries to guess the bounding box around a valid massif

    @param img: 2D array like
    @return: 4-typle (d0_min, d1_min, d0_max, d1_max)
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


def removeSaturatedPixel(ds, threshold=0.1, minimum=None, maximum=None):
    """
    @param ds: a dataset as  ndarray

    @param threshold: what is the upper limit? all pixel > max*(1-threshold) are discareded.
    @param minimum: minumum valid value (or True for auto-guess)
    @param maximum: maximum valid value
    @return: another dataset
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
        if maximum is  None:
            maxt = (1.0 - threshold) * ds.max()
        else:
            maxt = maximum
    if maximum is not None:
        maxt = min(maxt, maximum)
    invalid = (ds > maxt)
    if minimum:
        if  minimum is True: #automatic guess of the best minimum TODO: use the HWHM to guess the minumum...
            data_min = ds.min()
            x, y = numpy.histogram(numpy.log(ds - data_min + 1.0), bins=100)
            f = interp1d((y[1:] + y[:-1]) / 2.0, -x, bounds_error=False, fill_value= -x.min())
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
        ds -= minimum #- 1.0

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
        min0, min1, max0, max1 = boundingBox(dzone)
        ksize = min(max0 - min0, max1 - min1)
        subset = ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)]
        while subset.max() > maxt:
            subset = ndimage.median_filter(subset, ksize)
        ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)] = subset
    fabio.edfimage.edfimage(data=ds).write("removeSaturatedPixel.edf")
    return ds


def binning(inputArray, binsize):
    """
    @param inputArray: input ndarray
    @param binsize: int or 2-tuple representing the size of the binning
    @return: binned input ndarray
    """
    inputSize = inputArray.shape
    outputSize = []
    assert(len(inputSize) == 2)
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    for i, j in zip(inputSize, binsize):
        assert(i % j == 0)
        outputSize.append(i // j)

    if numpy.array(binsize).prod() < 50:
        out = numpy.zeros(tuple(outputSize))
        for i in xrange(binsize[0]):
            for j in xrange(binsize[1]):
                out += inputArray[i::binsize[0], j::binsize[1]]
    else:
        temp = inputArray.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    return out


def unBinning(binnedArray, binsize):
    """
    @param binnedArray: input ndarray
    @param binsize: 2-tuple representing the size of the binning
    @return: unBinned input ndarray
    """
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    outputShape = []
    for i, j in zip(binnedArray.shape, binsize):
        outputShape.append(i * j)
    out = numpy.zeros(tuple(outputShape), dtype=binnedArray.dtype)
    for i in xrange(binsize[0]):
        for j in xrange(binsize[1]):
            out[i::binsize[0], j::binsize[1]] += binnedArray
    return out


