import numpy
import _convolution, scipy.misc, scipy.ndimage
def gaussian(sigma, width=None):
    """
    Return a Gaussian window of length "width" with standard-deviation "sigma".

    @param sigma: standard deviation sigma
    @param width: length of the windows (int) By default 8*sigma+1,

    Width should be odd.

    The FWHM is 2*sqrt(2 * pi)*sigma

    """
    if width is None:
        width = int(8 * sigma + 1)
        if width % 2 == 0:
            width += 1
    sigma = float(sigma)
    x = numpy.arange(width) - (width - 1) / 2.0
    g = numpy.exp(-(x / sigma) ** 2 / 2.0)
    return g / g.sum()

if __name__ == "__main__":
    g1 = gaussian(1).astype("float32")
    l = scipy.misc.lena().astype("float32")
    ref = scipy.ndimage.filters.convolve1d(l, g1, axis= -1)
    obt = _convolution.horizontal_convolution(l, g1)
    print abs(ref - obt).max()
