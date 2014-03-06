import numpy
try:
    from _convolution import gaussian_filter

except ImportError:
    from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
import _convolution
from math import sqrt

class BlobDetection(object):
    """
    
    """
    def __init__(self, img, cur_sigma=0.25, init_sigma=0.5, dest_sigma=2.0, scale_per_octave=3):
        """
        Performs a blob detection:
        http://en.wikipedia.org/wiki/Blob_detection
        using a Difference of Gaussian + Pyramid of Gaussians
        
        @param img: input image
        @param cur_sigma: estimated smoothing of the input image. 0.25 correspond to no interaction between pixels.
        @param init_sigma: start searching at this scale (sigma=0.5: 10% interaction with first neighbor)
        @param dest_sigma: sigma at which the resolution is lowered (change of octave)
        @param scale_per_octave: Number of scale to be performed per octave
        """
        self.raw = numpy.ascontiguousarray(img, dtype=numpy.float32)
        self.cur_sigma = cur_sigma
        self.init_sigma = init_sigma
        self.dest_sigma = dest_sigma
        self.scale_per_octave = scale_per_octave
        self.data = None #current image

    def _initial_blur(self):
        """
        Blur the original image to achieve the requested level of blur init_sigma
        """
        sigma = sqrt(self.init_sigma ** 2 - self.cur_sigma ** 2)
        self.data = gaussian_filter(self.raw, sigma)

    def one_octave(self):
        """
        Return the blob coordinates for an octave 
        """
        pass

if __name__ == "__main__":
    import scipy.misc
    g1 = gaussian(1).astype("float32")
    l = scipy.misc.lena().astype("float32")
    ref = scipy.ndimage.filters.convolve1d(l, g1, axis= -1)
    obt = _convolution.horizontal_convolution(l, g1)
    print abs(ref - obt).max()
    ref = scipy.ndimage.filters.convolve1d(l, g1, axis=0)
    print abs(ref - obt).max()
    obt = _convolution.vertical_convolution(l, g1)
    print abs(ref - obt).max()
