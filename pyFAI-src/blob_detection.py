import numpy
try:
    from _convolution import gaussian_filter

except ImportError:
    from scipy.ndimage.filters import gaussian_filter
from math import sqrt

from utils import binning, timeit

@timeit
def local_max_min(prev_dog, cur_dog, next_dog, sigma, mask=None):
    """
    @param prev_dog, cur_dog, next_dog: 3 subsequent Difference of gaussian
    @param sigma: value of sigma for cur_dog 
    @parm mask: mask out keypoint next to the mask (or inside the mask)
    """

    kpm = numpy.zeros(shape=cur_dog.shape, dtype=numpy.uint8)
    slic = cur_dog[1:-1, 1:-1]
    kpm[1:-1, 1:-1] += (slic > cur_dog[:-2, 1:-1]) * (slic > cur_dog[2:, 1:-1])
    kpm[1:-1, 1:-1] += (slic > cur_dog[1:-1, :-2]) * (slic > cur_dog[1:-1, 2:])
    kpm[1:-1, 1:-1] += (slic > cur_dog[:-2, :-2]) * (slic > cur_dog[2:, 2:])
    kpm[1:-1, 1:-1] += (slic > cur_dog[2:, :-2]) * (slic > cur_dog[:-2, 2:])

    #with next DoG
    kpm[1:-1, 1:-1] += (slic > next_dog[:-2, 1:-1]) * (slic > next_dog[2:, 1:-1])
    kpm[1:-1, 1:-1] += (slic > next_dog[1:-1, :-2]) * (slic > next_dog[1:-1, 2:])
    kpm[1:-1, 1:-1] += (slic > next_dog[:-2, :-2]) * (slic > next_dog[2:, 2:])
    kpm[1:-1, 1:-1] += (slic > next_dog[2:, :-2]) * (slic > next_dog[:-2, 2:])
    kpm[1:-1, 1:-1] += (slic >= next_dog[1:-1, 1:-1])

    #with previous DoG
    kpm[1:-1, 1:-1] += (slic > prev_dog[:-2, 1:-1]) * (slic > prev_dog[2:, 1:-1])
    kpm[1:-1, 1:-1] += (slic > prev_dog[1:-1, :-2]) * (slic > prev_dog[1:-1, 2:])
    kpm[1:-1, 1:-1] += (slic > prev_dog[:-2, :-2]) * (slic > prev_dog[2:, 2:])
    kpm[1:-1, 1:-1] += (slic > prev_dog[2:, :-2]) * (slic > prev_dog[:-2, 2:])
    kpm[1:-1, 1:-1] += (slic >= prev_dog[1:-1, 1:-1])

    if mask is not None:
        not_mask = numpy.logical_not(mask[1:-1, 1:-1])
        valid_point = numpy.logical_and(not_mask, kpm >= 14)
    else:
        valid_point = (kpm >= 14)
    kpy, kpx = numpy.where(valid_point)
    l = kpx.size
    keypoints = numpy.empty((l,4),dtype=numpy.float32)
    keypoints[:, 0] = kpx
    keypoints[:, 1] = kpy
    keypoints[:,2] = sigma
    keypoints[:, 3] = cur_dog[(kpy, kpx)]
    return keypoints

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
        self.cur_sigma = float(cur_sigma)
        self.init_sigma = float(init_sigma)
        self.dest_sigma = float(dest_sigma)
        self.scale_per_octave = int(scale_per_octave)
        self.data = None    # current image
        self.sigmas = None  # contains pairs of absolute sigma and relative ones...
        self.blurs = []     # different blurred images
        self.dogs = []      # different difference of gaussians
        self.border_size = 5# size of the border
        self.keypoints = []

    def _initial_blur(self):
        """
        Blur the original image to achieve the requested level of blur init_sigma
        """
        if self.init_sigma > self.cur_sigma:
            sigma = sqrt(self.init_sigma ** 2 - self.cur_sigma ** 2)
            self.data = gaussian_filter(self.raw, sigma)
        else:
            self.data = self.raw

    def _calc_sigma(self):
        """
        Calculate all sigma to blur an image within an octave
        """
        if not self.data:
            self._initial_blur()
        previous = self.init_sigma
        incr = 0
        self.sigmas = [(previous, incr)]
        for i in range(1, self.scale_per_octave + 3):
            sigma_abs = self.init_sigma * (self.dest_sigma / self.init_sigma) ** (1.0 * i / (self.scale_per_octave))
            increase = previous * sqrt((self.dest_sigma / self.init_sigma) ** (2.0 / self.scale_per_octave) - 1.0)
            self.sigmas.append((sigma_abs, increase))
            previous = sigma_abs

    @timeit
    def _one_octave(self):
        """
        Return the blob coordinates for an octave 
        """
        if not self.sigmas:
            self._calc_sigma()
        previous = self.data
        for sigma_abs, sigma_rel in self.sigmas:
            if  sigma_rel == 0:
                self.blurs.append(previous)
            else:
                new_blur = gaussian_filter(previous, sigma_rel)
                self.blurs.append(new_blur)
                self.dogs.append(previous - new_blur)
                previous = new_blur
        for i in range(1, self.scale_per_octave + 1):
            sigma = self.sigmas[i][0]
            self.keypoints.append(local_max_min(self.dogs[i - 1], self.dogs[i], self.dogs[i + 1], sigma=sigma))
        #shrink data so that
        self.data = binning(self.blurs[self.scale_per_octave], 2)

if __name__ == "__main__":
#    import scipy.misc
    import fabio
    img = fabio.open("../test/testimages/halfccd.edf").data
    bd = BlobDetection(img)
    bd._one_octave()
    print bd.sigmas
