#!/usr/bin/python
from __future__ import absolute_import, print_function, with_statement, division
# this is a very simple tool that checks the calibratation
import pyFAI, fabio, numpy, sys, os, optparse, time
import pylab


def shift(input, shift):
    """
    Shift an array like  scipy.ndimage.interpolation.shift(input, shift, mode="wrap", order=0) but faster
    @param input: 2d numpy array
    @param shift: 2-tuple of integers
    @return: shifted image
    """
    re = numpy.zeros_like(input)
    s0, s1 = input.shape
    d0 = shift[0] % s0
    d1 = shift[1] % s1
    r0 = (-d0) % s0
    r1 = (-d1) % s1
    re[d0:, d1:] = input[:r0, :r1]
    re[:d0, d1:] = input[r0:, :r1]
    re[d0:, :d1] = input[:r0, r1:]
    re[:d0, :d1] = input[r0:, r1:]
    return re


def shiftFFT(inp, shift, method="fftw"):
    """
    Do shift using FFTs
    Shift an array like  scipy.ndimage.interpolation.shift(input, shift, mode="wrap", order="infinity") but faster
    @param input: 2d numpy array
    @param shift: 2-tuple of float
    @return: shifted image

    """
    d0, d1 = inp.shape
    v0, v1 = shift
    f0 = numpy.fft.ifftshift(numpy.arange(-d0 // 2, d0 // 2))
    f1 = numpy.fft.ifftshift(numpy.arange(-d1 // 2, d1 // 2))
    m1, m0 = numpy.meshgrid(f1, f0)
    e0 = numpy.exp(-2j * numpy.pi * v0 * m0 / float(d0))
    e1 = numpy.exp(-2j * numpy.pi * v1 * m1 / float(d1))
    e = e0 * e1
    if method.startswith("fftw") and (fftw3 is not None):

        input = numpy.zeros((d0, d1), dtype=complex)
        output = numpy.zeros((d0, d1), dtype=complex)
        with sem:
                fft = fftw3.Plan(input, output, direction='forward', flags=['estimate'])
                ifft = fftw3.Plan(output, input, direction='backward', flags=['estimate'])
        input[:, :] = inp.astype(complex)
        fft()
        output *= e
        ifft()
        out = input / input.size
    else:
        out = numpy.fft.ifft2(numpy.fft.fft2(inp) * e)
    return abs(out)

def maximum_position(img):
    """
    Same as scipy.ndimage.measurements.maximum_position:
    Find the position of the maximum of the values of the array.

    @param img: 2-D image
    @return: 2-tuple of int with the position of the maximum
    """
    maxarg = numpy.argmax(img)
    s0, s1 = img.shape
    return (maxarg // s1, maxarg % s1)

def center_of_mass(img):
    """
    Calculate the center of mass of of the array.
    Like scipy.ndimage.measurements.center_of_mass
    @param img: 2-D array
    @return: 2-tuple of float with the center of mass
    """
    d0, d1 = img.shape
    a0, a1 = numpy.ogrid[:d0, :d1]
    img = img.astype("float64")
    img /= img.sum()
    return ((a0 * img).sum(), (a1 * img).sum())

def measure_offset(img1, img2, method="numpy", withLog=False, withCorr=False):
    """
    Measure the actual offset between 2 images
    @param img1: ndarray, first image
    @param img2: ndarray, second image, same shape as img1
    @param withLog: shall we return logs as well ? boolean
    @return: tuple of floats with the offsets
    """
    method = str(method)
    ################################################################################
    # Start convolutions
    ################################################################################
    shape = img1.shape
    logs = []
    assert img2.shape == shape
    t0 = time.time()
    i1f = numpy.fft.fft2(img1)
    i2f = numpy.fft.fft2(img2)
    res = numpy.fft.ifft2(i1f * i2f.conjugate()).real
    t1 = time.time()
    ################################################################################
    # END of convolutions
    ################################################################################
    offset1 = maximum_position(res)
    res = shift(res, (shape[0] // 2 , shape[1] // 2))
    mean = res.mean(dtype="float64")
    maxi = res.max()
    std = res.std(dtype="float64")
    SN = (maxi - mean) / std
    new = numpy.maximum(numpy.zeros(shape), res - numpy.ones(shape) * (mean + std * SN * 0.9))
    com2 = center_of_mass(new)
    logs.append("MeasureOffset: fine result of the centered image: %s %s " % com2)
    offset2 = ((com2[0] - shape[0] // 2) % shape[0] , (com2[1] - shape[1] // 2) % shape[1])
    delta0 = (offset2[0] - offset1[0]) % shape[0]
    delta1 = (offset2[1] - offset1[1]) % shape[1]
    if delta0 > shape[0] // 2:
        delta0 -= shape[0]
    if delta1 > shape[1] // 2:
        delta1 -= shape[1]
    if (abs(delta0) > 2) or (abs(delta1) > 2):
        logs.append("MeasureOffset: Raw offset is %s and refined is %s. Please investigate !" % (offset1, offset2))
    listOffset = list(offset2)
    if listOffset[0] > shape[0] // 2:
        listOffset[0] -= shape[0]
    if listOffset[1] > shape[1] // 2:
        listOffset[1] -= shape[1]
    offset = tuple(listOffset)
    t2 = time.time()
    logs.append("MeasureOffset: fine result: %s %s" % offset)
    logs.append("MeasureOffset: execution time: %.3fs with %.3fs for FFTs" % (t2 - t0, t1 - t0))
    if withLog:
        if withCorr:
            return offset, logs, new
        else:
            return offset, logs
    else:
        if withCorr:
            return offset, new
        else:
            return offset


class CheckCalib(object):
    def __init__(self, poni, img):
        self.ponifile = poni
        self.ai = pyFAI.load(poni)
        self.img = fabio.open(img)
        self.r = None
        self.I = None
        self.resynth = None
        self.delta = None

    def __repr__(self, *args, **kwargs):
        return self.ai.__repr__()

    def integrate(self):
        self.r, self.I = self.ai.integrate1d(self.img.data, 2048, unit="q_nm^-1")

    def rebuild(self):
        if self.r is None:
            self.integrate()
        self.resynth = self.ai.calcfrom1d(self.r, self.I, self.img.data.shape, mask=None,
                   dim1_unit="q_nm^-1", correctSolidAngle=True)
        self.delta = self.resynth - self.img.data
        self.offset, log = measure_offset(self.resynth, self.img.data, withLog=1)
        print(os.linesep.join(log))
        print(self.offset)
if __name__ == "__main__":
    cc = CheckCalib(sys.argv[1], sys.argv[2])
    cc.integrate()
    cc.rebuild()
    pylab.ion()

    pylab.imshow(cc.delta, aspect="auto", interpolation=None, origin="bottom")
#    pylab.show()
    raw_input("Delta image")
    pylab.imshow(cc.img.data, aspect="auto", interpolation=None, origin="bottom")
    raw_input("raw image")
    pylab.imshow(cc.resynth, aspect="auto", interpolation=None, origin="bottom")
    raw_input("rebuild image")
    pylab.clf()
    pylab.plot(cc.r, cc.I)
    raw_input("powder pattern")
