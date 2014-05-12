#!/usr/bin/python
# coding: utf8
import sys, scipy,pylab
from math import sqrt
import fabio,numpy
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.blob_detection


def image_test():
    img = numpy.zeros((128*4,128*4))
    a = numpy.linspace(0.5, 8, 16)
    xc = [64,64,64,64,192,192,192,192,320,320,320,320,448,448,448,448]
    yc = [64,192,320,448,64,192,320,448,64,192,320,448,64,192,320,448]
    cpt = 0
    for sigma in a:
        img = make_gaussian(img,sigma,xc[cpt],yc[cpt])
        cpt = cpt + 1
    return img

def make_gaussian(im,sigma,xc,yc):
    e = 0.75
    angle = 0
    sx = sigma * (1+e)
    sy = sigma * (1-e)
    size = int( 8*sigma +1 )
    if size%2 == 0 :
        size += 1
    x = numpy.arange(0, size, 1, float)
    y = x[:,numpy.newaxis]
#     x = x * 2
    x0 = y0 = size // 2
    gausx = numpy.exp(-4*numpy.log(2) * (x-x0)**2 / sx**2)
    gausy = numpy.exp(-4*numpy.log(2) * (y-y0)**2 / sy**2)
    gaus = 0.01 + gausx * gausy
    im[xc-size/2:xc+size/2+1,yc-size/2:yc+size/2+1] = scipy.ndimage.rotate(gaus,angle, reshape = False)
    return im



# data = fabio.open(UtilsTest.options.args[0]).data
# msk = fabio.open(UtilsTest.options.args[1]).data

# data = fabio.open("../../testimages/halfccd.edf").data
# msk = fabio.open("../../testimages/halfccd_8_mask.tiff").data
data = image_test()

bd = pyFAI.blob_detection.BlobDetection(data)

pylab.ion()
f=pylab.figure(1)
ax = f.add_subplot(111)
ax.imshow(numpy.log1p(data), interpolation = 'nearest')

for i in range(4):
    print ('Octave #%i' %i)
    bd._one_octave(True, False , False)
    print("Octave #%i Total kp: %i" % (i, bd.keypoints.size))
    print     
    
# bd._one_octave(False, True ,False)
    
print ('Final size of keypoints : %i'% bd.keypoints.size)

i = 0
# for kp  in bd.keypoints:
#     ds = kp.scale
#     ax.annotate("", xy=(kp.x, kp.y), xytext=(kp.x+ds, kp.y+ds),
#                 arrowprops=dict(facecolor='blue', shrink=0.05),)

ax.plot(bd.keypoints.x, bd.keypoints.y, "og")

scales = bd.keypoints.scale
if scales.size > 0:
    h = pylab.figure(2)
    x,y,o = pylab.hist(numpy.sqrt(scales), bins = 100)
    h.show()
      
    index = numpy.where(x == x.max())
    kp = bd.keypoints[bd.keypoints.scale > y[index]]
else : kp = bd.keypoints

# pylab.figure()
# pylab.imshow(numpy.log1p(data), interpolation = 'nearest')
# pylab.plot(kp.x,kp.y,'og')

f.show()
raw_input()
