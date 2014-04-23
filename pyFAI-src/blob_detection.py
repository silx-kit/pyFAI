import numpy, itertools,scipy
try:
    from _convolution import gaussian_filter
except ImportError:
    from scipy.ndimage.filters import gaussian_filter
try:
    from . import _blob
except ImportError:
    _blob = None

try:
    from . import morphology
except ImportError:
    from scipy.ndimage import morphology
    pyFAI_morphology = False
else:
    pyFAI_morphology = True

from math import sqrt

from .utils import binning, timeit

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
    size =int( 8*sigma +1 )
    if size%2 == 0 :
           size += 1
    x = numpy.arange(0, size, 1, float)
    y = x[:,numpy.newaxis]
    x0 = y0 = size // 2
    gaus = numpy.exp(-4*numpy.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)
    im[xc-size/2:xc+size/2+1,yc-size/2:yc+size/2+1] = gaus
    return im

@timeit
def local_max(dogs, mask=None, n_5=True):
    """
    @param dogs: 3d array with (sigma,y,x) containing difference of gaussians 
    @parm mask: mask out keypoint next to the mask (or inside the mask)
    @param n_5: look for a larger neighborhood
    """
    ns = dogs.shape[0]
    kpma = numpy.zeros(shape=dogs.shape, dtype=numpy.uint8)
    for i in range(1, ns - 1):
        cur_dog = dogs[i]
        next_dog = dogs[i+1]
        prev_dog = dogs[i-1]
        slic = cur_dog[1:-1, 1:-1]
        kpm = kpma[i] 
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
        
        
        if n_5:
            target = 38
            slic = cur_dog[2:-2,2:-2]
            
            kpm[2:-2,2:-2] += (slic > cur_dog[:-4, 2:-2]) * (slic > cur_dog[4:, 2:-2]) #decalage horizontal
            kpm[2:-2, 2:-2] += (slic > cur_dog[2:-2, :-4]) * (slic > cur_dog[2:-2, 4:]) #decalage vertical
            kpm[2:-2, 2:-2] += (slic > cur_dog[:-4, :-4]) * (slic > cur_dog[4:, 4:])   #diagonale 
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, :-4]) * (slic > cur_dog[:-4, 4:])    
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, 1:-3]) * (slic > cur_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > cur_dog[1:-3, :-4]) * (slic > cur_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > cur_dog[3:-1, :-4]) * (slic > cur_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, 3:-1]) * (slic > cur_dog[:-4, 3:-1])
        
            #with next DoG
            kpm[2:-2, 2:-2] += (slic > next_dog[:-4, 2:-2]) * (slic > next_dog[4:, 2:-2])
            kpm[2:-2, 2:-2] += (slic > next_dog[2:-2, :-4]) * (slic > next_dog[2:-2, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[:-4, :-4]) * (slic > next_dog[4:, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, :-4]) * (slic > next_dog[:-4, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, 1:-3]) * (slic > next_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > next_dog[1:-3, :-4]) * (slic > next_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[3:-1, :-4]) * (slic > next_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, 3:-1]) * (slic > next_dog[:-4, 3:-1])
        
            #with previous DoG
            kpm[2:-2, 2:-2] += (slic > prev_dog[:-4, 2:-2]) * (slic > prev_dog[4:, 2:-2])
            kpm[2:-2, 2:-2] += (slic > prev_dog[2:-2, :-4]) * (slic > prev_dog[2:-2, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[:-4, :-4]) * (slic > prev_dog[4:, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, :-4]) * (slic > prev_dog[:-4, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, 1:-3]) * (slic > prev_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > prev_dog[1:-3, :-4]) * (slic > prev_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[3:-1, :-4]) * (slic > prev_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, 3:-1]) * (slic > prev_dog[:-4, 3:-1])
                    
        else: 
            target=14           
        if mask is not None:
            kpm +=mask
    return kpms == target


class BlobDetection(object):
    """
    
    """
    def __init__(self, img, cur_sigma=0.25, init_sigma=0.50, dest_sigma=1, scale_per_octave=2, mask=None):
        """
        Performs a blob detection:
        http://en.wikipedia.org/wiki/Blob_detection
        using a Difference of Gaussian + Pyramid of Gaussians
        
        @param img: input image
        @param cur_sigma: estimated smoothing of the input image. 0.25 correspond to no interaction between pixels.
        @param init_sigma: start searching at this scale (sigma=0.5: 10% interaction with first neighbor)
        @param dest_sigma: sigma at which the resolution is lowered (change of octave)
        @param scale_per_octave: Number of scale to be performed per octave
        @param mask: mask where pixel are not valid
        """
        self.raw = numpy.log(img.astype(numpy.float32))
        self.cur_sigma = float(cur_sigma)
        self.init_sigma = float(init_sigma)
        self.dest_sigma = float(dest_sigma)
        self.scale_per_octave = int(scale_per_octave)
        if mask is not None:
            self.mask = (mask != 0).astype(numpy.int8)
        else:
            self.mask = (img <= 0).astype(numpy.int8)
        to_mask = numpy.where(self.mask)
        self.do_mask = to_mask[0].size > 0
        if self.do_mask:
            self.raw[to_mask] = 0

            #initial grow of 4*sigma_dest ... subsequent re-grow of half
            grow = int(4.0 * self.dest_sigma)
            if not pyFAI_morphology:
                my, mx = numpy.ogrid[-grow:grow + 1, -grow:grow + 1]
                grow = (mx * mx + my * my) <= grow * grow
            self.cur_mask = morphology.binary_dilation(self.mask, grow)
            #subsequent grow
            grow = int(2.0 * self.dest_sigma)
            if not pyFAI_morphology:
                my, mx = numpy.ogrid[-grow:grow + 1, -grow:grow + 1]
                grow = (mx * mx + my * my) <= grow * grow
            self.grow = grow

        self.data = None    # current image
        self.sigmas = None  # contains pairs of absolute sigma and relative ones...
        self.blurs = []     # different blurred images
        self.dogs = []      # different difference of gaussians
        self.dogs_init = []
        self.border_size = 5# size of the border
        self.keypoints = []
        self.delta = []
        self.sigma_octave = 1.0

    def _initial_blur(self):
        """
        Blur the original image to achieve the requested level of blur init_sigma
        """
        if self.init_sigma > self.cur_sigma:
            sigma = sqrt(self.init_sigma * self.init_sigma - self.cur_sigma * self.cur_sigma)
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
        print(self.sigmas)

    @timeit
    def grow_mask(self):
        """
        Grow the mask according to the given sigma_destination
        
        At each octave, one deeds to re-grow the mask by half the size 
        """

    @timeit
    def _one_octave(self, shrink=True, do_SG4=True, n_5=False):
        """
        Return the blob coordinates for an octave
        
        @param shrink: perform the image shrinking after the octave processing
        @param   do_SG4: perform Savitsky-Golay 4th order fit. 
        """
        x=[]
        y=[]
        dx=[]
        dy=[]
        if not self.sigmas:
            self._calc_sigma()

        previous = self.data
        dog_shape = (len(self.sigmas) - 1,) + self.data.shape
        self.dogs = numpy.zeros(dog_shape, dtype=numpy.float32)
        
        idx = 0
        for sigma_abs, sigma_rel in self.sigmas:
            if  sigma_rel == 0:
                self.blurs.append(previous)
            else:
                new_blur = gaussian_filter(previous, sigma_rel)
                self.blurs.append(new_blur)
                self.dogs[idx] = previous - new_blur
                previous = new_blur
                idx += 1
                
        
        if self.dogs[0].shape == self.raw.shape:
            self.dogs_init = self.dogs
        
        if _blob:
            valid_points = _blob.local_max(self.dogs, self.cur_mask, n_5)
        else:
            valid_points = local_max(self.dogs, self.cur_mask, n_5)
        kps, kpy, kpx = numpy.where(valid_points)
        l = kpx.size
        
#         print 'shape of dogs'
#         print self.dogs.shape
        
        dtype = numpy.dtype([('x', numpy.float32), ('y', numpy.float32), ('scale', numpy.float32), ('I', numpy.float32)])
        keypoints = numpy.recarray((l,), dtype=dtype)
        sigmas = numpy.array([s[0] for s in self.sigmas])
        
        keypoints[:].x = kpx * self.sigma_octave
        keypoints[:].y = kpy * self.sigma_octave
        keypoints[:].scale = (self.sigma_octave * sigmas.take(kps)) ** 2 #scale = sigma^2
        keypoints[:].I = self.dogs[(kps, kpy, kpx)]

        print keypoints[:].scale

        if do_SG4:
            print keypoints.shape
            keypoints = self.refine_SG4(keypoints)
#                kx, ky, sigma, dlx, dly = self.refine_SG4(i, loc_keypoint[:].x, loc_keypoint[:].y, loc_keypoint[:].sigma)
#            else:
#                kx = loc_keypoint[:].x
#                ky = loc_keypoint[:].y
#                sigma = loc_keypoint[:].sigma
#                dlx = numpy.zeros_like(kx)
#                dly = numpy.zeros_like(ky)
#            x.append(kx)
#            y.append(ky)
#            dx.append(dlx)
#            dy.append(dly)
#            sigmas.append(sigma)
#            print(kx.__len__(), ky.__len__(), sigma.__len__())

        if shrink:
            #shrink data so that they can be treated by next octave
            print("In shrink")
            self.data = binning(self.blurs[self.scale_per_octave], 2) / 4.0
            self.sigma_octave *= 2
            self.blurs = []
            if self.do_mask:
                self.cur_mask = binning(self.cur_mask, 2).astype(numpy.int8) / 4
                self.cur_mask = morphology.binary_dilation(self.cur_mask, self.grow)
                
        if self.keypoints == [] : self.keypoints = keypoints 
        else: self.keypoints = numpy.concatenate((self.keypoints, keypoints))
#         self.keypoints = keypoints 
#        return numpy.concatenate(x), \
#               numpy.concatenate(y), \
#               numpy.concatenate(dx), \
#               numpy.concatenate(dy), \
#               numpy.concatenate(sigmas)
    
  
    def refine_SG4(self,keypoints):
        """ Savitzky Golay algorithm to check if a point is really the maximum """

#         deltax=[]
#         deltay=[]
#         k2x=[]
#         k2y=[]
#         sigmas=[]
        i=0
        l = keypoints[:].x.shape[0]
        print 'Before refinement'
        print l
        dtype = numpy.dtype([('x', numpy.float32), ('y', numpy.float32), ('scale', numpy.float32), ('I', numpy.float32),('valid',numpy.bool)])
        kps = numpy.recarray((l,), dtype=dtype)
        kps[:].x = keypoints[:].x
        kps[:].y = keypoints[:].y
        kps[:].I = keypoints[:].I
        kps[:].valid = False
        
        #savitsky golay ordre 4 patch 5
        SGX0Y0 = [0.04163265,-0.08081633,0.07836735,-0.08081633,0.04163265,-0.08081633,-0.01959184,0.20081633,-0.01959184,-0.08081633,0.07836735,0.20081633,0.44163265,0.20081633,0.07836735,-0.08081633,-0.01959184,0.20081633,-0.01959184,-0.08081633,0.04163265,-0.08081633,0.07836735,-0.08081633,0.04163265]
        SGX1Y0 = [0.07380952,-0.10476190,0.00000000,0.10476190,-0.07380952,-0.01190476,-0.14761905,0.00000000,0.14761905,0.01190476,-0.04047619,-0.16190476,0.00000000,0.16190476,0.04047619,-0.01190476,-0.14761905,0.00000000,0.14761905,0.01190476,0.07380952,-0.10476190,0.00000000,0.10476190,-0.07380952]
        SGX2Y0 = [-0.04914966,0.15374150,-0.20918367,0.15374150,-0.04914966,0.01207483,0.12312925,-0.27040816,0.12312925,0.01207483,0.03248299,0.11292517,-0.29081633,0.11292517,0.03248299,0.01207483,0.12312925,-0.27040816,0.12312925,0.01207483,-0.04914966,0.15374150,-0.20918367,0.15374150,-0.04914966]
        SGX0Y1 = [0.07380952,-0.01190476,-0.04047619,-0.01190476,0.07380952,-0.10476190,-0.14761905,-0.16190476,-0.14761905,-0.10476190,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,0.10476190,0.14761905,0.16190476,0.14761905,0.10476190,-0.07380952,0.01190476,0.04047619,0.01190476,-0.07380952]
        SGX1Y1 = [-0.07333333,0.10500000,0.00000000,-0.10500000,0.07333333,0.10500000,0.12333333,0.00000000,-0.12333333,-0.10500000,0.00000000,0.00000000,0.00000000,0.00000000,0.00000000,-0.10500000,-0.12333333,0.00000000,0.12333333,0.10500000,0.07333333,-0.10500000,0.00000000,0.10500000,-0.07333333]
        SGX0Y2 = [-0.04914966,0.01207483,0.03248299,0.01207483,-0.04914966,0.15374150,0.12312925,0.11292517,0.12312925,0.15374150,-0.20918367,-0.27040816,-0.29081633,-0.27040816,-0.20918367,0.15374150,0.12312925,0.11292517,0.12312925,0.15374150,-0.04914966,0.01207483,0.03248299,0.01207483,-0.04914966]

        for y,x in itertools.izip(keypoints[:].y,keypoints[:].x):
            scale = keypoints[i].scale
#             print scale,self.sigmas[0][0],self.sigmas[self.scale_per_octave+2][0]
            j = round(numpy.log(numpy.sqrt(scale)/self.sigmas[0][0])/numpy.log(2)*self.scale_per_octave)
#             print j,self.scale_per_octave
            if j > 0 and j < self.scale_per_octave+1:
#                 print 'ici'
                curr_dog = self.dogs_init[j]
                prev_dog = self.dogs_init[j-1]
                next_dog = self.dogs_init[j+1]

                if (x > 1 and x < curr_dog.shape[1]-2 and y > 1 and y < curr_dog.shape[0]-2):
                
                    
                    patch5 = curr_dog[y-2:y+3,x-2:x+3]
                    patch5_prev = prev_dog[y-2:y+3,x-2:x+3]
                    patch5_next = next_dog[y-2:y+3,x-2:x+3]
    
    #                 print curr_dog.shape[1]-3,x,y,patch5
                    dx = (SGX1Y0*patch5.ravel()).sum()
                    dy = (SGX0Y1*patch5.ravel()).sum()
                    d2x = (SGX2Y0*patch5.ravel()).sum()
                    d2y = (SGX0Y2*patch5.ravel()).sum()
                    dxy = (SGX1Y1*patch5.ravel()).sum()
    
                    s_next = (SGX0Y0*patch5_next.ravel()).sum()
                    s = (SGX0Y0*patch5.ravel()).sum()
                    s_prev = (SGX0Y0*patch5_prev.ravel()).sum()
                    d2s = (s_next + s_prev - 2.0*s) /4.0
                    ds = (s_next - s_prev) /2.0
                    
                    dx_next = (SGX1Y0*patch5_next.ravel()).sum()
                    dx_prev = (SGX1Y0*patch5_prev.ravel()).sum()
                    
                    dy_next = (SGX0Y1*patch5_next.ravel()).sum()
                    dy_prev = (SGX0Y1*patch5_prev.ravel()).sum()
                    
                    dxs = (dx_next - dx_prev)/2.0
                    dys = (dy_next - dy_prev)/2.0                
                                    
                    lap = numpy.array([[d2y,dxy,dys],[dxy,d2x,dxs],[dys,dxs,d2s]])
    
                    delta = (numpy.dot(numpy.linalg.inv(lap),[dy,dx,ds]))
                    err = numpy.linalg.norm(delta[:-1])
                    if  err < numpy.sqrt(2) and numpy.abs(dx) < 1 and numpy.abs(dy) < 1 :
#                         print delta
#                         k2x.append(x-delta[1])
#                         k2y.append(y-delta[0])
#                         sigmas.append(sigma-delta[2])
#                         deltax.append(-delta[1])
#                         deltay.append(-delta[0])
                        kps[i].x = keypoints[i].x-delta[1]
                        kps[i].y =  keypoints[i].y-delta[0]
                        kps[i].scale = (numpy.sqrt(keypoints[i].scale)-delta[2]) ** 2
                        kps[i].I = keypoints[i].I
                        kps[i].valid = True
                    
                    i = i + 1
        
#         return k2x,k2y,sigmas,deltax,deltay
        res = numpy.where(kps[:].valid == True)
        print 'After SG refinement'
        print kps[res].shape
        return kps[res]                     
                
        
if __name__ == "__main__":
    
    kx=[]
    ky=[]
    k2x=[]
    k2y=[]
    dx=[]
    dy=[]
    

    import fabio,pylab
#     img = fabio.open("../test/testimages/LaB6_0003.mar3450").data
#     img = fabio.open("../test/testimages/grid2k0000.edf").data
    img = fabio.open("../test/testimages/halfccd.edf").data
    img = numpy.log1p(img)
#     img = img[img.shape[0]/2-256:img.shape[0]/2+256,img.shape[1]/2-256:img.shape[1]/2+256]
#     img = image_test()

    bd = BlobDetection(img)
    kx,ky,dx,dy,sigma = bd._one_octave()
    print bd.sigmas
    
    #building histogram with the corrected sigmas
    sigma = numpy.asarray(sigma)
    pylab.figure(2)
    pylab.clf()
    pylab.hist(sigma, bins = 500)
    pylab.show()
    
    
    h = pylab.hist(sigma, bins = 500)
    n = h[0].__len__()
    Proba = h[0]/float(numpy.sum(h[0]))
#     print Proba.size,numpy.max(Proba)
    
    max = 0.0
#     print n
    
    for cpt in range( n ):
#         print cpt
        Proba1 = Proba[: cpt]
        Proba2 = Proba[cpt :]        
        P1 = numpy.sum(Proba1)
        P2 = numpy.sum(Proba2)
#         print P1,P2
        
        n1 = numpy.arange(cpt)
        n2 = numpy.arange(cpt,n)        
        Moy1 = sum( n1 * Proba1)/P1
        Moy2 = sum( n2 * Proba2)/P2
#         print "Moyennes"
#         print Moy1,Moy2
        
        VarInterC = P1*P2*(Moy1-Moy2)**2
#         print "Variance IC"
#         print VarInterC
        
        if VarInterC > max : 
            max = VarInterC
            index = cpt
           
#     print max,cpt
    print 'sigma pour la separation'
    print h[1][index]
#  building arrays x and y containing all the coordinates of the keypoints, only for vizualisation
    x=[]
    y=[]
    print bd.keypoints.__len__()
    for j in range( bd.keypoints.__len__() ):
        k = bd.keypoints[j]
        x.extend(numpy.transpose(k)[0])
        y.extend(numpy.transpose(k)[1])
         
     
    print x.__len__(), y.__len__(), kx.__len__(), ky.__len__()
    
    pylab.figure(1)
    pylab.clf()
    pylab.imshow((img),interpolation='nearest')
    pylab.plot(x,y,'or')
    pylab.show()
     
