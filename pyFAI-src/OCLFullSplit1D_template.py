class OCLFullSplit1d(object):
    def __init__(self,
                 pos,
                 int bins=100,
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 block_size=256,
                 platformid=None,
                 deviceid=None,
                 profile=False):
        
        
        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg

        if len(pos.shape) == 3:
            assert pos.shape[1] == 4
            assert pos.shape[2] == 2
        elif len(pos.shape) == 4:
            assert pos.shape[2] == 4
            assert pos.shape[3] == 2
        else:
            raise ValueError("Pos array dimentions are wrong")
        self.size = pos.size/8

        if  mask is not None:
            assert mask.size == self.size
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None
            
        self.pos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        self.pos0Range = numpy.empty(2,dtype=numpy.float32)
        self.pos1Range = numpy.empty(2,dtype=numpy.float32)
        
        if (pos0Range is not None) and (len(pos0Range) is 2):
            self.pos0Range[0] = min(pos0Range) # do it on GPU?
            self.pos0Range[1] = max(pos0Range)
            if (not self.allow_pos0_neg) and (self.pos0Range[0] < 0):
                self.pos0Range[0] = 0.0
                if self.pos0Range[1] < 0:
                    print "Warning: Invalid 0-dim range! Using the data derived range instead"
                    self.pos0Range[1] = 0.0
            #self.pos0Range[0] = pos0Range[0]
            #self.pos0Range[1] = pos0Range[1]
        else:
            self.pos0Range[0] = 0.0
            self.pos0Range[1] = 0.0
            
            
        if (pos1Range is not None) and (len(pos1Range) is 2):
            self.pos1Range[0] = min(pos1Range) # do it on GPU?
            self.pos1Range[1] = max(pos1Range)
            #self.pos1Range[0] = pos1Range[0]
            #self.pos1Range[1] = pos1Range[1]
        else:
            self.pos1Range[0] = 0.0
            self.pos1Range[1] = 0.0
            
        
        #tmp0 = (pos0Range is not None) and (len(pos0Range) is 2)
        #tmp1 = (pos1Range is not None) and (len(pos1Range) is 2)
        #if (not tmp0) and (not tmp1):
            #self._minMax()
        #elif (tmp0) and (not tmp1):
            #self._minMax()
            ##pull d_minmax, replace pos0, send back
            #self.pos0_min = min(pos0Range)
            #if (not allow_pos0_neg) and self.pos0_min < 0:
                #self.pos0_min = 0
            #self.pos0_maxin = max(pos0Range)
        #elif (not tmp0) and (tmp1):
            #self._minMax()
            ##pull d_minmax, replace pos1, send back
            #self.pos1_min = min(pos1Range)
            #self.pos1_maxin = max(pos1Range)
            
        #else:
            ##allocate d_minmax
            #self.pos0_min = min(pos0Range)
            #if (not allow_pos0_neg) and self.pos0_min < 0:
                #self.pos0_min = 0
            #self.pos0_maxin = max(pos0Range)
            #self.pos1_min = min(pos1Range)
            #self.pos1_maxin = max(pos1Range)
        
        self._minMax()

        
        #define workgroup_size
        #define bins
        #define error tolerance
        #define pos_size
        #run _minMax
        
        
        
    def _minMax(self):
        #check memory of d_pos + d_preresult + d_minmax
        #load d_pos
        #allocate d_preresult
        #allocate d_minmax
        #run reduce1
        #run reduce2
        #save reference to d_minMax
        #free d_preresult
        
        
    def _calcLUT(self):
        #check memory of d_pos + d_minmax + d_outMax + d_lutsize
        #allocate d_outMax
        #allocate d_lutsize
        #memset d_outMax
        #run lut1
        #run lut2
        #save d_lutsize
        #memset d_outMax
        #allocate d_data
        #allocate d_indices
        #run lut3
        #free d_pos
        #free d_minMax
        #free d_lutsize
        #run lut4
        #free d_outMax
    
    