
Modeling of the thickness of the sensor
=======================================

In this notebook we will re-use the experiment done at ID28 and
previously calibrated and model in 3D the detector.

This detector is a Pilatus 1M with a 450µm thick silicon sensor. Let's
first have a look at the absorption coefficients of this sensor
material:
https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z14.html

First we retieve the results of the previous step, then calculate the
absorption efficiency:

.. code:: ipython3

    %pylab nbagg


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: ipython3

    import numpy
    import fabio, pyFAI
    import json
    with open("id28.json") as f:
        calib = json.load(f)
    
    thickness = 450e-6
    wavelength = calib["wavelength"]
    dist = calib["param"][calib['param_names'].index("dist")]
    poni1 = calib["param"][calib['param_names'].index("poni1")]
    poni2 = calib["param"][calib['param_names'].index("poni2")]
    energy = pyFAI.units.hc/(wavelength*1e10)
    print("wavelength: %.3em,\t dist: %.3em,\t poni1: %.3em,\t poni2: %.3em,\t energy: %.3fkeV" % 
          (wavelength, dist, poni1, poni2, energy))



.. parsed-literal::

    wavelength: 6.968e-11m,	 dist: 2.845e-01m,	 poni1: 8.865e-02m,	 poni2: 8.931e-02m,	 energy: 17.793keV


Absorption coeficient at 17.8 keV
---------------------------------

.. code:: ipython3

    # density from https://en.wikipedia.org/wiki/Silicon
    rho = 2.3290 # g/cm^3
    
    #Absorption from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z14.html
    # Nota: enegies are in MeV !
    Si_abs = """
       2.00000E-03  2.777E+03  2.669E+03 
       3.00000E-03  9.784E+02  9.516E+02 
       4.00000E-03  4.529E+02  4.427E+02 
       5.00000E-03  2.450E+02  2.400E+02 
       6.00000E-03  1.470E+02  1.439E+02 
       8.00000E-03  6.468E+01  6.313E+01 
       1.00000E-02  3.389E+01  3.289E+01 
       1.50000E-02  1.034E+01  9.794E+00 
       2.00000E-02  4.464E+00  4.076E+00 
       3.00000E-02  1.436E+00  1.164E+00 
       4.00000E-02  7.012E-01  4.782E-01 
       5.00000E-02  4.385E-01  2.430E-01 
       6.00000E-02  3.207E-01  1.434E-01 
       8.00000E-02  2.228E-01  6.896E-02 
       1.00000E-01  1.835E-01  4.513E-02 
       1.50000E-01  1.448E-01  3.086E-02 
       2.00000E-01  1.275E-01  2.905E-02 
       3.00000E-01  1.082E-01  2.932E-02 
       4.00000E-01  9.614E-02  2.968E-02 
       5.00000E-01  8.748E-02  2.971E-02 
       6.00000E-01  8.077E-02  2.951E-02 
       8.00000E-01  7.082E-02  2.875E-02 
       1.00000E+00  6.361E-02  2.778E-02 
       1.25000E+00  5.688E-02  2.652E-02 
       1.50000E+00  5.183E-02  2.535E-02 
       2.00000E+00  4.480E-02  2.345E-02 
       3.00000E+00  3.678E-02  2.101E-02 
       4.00000E+00  3.240E-02  1.963E-02 
       5.00000E+00  2.967E-02  1.878E-02 
       6.00000E+00  2.788E-02  1.827E-02 
       8.00000E+00  2.574E-02  1.773E-02 
       1.00000E+01  2.462E-02  1.753E-02 
       1.50000E+01  2.352E-02  1.746E-02 
       2.00000E+01  2.338E-02  1.757E-02 """
    data = numpy.array([[float(i) for i in line.split()] for line in Si_abs.split("\n") if line])
    energy_tab, mu_over_rho, mu_en_over_rho = data.T
    abs_18 = numpy.interp(energy, energy_tab*1e3, mu_en_over_rho) 
    mu = abs_18*rho*1e+2
    eff = 1.0-numpy.exp(-mu*thickness)
    
    print("µ = %f m^-1 hence absorption efficiency for 450µm: %.1f %%"%(mu, eff*100))



.. parsed-literal::

    µ = 1537.024385 m^-1 hence absorption efficiency for 450µm: 49.9 %


.. code:: ipython3

    depth = numpy.linspace(0, 1000, 100)
    res = numpy.exp(-mu*depth*1e-6)
    fig, ax = subplots()
    ax.plot(depth, res, "-")
    ax.set_xlabel("Depth (µm)")
    ax.set_ylabel("Residual signal")
    ax.set_title("Silicon @ 17.8 keV")




.. parsed-literal::

    <IPython.core.display.Javascript object>



.. raw:: html

    <img src="data:image/png;base64,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" width="640">




.. parsed-literal::

    Text(0.5,1,'Silicon @ 17.8 keV')



This is consistent with:
http://henke.lbl.gov/optical\_constants/filter2.html

Now we can model the detector

Modeling of the detector:
-------------------------

The detector is seen as a 2D array of voxel. Let vox, voy and voz be the
dimention of the detector in the three dimentions.

.. code:: ipython3

    detector= pyFAI.detector_factory(calib["detector"])
    print(detector)
    
    vox = detector.pixel2 # this is not a typo
    voy = detector.pixel1 # x <--> axis 2
    voz = thickness
    
    print(vox, voy, voz)


.. parsed-literal::

    Detector Pilatus 1M	 PixelSize= 1.720e-04, 1.720e-04 m
    0.000172 0.000172 0.00045


The intensity grabbed in this voxel is the triple integral of the
absorbed signal coming from this pixel or from the neighboring ones.

There are 3 ways to perform this intergral: \* Volumetric analytic
integral. Looks feasible with a change of variable in the depth \* Slice
per slice, the remaining intensity depand on the incidence angle + pixel
splitting between neighbooring pixels \* raytracing: the decay can be
solved analytically for each ray, one has to throw many ray to average
out the signal.

For sake of simplicity, this integral will be calculated numerically
using this raytracing algorithm.
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf

Knowing the input position for a X-ray on the detector and its
propagation vector, this algorithm allows us to calculate the length of
the path in all voxel it crosses in a fairly efficient way.

To speed up the calculation, we will use a few tricks: \* One ray never
crosses more than 16 pixels, which is reasonable considering the
incidance angle \* we use numba to speed-up the calculation of loops in
python \* We will allocate the needed memory by chuncks of 1 million
elements

.. code:: ipython3

    from numba import jit 
    
    BLOCK_SIZE = 1<<20 # 1 milion
    BUFFER_SIZE = 16 
    BIG = numpy.finfo(numpy.float32).max
    
    mask = numpy.load("mask.npy").astype(numpy.int8)
    from scipy.sparse import csr_matrix, csc_matrix, linalg

.. code:: ipython3

    @jit
    def calc_one_ray(entx, enty, 
                     kx, ky, kz,
                     vox, voy, voz):
        """For a ray, entering at position (entx, enty), with a propagation vector (kx, ky,kz),
        calculate the length spent in every voxel where energy is deposited from a bunch of photons comming in the detector 
        at a given position and and how much energy they deposit in each voxel. 
        
        Direct implementation of http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
        
        :param entx, enty: coordinate of the entry point in meter (2 components, x,y)
        :param kx, ky, kz: vector with the direction of the photon (3 components, x,y,z)
        :param vox, voy, voz: size of the voxel in meter (3 components, x,y,z)
        :return: coordinates voxels in x, y and length crossed when leaving the associated voxel
        """
        array_x = numpy.empty(BUFFER_SIZE, dtype=numpy.int32)
        array_x[:] = -1
        array_y = numpy.empty(BUFFER_SIZE, dtype=numpy.int32)
        array_y[:] = -1
        array_len = numpy.empty(BUFFER_SIZE, dtype=numpy.float32)
        
        #normalize the input propagation vector
        n = numpy.sqrt(kx*kx + ky*ky + kz*kz)
        kx /= n
        ky /= n
        kz /= n
        
        assert kz>0
        step_X = -1 if kx<0.0 else 1
        step_Y = -1 if ky<0.0 else 1
        
        assert vox>0
        assert voy>0
        assert voz>0
            
        X = int(entx//vox)
        Y = int(enty//voy)
        
        if kx>0.0:
            t_max_x = ((entx//vox+1)*(vox)-entx)/ kx
        elif kx<0.0:
            t_max_x = ((entx//vox)*(vox)-entx)/ kx
        else:
            t_max_x = BIG
    
        if ky>0.0:
            t_max_y = ((enty//voy+1)*(voy)-enty)/ ky
        elif ky<0.0:
            t_max_y = ((enty//voy)*(voy)-enty)/ ky
        else:
            t_max_y = BIG
        
        #Only one case for z as the ray is travelling in one direction only
        t_max_z = voz / kz
           
        t_delta_x = abs(vox/kx) if kx!=0 else BIG
        t_delta_y = abs(voy/ky) if ky!=0 else BIG
        t_delta_z = voz/kz
        
        finished = False
        last_id = 0
        array_x[last_id] = X
        array_y[last_id] = Y
        
        while not finished:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    array_len[last_id] = t_max_x
                    last_id+=1
                    X += step_X
                    array_x[last_id] = X
                    array_y[last_id] = Y
                    t_max_x += t_delta_x
                else:
                    array_len[last_id] = t_max_z
                    finished = True
            else:
                if t_max_y < t_max_z:
                    array_len[last_id] = t_max_y
                    last_id+=1
                    Y += step_Y
                    array_x[last_id] = X
                    array_y[last_id] = Y                
                    t_max_y += t_delta_y
                else:
                    array_len[last_id] = t_max_z
                    finished = True
            if last_id>=array_len.size-1:
                print("resize arrays")
                old_size = len(array_len)
                new_size = (old_size//BUFFER_SIZE+1)*BUFFER_SIZE
                new_array_x = numpy.empty(new_size, dtype=numpy.int32)
                new_array_x[:] = -1
                new_array_y = numpy.empty(new_size, dtype=numpy.int32)
                new_array_y[:] = -1
                new_array_len = numpy.empty(new_size, dtype=numpy.float32)
                new_array_x[:old_size] = array_x
                new_array_y[:old_size] = array_y
                new_array_len[:old_size] = array_len
                array_x = new_array_x
                array_y = new_array_y
                array_len = new_array_len
        return array_x[:last_id], array_y[:last_id], array_len[:last_id]
    
    print(calc_one_ray(0.0,0.0, 1,1,1, 172e-6, 172e-6, 450e-6))
    import random
    %timeit calc_one_ray(10+random.random(),11+random.random(),\
                         random.random()-0.5,random.random()-0.5,0.5+random.random(), \
                         vox, voy, voz)
    %timeit calc_one_ray.py_func(10+random.random(),11+random.random(),\
                         random.random()-0.5,random.random()-0.5,0.5+random.random(), \
                         vox, voy, voz)


.. parsed-literal::

    (array([0, 0, 1, 1], dtype=int32), array([0, 1, 1, 2], dtype=int32), array([ 0.00029791,  0.00029791,  0.00059583,  0.00059583], dtype=float32))
    The slowest run took 7.37 times longer than the fastest. This could mean that an intermediate result is being cached.
    5.24 µs ± 5.85 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    9.47 µs ± 44 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


Now that we are able to perform raytracing for any ray comming in the
detector, we can calculate the contribution to the neighboring pixels,
using the absorption law (the length travelled is already known). To
average-out the signal, we will sample a few dozens of rays per pixel to
get an approximatation of the volumic integrale.

Now we need to store the results so that this transformation can be
represented as a sparse matrix multiplication:

b = M.a

Where b is the recorded image (blurred) and a is the "perfect" signal. M
being the sparse matrix where every pixel of a gives a limited number of
contribution to b.

Each pixel in *b* is represented by one line in *M* and we store the
indices of *a* of interest with the coefficients of the matrix. So if a
pixel i,j contributes to (i,j), (i+1,j), (i+1,j+1), there are only 3
elements in the line. This is advantagous for storage.

We will use the CSR sparse matrix representation:
https://en.wikipedia.org/wiki/Sparse\_matrix#Compressed\_sparse\_row\_.28CSR.2C\_CRS\_or\_Yale\_format.29
where there are 3 arrays: \* data: containing the actual non zero values
\* indices: for a given line, it contains the column number of the
assocated data (at the same indice) \* idptr: this array contains the
index of the start of every line.

.. code:: ipython3

    from numba import jitclass, int8, int32, int64, float32, float64
    spec = [("vox",float64),("voy",float64),("voz",float64),("mu",float64),
            ("dist",float64),("poni1",float64),("poni2",float64),
            ("width", int64),("height", int64),("mask", int8[:,:]),
            ("sampled", int64), ("data", float32[:]),("indices", int32[:]),("idptr", int32[:]),
           ]
    @jitclass(spec)
    class ThickDetector(object):
        "Calculate the point spread function as function of the geometry of the experiment"
        
        def __init__(self, vox, voy, thickness, mask, mu, 
                     dist, poni1, poni2):
            """Constructor of the class:
            
            :param vox, voy: detector pixel size in the plane
            :param thickness: thickness of the sensor in meters
            :param mask: 
            :param mu: absorption coefficient of the sensor material
            :param dist: sample detector distance as defined in the geometry-file
            :param poni1, poni2: coordinates of the PONI as defined in the geometry 
            """
            self.vox = vox
            self.voy = voy
            self.voz = thickness
            self.mu = mu
            self.dist=dist
            self.poni1 = poni1
            self.poni2 = poni2
            self.width = mask.shape[-1]
            self.height = mask.shape[0]
            self.mask = mask
            self.sampled = 0
            self.data = numpy.zeros(BLOCK_SIZE, dtype=numpy.float32)
            self.indices = numpy.zeros(BLOCK_SIZE,dtype=numpy.int32)
            self.idptr = numpy.zeros(self.width*self.height+1, dtype=numpy.int32)
            
        def calc_one_ray(self, entx, enty):
            """For a ray, entering at position (entx, enty), with a propagation vector (kx, ky,kz),
            calculate the length spent in every voxel where energy is deposited from a bunch of photons comming in the detector 
            at a given position and and how much energy they deposit in each voxel. 
    
            Direct implementation of http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
    
            :param entx, enty: coordinate of the entry point in meter (2 components, x,y)
            :return: coordinates voxels in x, y and length crossed when leaving the associated voxel
            """
            array_x = numpy.empty(BUFFER_SIZE, dtype=numpy.int32)
            array_x[:] = -1
            array_y = numpy.empty(BUFFER_SIZE, dtype=numpy.int32)
            array_y[:] = -1
            array_len = numpy.empty(BUFFER_SIZE, dtype=numpy.float32)
    
            #normalize the input propagation vector
            kx = entx - self.poni2
            ky = enty - self.poni1
            kz = self.dist
            n = numpy.sqrt(kx*kx + ky*ky + kz*kz)
            kx /= n
            ky /= n
            kz /= n
    
            step_X = -1 if kx<0.0 else 1
            step_Y = -1 if ky<0.0 else 1
    
            X = int(entx/self.vox)
            Y = int(enty/self.voy)
    
            if kx>0.0:
                t_max_x = ((entx//self.vox+1)*(self.vox)-entx)/ kx
            elif kx<0.0:
                t_max_x = ((entx//self.vox)*(self.vox)-entx)/ kx
            else:
                t_max_x = BIG
    
            if ky>0.0:
                t_max_y = ((enty//self.voy+1)*(self.voy)-enty)/ ky
            elif ky<0.0:
                t_max_y = ((enty//self.voy)*(self.voy)-enty)/ ky
            else:
                t_max_y = BIG
    
            #Only one case for z as the ray is travelling in one direction only
            t_max_z = self.voz / kz
    
            t_delta_x = abs(self.vox/kx) if kx!=0 else BIG
            t_delta_y = abs(self.voy/ky) if ky!=0 else BIG
            t_delta_z = self.voz/kz
    
            finished = False
            last_id = 0
            array_x[last_id] = X
            array_y[last_id] = Y
    
            while not finished:
                if t_max_x < t_max_y:
                    if t_max_x < t_max_z:
                        array_len[last_id] = t_max_x
                        last_id+=1
                        X += step_X
                        array_x[last_id] = X
                        array_y[last_id] = Y
                        t_max_x += t_delta_x
                    else:
                        array_len[last_id] = t_max_z
                        last_id+=1
                        finished = True
                else:
                    if t_max_y < t_max_z:
                        array_len[last_id] = t_max_y
                        last_id+=1
                        Y += step_Y
                        array_x[last_id] = X
                        array_y[last_id] = Y                
                        t_max_y += t_delta_y
                    else:
                        array_len[last_id] = t_max_z
                        last_id+=1
                        finished = True
                if last_id>=array_len.size-1:
                    print("resize arrays")
                    old_size = len(array_len)
                    new_size = (old_size//BUFFER_SIZE+1)*BUFFER_SIZE
                    new_array_x = numpy.empty(new_size, dtype=numpy.int32)
                    new_array_x[:] = -1
                    new_array_y = numpy.empty(new_size, dtype=numpy.int32)
                    new_array_y[:] = -1
                    new_array_len = numpy.empty(new_size, dtype=numpy.float32)
                    new_array_x[:old_size] = array_x
                    new_array_y[:old_size] = array_y
                    new_array_len[:old_size] = array_len
                    array_x = new_array_x
                    array_y = new_array_y
                    array_len = new_array_len
            return array_x[:last_id], array_y[:last_id], array_len[:last_id]
    
        def one_pixel(self, row, col, sample):
            """calculate the contribution of one pixel to the sparse matrix and populate it.
    
            :param row: row index of the pixel of interest
            :param col: column index of the pixel of interest
            :param sample: Oversampling rate, 10 will thow 10x10 ray per pixel
    
            :return: the extra number of pixel allocated
            """
            if self.mask[row, col]:
                return (numpy.empty(0, dtype=numpy.int32),
                        numpy.empty(0, dtype=numpy.float32))
    
            counter = 0
            tmp_size = 0
            last_buffer_size = BUFFER_SIZE
            tmp_idx = numpy.empty(last_buffer_size, dtype=numpy.int32)
            tmp_idx[:] = -1
            tmp_coef = numpy.zeros(last_buffer_size, dtype=numpy.float32)
    
            pos = row * self.width + col
            start = self.idptr[pos]
            for i in range(sample):
                posx = (col+1.0*i/sample)*vox
                for j in range(sample):
                    posy = (row+1.0*j/sample)*voy
                    array_x, array_y, array_len = self.calc_one_ray(posx, posy)
    
                    rem = 1.0
                    for i in range(array_x.size):
                        x = array_x[i]
                        y = array_y[i]
                        l = array_len[i]
                        if (x<0) or (y<0) or (y>=self.height) or (x>=self.width):
                            break
                        elif (self.mask[y, x]):
                            continue
                        idx = x + y*self.width
                        dos = numpy.exp(-self.mu*l)
                        value = rem - dos
                        rem = dos
                        for j in range(last_buffer_size):
                            if tmp_size >= last_buffer_size:
                                #Increase buffer size
                                new_buffer_size = last_buffer_size + BUFFER_SIZE
                                new_idx = numpy.empty(new_buffer_size, dtype=numpy.int32)
                                new_coef = numpy.zeros(new_buffer_size, dtype=numpy.float32)
                                new_idx[:last_buffer_size] = tmp_idx
                                new_idx[last_buffer_size:] = -1
                                new_coef[:last_buffer_size] = tmp_coef
                                last_buffer_size = new_buffer_size
                                tmp_idx = new_idx
                                tmp_coef = new_coef
    
                            if tmp_idx[j] == idx:
                                tmp_coef[j] += value
                                break
                            elif tmp_idx[j] < 0:
                                tmp_idx[j] = idx
                                tmp_coef[j] = value
                                tmp_size +=1
                                break     
            return tmp_idx[:tmp_size], tmp_coef[:tmp_size]
    
        def calc_csr(self, sample):
            """Calculate the CSR matrix for the whole image
            :param sample: Oversampling factor
            :return: CSR matrix
            """
            size = self.width * self.height
            allocated_size = BLOCK_SIZE
            idptr = numpy.zeros(size+1, dtype=numpy.int32) 
            indices = numpy.zeros(allocated_size, dtype=numpy.int32)
            data = numpy.zeros(allocated_size, dtype=numpy.float32)
            self.sampled = sample*sample
            pos = 0
            start = 0
            for row in range(self.height):
                for col in range(self.width):    
                    line_idx, line_coef = self.one_pixel(row, col, sample)
                    line_size = line_idx.size
                    if line_size == 0:
                        new_size = 0
                        pos+=1
                        idptr[pos] = start
                        continue
    
                    stop = start + line_size
                    
                    if stop >= allocated_size:
                        new_buffer_size = allocated_size +  BLOCK_SIZE
                        new_idx = numpy.zeros(new_buffer_size, dtype=numpy.int32)
                        new_coef = numpy.zeros(new_buffer_size, dtype=numpy.float32)
                        new_idx[:allocated_size] = indices
                        new_coef[:allocated_size] = data
                        allocated_size = new_buffer_size
                        indices = new_idx
                        data = new_coef
    
                    indices[start:stop] = line_idx
                    data[start:stop] = line_coef
                    pos+=1
                    idptr[pos] = stop
                    start = stop
        
            last = idptr[-1]
            self.data = data
            self.indices = indices
            self.idptr = idptr
            return (self.data[:last]/self.sampled, indices[:last], idptr)


.. code:: ipython3

    thick = ThickDetector(vox,voy, thickness=thickness, mu=mu, dist=dist, poni1=poni1, poni2=poni2, mask=mask)
    %time thick.calc_csr(1)


.. parsed-literal::

    CPU times: user 2.17 s, sys: 4 ms, total: 2.18 s
    Wall time: 2.17 s




.. parsed-literal::

    (array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
     array([      2,       2,       4, ..., 1023180, 1023181, 1023182], dtype=int32),
     array([      0,       0,       0, ..., 1902581, 1902582, 1902583], dtype=int32))



.. code:: ipython3

    thick = ThickDetector(vox,voy, thickness=thickness, mu=mu, dist=dist, poni1=poni1, poni2=poni2, mask=mask)
    %time pre_csr = thick.calc_csr(8)


.. parsed-literal::

    CPU times: user 22.7 s, sys: 104 ms, total: 22.8 s
    Wall time: 22.7 s


Validation of the CSR matrix obtained:
--------------------------------------

For this we will build a simple 2D image with one pixel in a regular
grid and calculate the effect of the transformation calculated
previously on it.

.. code:: ipython3

    dummy_image = numpy.zeros(mask.shape, dtype="float32")
    dummy_image[::5,::5] = 1
    #dummy_image[mask] = -1
    csr = csr_matrix(pre_csr)
    dummy_blurred = csr.T.dot(dummy_image.ravel()).reshape(mask.shape)
    fix, ax = subplots(2,2, figsize=(8,8))
    ax[0,0].imshow(dummy_image)
    ax[0,1].imshow(csr.dot(dummy_image.ravel()).reshape(mask.shape))
    ax[1,1].imshow(csr.T.dot(dummy_image.ravel()).reshape(mask.shape))




.. parsed-literal::

    <IPython.core.display.Javascript object>



.. raw:: html

    <img src="data:image/png;base64,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" width="800">




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fb4a3510390>



.. code:: ipython3

    ax[0,0].set_xlim(964,981)
    ax[0,0].set_ylim(0,16)
    ax[0,1].set_xlim(964,981)
    ax[0,1].set_ylim(0,16)
    ax[1,1].set_xlim(964,981)
    ax[1,1].set_ylim(0,16)




.. parsed-literal::

    (0, 16)



.. code:: ipython3

    blured = csr.T.dot(dummy_image.ravel())
    
    # Invert this matrix: see https://arxiv.org/abs/1006.0758
    
    %time res = linalg.lsmr(csr.T, blured)
    
    restored = res[0].reshape(mask.shape)
    ax[1,0].imshow(restored)
    ax[1,0].set_xlim(964,981)
    ax[1,0].set_ylim(0,16)
    
    print(res[1:])


.. parsed-literal::

    CPU times: user 668 ms, sys: 52 ms, total: 720 ms
    Wall time: 724 ms
    (1, 31, 0.0003684464153090098, 4.0084031089374142e-05, 2.162035602251742, 4.8737283998183099, 195.49168107004448)


Conclusion of the raytracing part:
----------------------------------

We are able to simulate the path and the absorption of the photon in the
thickness of the detector. Numba helped substentially to make the
raytracing calculation much faster. The signal of each pixel is indeed
spread on the neighboors, depending on the position of the PONI and this
effect can be inverted using sparse-matrix pseudo-inversion.

We will now save this sparse matrix to file in order to be able to
re-use it in next notebook. But before saving it, it makes sense to
spend some time in generating a high quality sparse matrix in throwing
thousand rays per pixel in a grid of 32x32.

.. code:: ipython3

    %time pre_csr = thick.calc_csr(32)
    hq_csr = csr_matrix(pre_csr)
    from scipy.sparse import save_npz
    save_npz("csr.npz",hq_csr)


.. parsed-literal::

    CPU times: user 5min 30s, sys: 1.38 s, total: 5min 32s
    Wall time: 5min 30s

