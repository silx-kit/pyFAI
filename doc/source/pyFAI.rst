Aim of PyFAI
------------
:math:`2D` area detectors like ccd or pixel detectors have become
popular in the last 15 years for diffraction experiments (e.g. for waxs,
saxs, single crystal and powder diffraction (xrpd)). These detectors
have a large sensitive area of millions of pixels with high spatial
resolution. The software package pyFAI has been designed to reduce saxs,
waxs and xrpd images taken with those detectors into :math:`1D` curves
(azimuthal integration) usable by other software for in-depth analysis
such as Rietveld refinement, or :math:`2D` images (a radial
transformation named *caking*). As a library, the aim of pyFAI is to be
integrated into other tools like PyMca or edna or LImA with a clean pythonic
interface. However pyFAI features also command line and graphical tools for batch
processing, converting data into *q-space* (q being the momentum
transfer) or 2\ :math:`\theta`-space (:math:`\theta` being the Bragg
angle) and a calibration graphical interface for optimizing the geometry
of the experiment using the Debye-Scherrer rings of a reference sample.
PyFAI shares the geometry definition of spd but can directly import
geometries determined by the software fit2d. PyFAI has been designed to
work with any kind of detector and geometry (transmission or reflection)
and relies on FabIO, a library able to read more than 20 image formats
produced by detectors from 12 different manufacturers. During the
transformation from cartesian space :math:`(x,y)` to polar space
:math:`(2\theta, \chi )`, both local and total intensities are conserved
in order to obtain accurate quantitative results. Technical details on
how this integration is implemented and how it has been ported to native
code and parallelized on graphic cards are discussed in this paper.

Introduction
------------

With the advent of hyperspectral experiments like diffraction tomography
in the world of synchrotron radiation, existing software tools for
azimuthal integration like fit2d\  and spd\  reached their performance
limits owing to the fast data rate needed by such experiments. Even when
integrated into massively parallel frameworks like edna\ , such
stand-alone programs, due to their monolithic nature, cannot keep the
pace with the data flow of new detectors. Therefore we decided to
implemente from scratch a novel azimuthal integration tool which is
designed to take advantage of modern parallel hardware features.

Python Fast Azimuthal Integration
---------------------------------

PyFAI is implemented in Python programming language, which is open
source and already very popular for scientific data analysis (PyMca,
PyNX, …).

Geometry and calibration
........................

PyFAI and spd\  share the same 6-parameter geometry definition:
distance, point of normal incidence (2 coordinates) and 3 rotations
around the main axis; these parameters are saved in text files usually
with the *.poni* extension. The program *pyFAI-calib* helps calibrating
the experimental setup using a constrained least squares optimization on
the Debye-Scherrer rings of a reference sample (:math:`LaB_6`, silver
behenate, …). Alternatively, geometries calibrated using fit2d\  can be
imported into pyFAI, including geometric distortions (i.e. optical-fiber
tapers distortion) described as *spline-files*.

PyFAI executables
.................

PyFAI was designed to be used by scientists needing a simple and
effective tool for azimuthal integration. Two command line programs
*pyFAI-waxs* and *pyFAI-saxs* are provided with pyFAI for performing the
integration of one or more images. The waxs version outputs result in
:math:`2\theta /I`, whereas the saxs version outputs result in
:math:`q/I(/\sigma)`. Options for these programs are parameter file (*poni-file*)
describing the geometry and the mask file. They can also do some
pre-processing like dark-noise subtraction and flat-field correction
(solid-angle correction is done by default).

A new Grqaphical interface based on Qt is under development

Python library
..............

PyFAI is first and foremost a library: a tool of the scientific toolbox
built around IPython and NumPy to perform data analysis either
interactively or via scripts. Figure [notebook] shows an interactive
session where an integrator is created, and an image loaded and
integrated before being plotted.

.. figure:: img/notebook.png
   :align: center
   :alt: image

Regrouping mechanism
--------------------

In pyFAI, regrouping is performed using a histogram-like algorithm. Each
pixel of the image is associated to its polar coordinates
:math:`(2\theta , \chi )` or :math:`(q, \chi )`, then a pair of
histograms versus :math:`2\theta` (or :math:`q`) are built, one non
weighted for measuring the number of pixels falling in each bin and
another weighted by pixel intensities (after dark-current subtraction,
and corrections for flat-field, solid-angle and polarization). The
division of the weighted histogram by the number of pixels per bin gives
the diffraction pattern. :math:`2D` regrouping (called *caking* in
fit2d) is obtained in the same way using two-dimensional histograms over
radial (:math:`2\theta` or :math:`q`) and azimuthal angles
(:math:`\chi`).

Pixel splitting algorithm
.........................

Powder diffraction patterns obtained by histogramming have a major
weakness where pixel statistics are low. A manifestation of this
weakness becomes apparent in the :math:`2D`-regrouping where most of the
bins close to the beam-stop are not populated by any pixel. In this figure,
many pixels are missing in the low :math:`2\theta` region, due
to the arbitrary discretization of the space in pixels as intensities
were assigned to each pixel center which does not reflect the physical
reality of the scattering experiment.

.. figure:: img/2Dhistogram.png
   :align: center
   :alt: image

PyFAI solves this problem by pixel
splitting : in addition to the pixel position, its
spatial extension is calculated and each pixel is then split and
distributed over the corresponding bins, the intensity being considered
as homogeneous within a pixel and spread accordingly.

.. figure:: img/2DwithSplit.png
   :align: center
   :alt: image

Performances and migration to native code
.........................................

Originally, regrouping was implemented using the histogram provided by
NumPy, then re-implemented in Cython with pixel splitting to achieve a
four-fold speed-up. The computation time scales like O(N) with the size
of the input image. The number of output bins shows only little
influence; overall the single threaded Cython implementation has been
stated at 30 Mpix/s (on a 3.4 GHz Intel core i7-2600).


Parallel implementation
.......................

The method based on histograms works well on a single processor but runs
into problems requiring so called "atomic operations" when run in parallel.
Processing pixels in the input data order causes write access conflicts which
become less efficient with the increase of number of computing units.
This is the main limit of the method exposed previously;
especially on GPU where hundreds of threads are executed simultaneously.

To overcome this limitation; instead of looking at where input pixels GO TO
in the output image, we instead look at where the output pixels COME FROM
in the input image.
The correspondence between pixels and output bins can be stored in a
look-up table (LUT) together with the pixel weight which make the integration
look like a simple (if large and sparse) matrix vector product.
This look-up table size depends on whether pixels are split over multiple
bins and to exploit the sparse structure, both index and weight of the pixel
have to be stored.
We measured that 500 Mb are needed to store the LUT to integrate a 16 megapixel image,
which fits onto a reasonable quality graphics card nowadays.
By making this change we switched from a “linear read / random write” forward algorithm
to a “random read / linear write” backward algorithm which is more suitable for parallelization.
This algorithm was implemented in Cython-OpenMP and OpenCL.
When using OpenCL for the GPU we used a compensated, or Kahan summation to reduce
the error accumulation in the histogram summation (at the cost of more operations to be done).
This allows accurate results to be obtained on cheap hardware that performs calculations
in single precision floating-point arithmetic (32 bits) which are available on consumer
grade graphic cards.
Double precision operations are currently limited to high price and performance computing dedicated GPUs.
The additional cost of Kahan summation, 4x more arithmetic operations, is hidden by smaller data types,
the higher number of single precision units and that the GPU is usually limited by the memory bandwidth anyway.

The perfomances of the parallel implementation based on a LUT are above 125 MPix/s (on a 3.4 GHz Intel core i7-2600)
and can reach 200 MPix/s on recent multi-socket, multi-core computer or on high-end GPUs like Tesla cards.

.. figure:: img/benchmark.png
   :align: center
   :alt: benchmark performed on a 2010 consumer computer


Conclusion
----------

The library pyFAI was developed with two main goals:

-  Performing azimuthal integration with a clean programming interface.

-  No compromise on the quality of the results is accepted: a careful
   management of the geometry and precise pixel splitting ensures total
   and local intensity preservation.

PyFAI is the first implementation of an azimuthal integration algorithm
on a gpu as far as we are aware of, and the stated twenty-fold speed up
opens the door to a new kind of analysis, not even considered before.
With a good interface close to the camera, we believe PyFAI is able to sustain the data
streams from the next generation high-speed detectors.

Acknowledgments
...............

Porting pyFAI to GPU would have not been possible without
the financial support of LinkSCEEM-2 (RI-261600).

References:
.........--

- The philosophy of pyFAI is described in the proceedings of SRI2012:
  doi:10.1088/1742-6596/425/20/202012
  http://iopscience.iop.org/1742-6596/425/20/202012/

- The LUT implementation (ported to GPU) is described in the proceedings
  of EPDIC13:  http://epdic13.grenoble.cnrs.fr/spip.php?article43
  (to be published)
  
- [FIT2D] Hammersley A. P., Svensson S. O., Hanfland M., Fitch A. N. and Hausermann D. 
  1996 High Press. Res. vol 14 p 235–248

- [SPD] Bösecke P. 2007 J. Appl. Cryst. vol 40 s 423–s427

- [EDNA] Incardona M. F., Bourenkov G. P., Levik K., Pieritz R. A., Popov A. N. and Svensson O. 
  2009 J. Synchrotron Rad. vol 16 p 872–879

- [PyMca] Solé V. A., Papillon E., Cotte M., Walter P. and Susini J. 
  2007 Spectrochim. Acta Part B vol vol 62 p 63 – 68

- [PyNX] Favre-Nicolin V., Coraux J., Richard M. I. and Renevier H. 
  2011 J. Appl. Cryst. vol 44 p 635–640

- [iPython] Pérez F and Granger B E 
  2007 Comput. Sci. Eng. vol 9 p 21–29 URL http://ipython.org
  
- [NumPy] Oliphant T E 2007 Comput. Sci. Eng. 9 10–20

- [Cython] Behnel S, Bradshaw R, Citro C, Dalcin L, Seljebotn D and Smith K 2011 Comput. Sci. Eng. 13 31 –39

- [OpenCL] Khronos OpenCL Working Group 2010 The OpenCL Specification, version 1.1 URL http://www.khronos.org/registry/cl/specs/opencl-1.1.pdf

- [FabIO] Sorensen H O, Knudsen E, Wright J, Kieffer J et al. 
  2007–2013 FabIO: I/O library for images produced by 2D X-ray detectors URL http://fable.sf.net/
  
- [Matplotlib] Hunter J D 2007 Comput. Sci. Eng. 9 90–95 ISSN 1521-9615

- [SciPy] Jones E, Oliphant T, Peterson P et al. 
  2001– SciPy: Open source scientific tools for Python URL
  http://www.scipy.org/
  
- [FFTw] Frigo M and Johnson S G 
  2005 Proceedings of the IEEE 93 216–231
  