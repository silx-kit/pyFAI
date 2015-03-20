Design of the Python Fast Azimuthal Integrator
==============================================

Author: Jérôme Kieffer

Date: 20/03/2015

Keywords: Design

Target: Developers interested in using the library

Reference: API documentation

Abstract
--------

The core part of pyFAI is the AzimuthalIntegator objects, named
*ai* hereafter.
This document describes the two importants methods of the class,
how it is related to Detector, Geometry, and integration engines.

One of the core idea is to have a complete representation of the geometry
and perform the azimuthal integration as a single geometrical re-binning
which take into account all effects like:

* Detector distortion
* Polar transformation
* assignment to the output space


This document focuses on the core of pyFAI while peripheral code
dealing with graphical user interfaces, image analysis online data
analysis integration are not covered.

AzimuthalIntegrator
-------------------

This class is the core of pyFAI, and it is the only one likely to be used by
external developers/users. It is usually instantiated via a function of the
module to load a poni-file:

.. code-block:: python
	>>> import pyFAI
	>>> ai = pyFAI.load("Pilatus1M.poni")
	>>> print(ai)

	Detector Detector	 Spline= None	 PixelSize= 1.720e-04, 1.720e-04 m

	SampleDetDist= 1.583231e+00m	PONI= 3.341702e-02, 4.122778e-02m	rot1=0.006487  rot2= 0.007558  rot3= 0.000000 rad

	DirectBeamDist= 1583.310mm	Center: x=179.981, y=263.859 pix	Tilt=0.571 deg  tiltPlanRotation= 130.640 deg

As one can see, the *ai* contains the detector geometry (type, pixel size,
distortion) as well as the
geometry of the experimental setup. The geometry is given in two equivalent
forms: the internal representation of pyFAI (second line) and the one used by
FIT2D.

The *ai* is responsible for azimuthal integration, either the integration along
complete ring, called full-integration, obtained via *ai.integrate1d* method.
The sector-wise integration is obtained via the *ai.integrate2d* method.
The options for those two methods are really similar and differ only by the
parameters related to the azimuthal dimension of the averaging for *ai.integrate2d*.

Azimuthal integration methods
_____________________________

Both integration method take as first argument the image coming from the detector
as a numpy array. This is the only mandatory parameter.

Important parameters are the number of bins in radial and azimuthal dimensions.
Other parameters are the pre-processing information like dark and flat pixel wise
correction (as array), the polarization factor and the solid-angle correction to
be applied.

Because multiple radial output space are possible (q, r, 2\theta) each with multiple
units, if one wants to avoid interpolation, it is important to export directly the data
in the destination space, specifying the unit="2th_deg" or "q_nm^-1"

Many more option exists, please refer to the documentation of AzimuthalIntegration integrate_

The AzimuthalIntegration class inherits from the Geometry class and hold
references to configured rebinning engines.

Geometry
--------
The Geometry class contains a reference to the detector (composition)
and the logic to calculate the position in space of the various pixels.
All arrays in the class are cached and calculated on demand.

The Geometry class relies on the detector to provide the pixel position in space
and subsequently transforms it in 2\theta coordinates, or q, \chi, r ...
This can either be performed in the class itself or by calling
function in the parallel implemented Cython module _geometry.
Those transformation could be GPU-ized in the future.

Detector
--------
PyFAI deals only with area detector, indexed in 2 dimension but can
handle pixel located in a 3D space.

The *pyFAI.detectors* module contains the master *Detector* class
which is capable of describing any detector.
About 40 types of detectors, inheriting and specializing the *Detector*
class are provided, offering convienient access to most commercial detectors.
A factory is provided to easily instantiate a detector from its name.

A detector class is responsible for two main tasks:

- provide the coordinate in space of any pixel position (center, corner, ...)
- Handle the mask: some detector feature automatic mask calculation (i.e. module based detectors).

The disortion of the detector is handled here and could be GPU-ized in the future.

Rebinning engines
-----------------

Once the geometry (radial and azimuthal coordinates) calculated for every pixel
on the detector, the image from the detector is rebinned into the output space.
Two types of rebinning engines exists:

Histograms
	They take each single pixel from the image and transfer it to the destination bin, like histograms do.
	This family of algorithms is rather easy to implement and provides good single threaded performances,
	but it is hard to parallelize (efficiently) due to the need of atomic operations.

Sparse matrix multiplication
    By recording where every single ends one can transform the previous histogram into a
    large sparse matrix multiplication which is either stored as a Look-Up Table (actually an array of struct, also called LIL)
    or more efficiently in the CSR_ format.
    Those rebinning engines are trivially parallel and provide the best performances.

Pixel splitting
---------------

Three levels of pixel splitting schemes are available within pyFAI:

No splitting
	The whole intensity is assigned to the center of the pixel and rebinned using a simple histogram

Bounding box pixel splitting
	The pixel is abstracted by a box surrounding it with, making calculation easier but blurring a bit the image

Tight pixel splitting
	The pixel is represented by its actual corner position, offering a very precise positionning in space.

The main issue with pixel splitting arose from 2D integration and the habdling of pixel laying on the chi-discontinuity.

References:
-----------

:: _integrate: http://pythonhosted.org/pyFAI/api/pyFAI.html#pyFAI.azimuthalIntegrator.AzimuthalIntegrator.integrate1d

:: _CSR: http://en.wikipedia.org/wiki/Sparse_matrix