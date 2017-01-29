:Author: Jérôme Kieffer
:Date: 04/01/2017
:Keywords: Definition of technical terms
:Target: General audience

========
Glossary
========

You will find some explanation about technical terms often used in pyFAI and its documentation.

AzimuthalIntegrator
===================

The azimuthal integrator (*ai* hereafter)  is an object, in the sense of Object
Oriented Programming which can transform an image into a powder diffraction pattern
using the integrate1d or integrate2d methods.
It contains (inherits) a Geometry and caches the look-up table for optimized integration.
Ai objects are usually obtained from loading a *PONI-file* obtained after calibration.
See :ref:`AzimuthalIntegrator`

Calibration
===========
Action of determining the geometry of an experiment geometry using a reference sample.
Calibration is performed using the pyFAI-calib tool to generate the *PONI-file*.,
See its :ref:`calibration`


Detector
========

Detectors are all area detectors in pyFAI, which mean pixels are indexed by 2 indices
often references as (slow, fast) or (row, column) or simply (1, 2).
Indices are using the C-convention.
The detector object is responsible for calculating the pixel position in space (so in 3D).
and assumes each pixel is defined by 4-corners.
Detectors can be defined by many ways: from a shape and a pixel size, from a
tabulated detector (there are over 60 detector defined) or from a NeXus file.
See :ref:`detector`

Distortion
==========
Distortion correction is the action of correcting the spatial distortion from an
image due to a detector. The object is defined from detector and the target is
an image with a regular grid and a given pixel size and shape.

Geometry
========
The geometry object is in charge of calculating all kind of coordinates (cartesian, polar, ...)
for any point in the image, a step both needed for integration and calibration.
The geometry object composes (contains) a detector and contains the 3 translations (dist, poni1 and poni2)
and 3 rotations (rot1, rot2, rot3) of the detector in space. It knows in addition about the wavelength.
The object is saved in a PONI-file, see :ref:`Geometry`

Image
=====
see :ref:`Image`

PONI
====
Acronym for Point Of Normal Incidence.
It is the pixel coordinates of the orthogonal projection of the sample position
on the detector (for plane detectors, or the plane z=0 for other).
The *PONI* coincide with the beam center in the case of orthogonal setup but most
of the time differs.

PONI-File
=========
Small text file generated from pyFAi-calib or when saving a geometry object which
contains the detector description, its position in space and the wavelength.
Used to initialize AzimuthalIntegrator or Geometry objects.

Worker
======
A worker allow, once parameterized, to perform always the same processing on all
images in a stack, for example azimuthal integration, distortion correction or
simple pixel-wise operation.

