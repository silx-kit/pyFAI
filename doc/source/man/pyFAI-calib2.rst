Calibration tool: pyFAI-calib2
==============================

calibration - GUI tool for determining the geometry of a detector using
a reference sample.

Please have a look at the :ref:`cookbook_calibration_gui` for a 5 minutes introduction.

Purpose
-------

Calibrate the diffraction setup geometry based on Debye-Sherrer rings
images without a priori knowledge of your setup. You will need to
provide a calibrant or a "d-spacing" file containing the spacing of
Miller plans in Angstrom (in decreasing order).

Calibrants available:
``Al``, ``LaB6``, ``TiO2``, ``Pt`, ``Ni``, ``CuO``, ``quartz``, ``Si``, ``mock``, ``Si_SRM640e``, ``LaB6_SRM660a``,
``PBBA``, ``cristobaltite``, ``Si_SRM640``, ``NaCl``, ``AgBh``, ``CrOx``, ``LaB6_SRM660c``, ``C14H30O``,
``Si_SRM640a``, ``Au``, ``alpha_Al2O3``, ``ZnO``, ``Si_SRM640d``, ``Cr2O3``, ``Si_SRM640c``,
``LaB6_SRM660b``, ``Si_SRM640b``, ``hydrocerussite``, ``CeO2`` or search in the
`American Mineralogist database  <http://rruff.geo.arizona.edu/AMS/amcsd.php>`_

Usage:
------

``pyFAI-calib2 [options] [input_image.edf]``


positional arguments:
+++++++++++++++++++++

FILE
   List of files to calibrate

optional arguments:
+++++++++++++++++++

**-h**, **--help**
   show this help message and exit

**-V**, **--version**
   show program's version number and exit

**-o** FILE, **--out** FILE
   Filename where processed image is saved

**-v**, **--verbose**
   switch to debug/verbose mode

**--debug**
   Set logging system in debug mode

**--opengl**, **--gl**
   Enable OpenGL rendering (else matplotlib is used)

**-c** FILE, **--calibrant** FILE
   Calibrant name or file containing d-spacing of the reference sample
   (case sensitive)

**-w** WAVELENGTH, **--wavelength** WAVELENGTH
   wavelength of the X-Ray beam in Angstrom.

**-e** ENERGY, **--energy** ENERGY
   energy of the X-Ray beam in keV (hc=12.398419843320026keV.A).

**-P** POLARIZATION_FACTOR, **--polarization** POLARIZATION_FACTOR
   polarization factor, from **-1** (vertical) to +1 (horizontal),
   default is None (no correction), synchrotrons are around 0.95

**-D** DETECTOR_NAME, **--detector** DETECTOR_NAME
   Detector name (instead of pixel size+spline)

**-m** MASK, **--mask** MASK
   file containing the mask (for image reconstruction)

**-p** PIXEL, **--pixel** PIXEL
   size of the pixel in micron

**-s** SPLINE, **--spline** SPLINE
   spline file describing the detector distortion

**-n** NPT, **--pt** NPT
   file with datapoints saved. Example: basename.npt

**-i** FILE, **--poni** FILE
   file containing the diffraction parameter (poni-file) [not used].

**-b** BACKGROUND, **--background** BACKGROUND
   Automatic background subtraction if no value are provided [not used]

**-d** DARK, **--dark** DARK
   list of comma separated dark images to average and subtract [not
   used]

**-f** FLAT, **--flat** FLAT
   list of comma separated flat images to average and divide [not used]

**--filter** FILTER
   select the filter, either mean(default), max or median [not used]

**-l** DIST_MM, **--distance** DIST_MM
   sample-detector distance in millimeter. Default: 100mm

**--dist** DIST
   sample-detector distance in meter. Default: 0.1m

**--poni1** PONI1
   poni1 coordinate in meter. Default: center of detector

**--poni2** PONI2
   poni2 coordinate in meter. Default: center of detector

**--rot1** ROT1
   rot1 in radians. default: 0

**--rot2** ROT2
   rot2 in radians. default: 0

**--rot3** ROT3
   rot3 in radians. default: 0

**--dist-range** DIST_MIN DIST_MAX
   Range (2 values) for sample-detector distance in meter. Default: None

**--poni1-range** PONI1_MIN PONI1_MAX
   Range for poni1 coordinate in meter. Default: None

**--poni2-range** PONI2_MIN PONI2_MAX
   Range (2 values) for poni2 coordinate in meter. Default: None

**--rot1-range** ROT1_MIN ROT1_MAX
   Range (2 values) for rot1 in radians. default: None

**--rot2-range** ROT2_MIN ROT2_MAX
   Range (2 values) for rot2 in radians. default: None

**--rot3-range** ROT3_MIN ROT3_MAX
   Range (2 values) for rot3 in radians. default: None

**--wavelength-range** WAVELENGTH_MIN WAVELENGTH_MAX
   Range (2 values) for wavelength in meters. default: None

**--fix-wavelength**
   fix the wavelength parameter. Default: Activated

**--free-wavelength**
   free the wavelength parameter. Default: Deactivated

**--fix-dist**
   fix the distance parameter

**--free-dist**
   free the distance parameter. Default: Activated

**--fix-poni1**
   fix the poni1 parameter. Default: Deactivated

**--free-poni1**
   free the poni1 parameter. Default: Activated

**--fix-poni2**
   fix the poni2 parameter. Default: Deactivated

**--free-poni2**
   free the poni2 parameter. Default: Activated

**--fix-rot1**
   fix the rot1 parameter. Default: Deactivated

**--free-rot1**
   free the rot1 parameter. Default: Activated

**--fix-rot2**
   fix the rot2 parameter. Default: Deactivated

**--free-rot2**
   free the rot2 parameter. Default: Activated

**--fix-rot3**
   fix the rot3 parameter. Default: Deactivated

**--free-rot3**
   free the rot3 parameter. Default: Activated

**--npt** NPT_1D
   Number of point in 1D integrated pattern, Default: 1024

**--npt-azim** NPT_2D_AZIM
   Number of azimuthal sectors in 2D integrated images. Default: 360

**--npt-rad** NPT_2D_RAD
   Number of radial bins in 2D integrated images. Default: 400

**--qtargs** QTARGS
   Arguments propagated to Qt

**--tilt**
   Allow initially detector tilt to be refined (rot1, rot2, rot3).
   Default: Activated

**--no-tilt**
   Deactivated tilt refinement and set all rotation to 0

**--saturation** SATURATION
   consider all pixel>max*(1-saturation) as saturated and reconstruct
   them, default: 0 (deactivated)

**--weighted**
   weight fit by intensity, by default not.

**--unit** UNIT
   Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1, q_A^-1,
   r_mm. Default: 2th_deg

**--no-gui**
   force the program to run without a Graphical interface

**--no-interactive**
   force the program to run and exit without prompting for refinements

**-r**, **--reconstruct**
   Reconstruct image where data are masked or <0 (for Pilatus detectors
   or detectors with modules)

**-g** GAUSSIAN, **--gaussian** GAUSSIAN
   Size of the gaussian kernel. Size of the gap (in pixels) between two
   consecutive rings, by default 100 Increase the value if the arc is
   not complete; decrease the value if arcs are mixed together.

**--square**
   Use square kernel shape for neighbor search instead of diamond shape

Tips & Tricks
-------------

The output of this program is a "PONI" file containing the detector
description and the 6 refined parameters (distance, center, rotation)
and wavelength. An 1D and 2D diffraction patterns are also produced.
(.dat and .azim files)
