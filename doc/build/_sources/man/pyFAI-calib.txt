Calibration tool: pyFAI-calib
=============================

Purpose
-------

Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
without a priori knowledge of your setup.
You will need to provide a calibrant or a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order).

If you are using a standard calibrant, look at
https://github.com/kif/pyFAI/tree/master/calibration
or search in the American Mineralogist database:
[AMD]_ or in the [COD]_.
The --calibrant option is mandatory !

Calibrants available: Ni, CrOx, NaCl, Si_SRM640e,
Si_SRM640d, Si_SRM640a, Si_SRM640b, Cr2O3, AgBh, Si_SRM640, CuO, PBBA,
alpha_Al2O3, SI_SRM640c, quartz, C14H30O, cristobaltite, Si, LaB6, CeO2,
LaB6_SRM660a, LaB6_SRM660b, LaB6_SRM660c, TiO2, ZnO, Al, Au

You will need in addition:
 * The radiation energy (in keV) or its wavelength (in A)
 * The description of the detector:
  - it name or
  - it's pixel size or
  - the spline file describing its distortion or
  - the NeXus file describing the distortion

Many option are available among those:
 * dark-current / flat field corrections
 * Masking of bad regions
 * reconstruction of missing region (module based detectors), see option -r
 * Polarization correction
 * Automatic desaturation (time consuming!)
 * Intensity weighted least-squares refinements

The output of this program is a "PONI" file containing the detector
description and the 6 refined parameters (distance, center, rotation) and
wavelength. An 1D and 2D diffraction patterns are also produced. (.dat and
.azim files)


Usage:
------
pyFAI-calib [options] -w 1 -D detector -c calibrant imagefile.edf


Options:
--------

  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -o FILE, --out FILE   Filename where processed image is saved
  -v, --verbose         switch to debug/verbose mode
  -c FILE, --calibrant FILE
                        Calibrant name or file containing d-spacing of the
                        reference sample (MANDATORY, case sensitive !)
  -w WAVELENGTH, --wavelength WAVELENGTH
                        wavelength of the X-Ray beam in Angstrom. Mandatory
  -e ENERGY, --energy ENERGY
                        energy of the X-Ray beam in keV
                        (hc=12.398419292keV.A).
  -P POLARIZATION_FACTOR, --polarization POLARIZATION_FACTOR
                        polarization factor, from -1 (vertical) to +1
                        (horizontal), default is None (no correction),
                        synchrotrons are around 0.95
  -i FILE, --poni FILE  file containing the diffraction parameter (poni-file).
                        MANDATORY for pyFAI-recalib!
  -b BACKGROUND, --background BACKGROUND
                        Automatic background subtraction if no value are
                        provided
  -d DARK, --dark DARK  list of comma separated dark images to average and
                        subtract
  -f FLAT, --flat FLAT  list of comma separated flat images to average and
                        divide
  -s SPLINE, --spline SPLINE
                        spline file describing the detector distortion
  -D DETECTOR_NAME, --detector DETECTOR_NAME
                        Detector name (instead of pixel size+spline)
  -m MASK, --mask MASK  file containing the mask (for image reconstruction)
  -n NPT, --pt NPT      file with datapoints saved. Default: basename.npt
  --filter FILTER       select the filter, either mean(default), max or median
  -l DISTANCE, --distance DISTANCE
                        sample-detector distance in millimeter. Default: 100mm
  --dist DIST           sample-detector distance in meter. Default: 0.1m
  --poni1 PONI1         poni1 coordinate in meter. Default: center of detector
  --poni2 PONI2         poni2 coordinate in meter. Default: center of detector
  --rot1 ROT1           rot1 in radians. default: 0
  --rot2 ROT2           rot2 in radians. default: 0
  --rot3 ROT3           rot3 in radians. default: 0
  --fix-dist            fix the distance parameter
  --free-dist           free the distance parameter. Default: Activated
  --fix-poni1           fix the poni1 parameter
  --free-poni1          free the poni1 parameter. Default: Activated
  --fix-poni2           fix the poni2 parameter
  --free-poni2          free the poni2 parameter. Default: Activated
  --fix-rot1            fix the rot1 parameter
  --free-rot1           free the rot1 parameter. Default: Activated
  --fix-rot2            fix the rot2 parameter
  --free-rot2           free the rot2 parameter. Default: Activated
  --fix-rot3            fix the rot3 parameter
  --free-rot3           free the rot3 parameter. Default: Activated
  --fix-wavelength      fix the wavelength parameter. Default: Activated
  --free-wavelength     free the wavelength parameter. Default: Deactivated
  --tilt                Allow initially detector tilt to be refined (rot1,
                        rot2, rot3). Default: Activated
  --no-tilt             Deactivated tilt refinement and set all rotation to 0
  --saturation SATURATION
                        consider all pixel>max*(1-saturation) as saturated and
                        reconstruct them, default: 0 (deactivated)
  --weighted            weight fit by intensity, by default not.
  --npt NPT_1D          Number of point in 1D integrated pattern, Default:
                        1024
  --npt-azim NPT_2D_AZIM
                        Number of azimuthal sectors in 2D integrated images.
                        Default: 360
  --npt-rad NPT_2D_RAD  Number of radial bins in 2D integrated images.
                        Default: 400
  --unit UNIT           Valid units for radial range: 2th_deg, 2th_rad,
                        q_nm^-1, q_A^-1, r_mm. Default: 2th_deg
  --no-gui              force the program to run without a Graphical interface
  --no-interactive      force the program to run and exit without prompting
                        for refinements
  -r, --reconstruct     Reconstruct image where data are masked or <0 (for
                        Pilatus detectors or detectors with modules)
  -g GAUSSIAN, --gaussian GAUSSIAN
                        Size of the gaussian kernel. Size of the gap (in
                        pixels) between two consecutive rings, by default 100
                        Increase the value if the arc is not complete;
                        decrease the value if arcs are mixed together.
  --square              Use square kernel shape for neighbor search instead of
                        diamond shape
  -p PIXEL, --pixel PIXEL
                        size of the pixel in micron


Tips & Tricks
-------------

PONI-files are ASCII files and each new refinement adds an entry in the file.
So if you are unhappy with the last step, just edit this file and remove the last
entry (time-stamps will help you).



Example of usage:
-----------------


.. command-output:: pyFAI-calib --help
    :nostderr:

Pilatus 1M image of Silver Behenate taken at ESRF-BM26:
.......................................................

::

	pyFAI-calib -D Pilatus1M -c AgBh -r -w 1.0 test/testimages/Pilatus1M.edf

We use the parameter -r to reconstruct the missing part between the modules of the
Pilatus detector.


Half a FReLoN CCD image of Lantanide hexaboride taken at ESRF-ID11:
...................................................................

::

	pyFAI-calib -s test/testimages/halfccd.spline -c LaB6 -w 0.3 test/testimages/halfccd.edf -g 250


This image is rather spotty. We need to blur a lot to get the continuity of the rings.
This is achieved by the -g parameter.
While the sample is well diffracting and well known, the wavelength has been guessed.
One should refine the wavelength when the peaks extracted are correct


All those images are part of the test-suite of pyFAI. To download them from internet, run

::

	python setup.py build test

Downloaded test images  are located in tests/testimages
