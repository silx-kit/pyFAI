Calibration tool: pyFAI-calib
=============================

Purpose
-------

Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
without a priori knowledge of your setup.
Most standard calibrants are directly installed together with pyFAI. 
If you prefer using your own, you can provide a "d-spacing" file
containing the spacing of Miller plans in Angstrom (in decreasing order). 
Most crystal powders used for calibration are available in the American_Mineralogist_database_
.. _American_Mineralogist_database: http://rruff.geo.arizona.edu/AMS/amcsd.php


You will need in addition:
 * The radiation energy (in keV) or its wavelength (in A)
 * The description of the detector: it name or it's pixel size or the spline
 file describing its distortion

Many option are available among those:
 * dark-current / flat field corrections
 * Masking of bad regions
 * reconstruction of missing region (module based detectors)
 * Polarization correction
 * Automatic desaturation (time consuming!)
 * Intensity weighted least-squares refinements

The output of this program is a "PONI" file containing the detector
description and the 6 refined parameters (distance, center, rotation) and
wavelength. An 1D and 2D diffraction patterns are also produced. (.dat and
.azim files)


Usage:
------
pyFAI-calib [options] -w 1 -D detector -c calibrant.D imagefile.edf



Options:
--------

  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -o FILE, --out=FILE   Filename where processed image is saved
  -v, --verbose         switch to debug/verbose mode
  -c FILE, --calibrant=FILE
                        Calibrant name or file containing d-spacing of the
                        reference sample (MANDATORY)
  -w WAVELENGTH, --wavelength=WAVELENGTH
                        wavelength of the X-Ray beam in Angstrom
  -e ENERGY, --energy=ENERGY
                        energy of the X-Ray beam in keV (hc=12.398419292keV.A)
  -P POLARIZATION_FACTOR, --polarization=POLARIZATION_FACTOR
                        polarization factor, from -1 (vertical) to +1
                        (horizontal), default is None (no correction),
                        synchrotrons are around 0.95
  -b BACKGROUND, --background=BACKGROUND
                        Automatic background subtraction if no value are
                        provided
  -d DARK, --dark=DARK  list of dark images to average and subtract
  -f FLAT, --flat=FLAT  list of flat images to average and divide
  -s SPLINE, --spline=SPLINE
                        spline file describing the detector distortion
  -D DETECTOR_NAME, --detector=DETECTOR_NAME
                        Detector name (instead of pixel size+spline)
  -m MASK, --mask=MASK  file containing the mask (for image reconstruction)
  -n NPT, --pt=NPT      file with datapoints saved. Default: basename.npt
  --filter=FILTER       select the filter, either mean(default), max or median
  -l DISTANCE, --distance=DISTANCE
                        sample-detector distance in millimeter
  --poni1=PONI1         poni1 coordinate in meter
  --poni2=PONI2         poni2 coordinate in meter
  --rot1=ROT1           rot1 in radians
  --rot2=ROT2           rot2 in radians
  --rot3=ROT3           rot3 in radians
  --fix-dist            fix the distance parameter
  --free-dist           free the distance parameter
  --fix-poni1           fix the poni1 parameter
  --free-poni1          free the poni1 parameter
  --fix-poni2           fix the poni2 parameter
  --free-poni2          free the poni2 parameter
  --fix-rot1            fix the rot1 parameter
  --free-rot1           free the rot1 parameter
  --fix-rot2            fix the rot2 parameter
  --free-rot2           free the rot2 parameter
  --fix-rot3            fix the rot3 parameter
  --free-rot3           free the rot3 parameter
  --fix-wavelength      fix the wavelength parameter
  --free-wavelength     free the wavelength parameter
  --saturation=SATURATION
                        consider all pixel>max*(1-saturation) as saturated and
                        reconstruct them
  --weighted            weight fit by intensity, by default not.
  --npt=NPT_1D          Number of point in 1D integrated pattern, Default:
                        1024
  --npt-azim=NPT_2D_AZIM
                        Number of azimuthal sectors in 2D integrated images.
                        Default: 360
  --npt-rad=NPT_2D_RAD  Number of radial bins in 2D integrated images.
                        Default: 400
  --unit=UNIT           Valid units for radial range: 2th_deg, 2th_rad,
                        q_nm^-1, q_A^-1, r_mm. Default: 2th_deg
  --no-gui              force the program to run without a Graphical interface
  --no-interactive      force the program to run and exit without prompting
                        for refinements
  -r, --reconstruct     Reconstruct image where data are masked or <0  (for
                        Pilatus detectors or detectors with modules)
  -g GAUSSIAN, --gaussian=GAUSSIAN
                        Size of the gaussian kernel. Size of the gap (in
                        pixels) between two consecutive rings, by default 100
                        Increase the value if the arc is not complete;
                        decrease the value if arcs are mixed together.
  --square              Use square kernel shape for neighbor search instead of
                        diamond shape
  -p PIXEL, --pixel=PIXEL
                        size of the pixel in micron

Example of usage:
-----------------

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
