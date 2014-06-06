Calibration tool: pyFAI-recalib
===============================

Purpose
-------

Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
with a priori knowledge of your setup (an input PONI-file).
If you are using a standard calibrant, look at the list provided by pyFAI at:
https://github.com/kif/pyFAI/tree/master/calibration.
Else, you will need a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order), they can be found on the American Mineralogist 
database [AMD]_ or in the [COD]_.

You will need in addition:
 * The radiation energy (in keV) or its wavelength (in A)

Many option are available among those:
 * dark-current / flat field corrections
 * Masking of bad regions
 * Polarization correction
 * Automatic desaturation (time consuming!)
 * Intensity weighted least-squares refinements

The output of this program is a "PONI" file containing the detector description
and the 6 refined parameters (distance, center, rotation) and wavelength.
An 1D and 2D diffraction patterns are also produced. (.dat and .azim files)

The main difference with pyFAI-calib is the way control-point hence Debye-Sherrer
rings are extracted. While pyFAI-calib relies on the contiguity of a region of peaks
called massif; pyFAI-recalib knows approximatly the geometry and is able to select
the region where the ring should be. From this region it selects automatically
the various peaks; making pyFAI-recalib able to run without graphical interface and
without human intervention (--no-gui --no-interactive options).

Usage:
------

pyFAI-recalib [options] -w 1 -p imagefile.poni -S calibrant.D imagefile.edf

Options:
--------
  -h, --help            show  help message and exit
  -V, --version         print version of the program and quit
  -o FILE, --out=FILE   Filename where processed image is saved
  -v, --verbose         switch to debug/verbose mode
  -S FILE, --spacing=FILE
                        file containing d-spacing of the reference sample
                        (MANDATORY)
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
  -r MAX_RINGS, --ring=MAX_RINGS
                        maximum number of rings to extract. Default: all
                        accessible
  -p FILE, --poni=FILE  file containing the diffraction parameter (poni-file).
                        MANDATORY
  -k, --keep            Keep existing control point and append new

Tips & Tricks
-------------

PONI files are ASCII files and each new refinement adds an entry int the file.
So if you are unhappy with the last step, just edit this file and remove the last
entry (timestamps will help you).

