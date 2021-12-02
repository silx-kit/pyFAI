Calibration tool: pyFAI-recalib
===============================

pyFAI-recalib is now obsolete. All feature provided by it are now available as
part of pyFAI-calib.

calibration - DEPRECATED tool for refining the geometry of a detector
using a reference sample and a previously known calibration file.

DESCRIPTION
-----------

usage: pyFAI-recalib [options] **-i** ponifile **-w** 1 **-c**
calibrant.D imagefile.edf

Calibrate the diffraction setup geometry based on Debye-Sherrer rings
images with a priori knowledge of your setup (an input PONI-file). You
will need to provide a calibrant or a "d-spacing" file containing the
spacing of Miller plans in Angstrom (in decreasing order). Calibrants
available: Al, LaB6, TiO2, Pt, Ni, CuO, quartz, Si, mock, Si_SRM640e,
LaB6_SRM660a, PBBA, cristobaltite, Si_SRM640, NaCl, AgBh, CrOx,
LaB6_SRM660c, C14H30O, Si_SRM640a, Au, alpha_Al2O3, ZnO, Si_SRM640d,
Cr2O3, Si_SRM640c, LaB6_SRM660b, Si_SRM640b, hydrocerussite, CeO2 or
search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php The **--calibrant** option is
mandatory !

positional arguments:
---------------------

FILE
   List of files to calibrate

optional arguments:
-------------------

**-h**, **--help**
   show this help message and exit

**-V**, **--version**
   show program's version number and exit

**-o** FILE, **--out** FILE
   Filename where processed image is saved

**-v**, **--verbose**
   switch to debug/verbose mode

**-c** FILE, **--calibrant** FILE
   Calibrant name or file containing d-spacing of the reference sample
   (MANDATORY, case sensitive !)

**-w** WAVELENGTH, **--wavelength** WAVELENGTH
   wavelength of the X-Ray beam in Angstrom. Mandatory

**-e** ENERGY, **--energy** ENERGY
   energy of the X-Ray beam in keV (hc=12.398419843320026keV.A).

**-P** POLARIZATION_FACTOR, **--polarization** POLARIZATION_FACTOR
   polarization factor, from **-1** (vertical) to +1 (horizontal),
   default is None (no correction), synchrotrons are around 0.95

**-i** FILE, **--poni** FILE
   file containing the diffraction parameter (poni-file). MANDATORY for
   pyFAI-recalib!

**-b** BACKGROUND, **--background** BACKGROUND
   Automatic background subtraction if no value are provided

**-d** DARK, **--dark** DARK
   list of comma separated dark images to average and subtract

**-f** FLAT, **--flat** FLAT
   list of comma separated flat images to average and divide

**-s** SPLINE, **--spline** SPLINE
   spline file describing the detector distortion

**-D** DETECTOR_NAME, **--detector** DETECTOR_NAME
   Detector name (instead of pixel size+spline)

**-m** MASK, **--mask** MASK
   file containing the mask (for image reconstruction)

**-n** NPT, **--pt** NPT
   file with datapoints saved. Default: basename.npt

**--filter** FILTER
   select the filter, either mean(default), max or median

**-l** DISTANCE, **--distance** DISTANCE
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

**--fix-dist**
   fix the distance parameter

**--free-dist**
   free the distance parameter. Default: Activated

**--fix-poni1**
   fix the poni1 parameter

**--free-poni1**
   free the poni1 parameter. Default: Activated

**--fix-poni2**
   fix the poni2 parameter

**--free-poni2**
   free the poni2 parameter. Default: Activated

**--fix-rot1**
   fix the rot1 parameter

**--free-rot1**
   free the rot1 parameter. Default: Activated

**--fix-rot2**
   fix the rot2 parameter

**--free-rot2**
   free the rot2 parameter. Default: Activated

**--fix-rot3**
   fix the rot3 parameter

**--free-rot3**
   free the rot3 parameter. Default: Activated

**--fix-wavelength**
   fix the wavelength parameter. Default: Activated

**--free-wavelength**
   free the wavelength parameter. Default: Deactivated

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

**--npt** NPT_1D
   Number of point in 1D integrated pattern, Default: 1024

**--npt-azim** NPT_2D_AZIM
   Number of azimuthal sectors in 2D integrated images. Default: 360

**--npt-rad** NPT_2D_RAD
   Number of radial bins in 2D integrated images. Default: 400

**--unit** UNIT
   Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1, q_A^-1,
   r_mm. Default: 2th_deg

**--no-gui**
   force the program to run without a Graphical interface

**--no-interactive**
   force the program to run and exit without prompting for refinements

**-r** MAX_RINGS, **--ring** MAX_RINGS
   maximum number of rings to extract. Default: all accessible

**-k**, **--keep**
   Keep existing control point and append new

The main difference with pyFAI-calib is the way control-point hence
DebyeSherrer rings are extracted. While pyFAI-calib relies on the
contiguity of a region of peaks called massif; pyFAI-recalib knows
approximatly the geometry and is able to select the region where the
ring should be. From this region it selects automatically the various
peaks; making pyFAI-recalib able to run without graphical interface and
without human intervention (**--no-gui** and **--nointeractive**
options). Note that \`pyFAI-recalib\` program is obsolete as the same
functionality is available from within pyFAI-calib, using the
\`recalib\` command in the refinement process. Two option are available
for recalib: the numbe of rings to extract (similar to the **-r** option
of this program) and a new option which lets you choose between the
original \`massif\` algorithm and newer ones like \`blob\` and
\`watershed\` detection.
