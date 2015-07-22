Calibration tool: MX-calibrate
==============================

Purpose
-------

Calibrate automatically a set of frames taken at various sample-detector distance.

This tool has been developed for ESRF MX-beamlines where an acceptable calibration is
usually present is the header of the image. PyFAI reads it and does a "recalib" on
each of them before exporting a linear regression of all parameters versus this distance.

Most standard calibrants are directly installed together with pyFAI.
If you prefer using your own, you can provide a "d-spacing" file
containing the spacing of Miller plans in Angstrom (in decreasing order).
Most crystal powders used for calibration are available in the American Mineralogist
database [AMD]_ or in the [COD]_.


Usage:
------

MX-Calibrate -w 1.54 -c CeO2 file1.cbf file2.cbf ...

Options:
--------
usage: MX-Calibrate -w 1.54 -c CeO2 file1.cbf file2.cbf ...

Calibrate automatically a set of frames taken at various sample-detector
distance. Return the linear regression of the fit in funtion of the sample-
setector distance.

positional arguments:
  FILE                  List of files to calibrate

optional arguments:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -v, --verbose         switch to debug/verbose mode
  -c FILE, --calibrant FILE
                        file containing d-spacing of the calibrant reference
                        sample (MANDATORY)
  -w WAVELENGTH, --wavelength WAVELENGTH
                        wavelength of the X-Ray beam in Angstrom
  -e ENERGY, --energy ENERGY
                        energy of the X-Ray beam in keV (hc=12.398419292keV.A)
  -P POLARIZATION_FACTOR, --polarization POLARIZATION_FACTOR
                        polarization factor, from -1 (vertical) to +1
                        (horizontal), default is 0, synchrotrons are around
                        0.95
  -b BACKGROUND, --background BACKGROUND
                        Automatic background subtraction if no value are
                        provided
  -d DARK, --dark DARK  list of dark images to average and subtract
  -f FLAT, --flat FLAT  list of flat images to average and divide
  -s SPLINE, --spline SPLINE
                        spline file describing the detector distortion
  -p PIXEL, --pixel PIXEL
                        size of the pixel in micron
  -D DETECTOR_NAME, --detector DETECTOR_NAME
                        Detector name (instead of pixel size+spline)
  -m MASK, --mask MASK  file containing the mask (for image reconstruction)
  --filter FILTER       select the filter, either mean(default), max or median
  --saturation SATURATION
                        consider all pixel>max*(1-saturation) as saturated and
                        reconstruct them
  -r MAX_RINGS, --ring MAX_RINGS
                        maximum number of rings to extract
  --weighted            weight fit by intensity
  -l DISTANCE, --distance DISTANCE
                        sample-detector distance in millimeter
  --tilt                Allow initially detector tilt to be refined (rot1,
                        rot2, rot3). Default: Activated
  --no-tilt             Deactivated tilt refinement and set all rotation to 0
  --dist DIST           sample-detector distance in meter
  --poni1 PONI1         poni1 coordinate in meter
  --poni2 PONI2         poni2 coordinate in meter
  --rot1 ROT1           rot1 in radians
  --rot2 ROT2           rot2 in radians
  --rot3 ROT3           rot3 in radians
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
  --no-gui              force the program to run without a Graphical interface
  --gui                 force the program to run with a Graphical interface
  --no-interactive      force the program to run and exit without prompting
                        for refinements
  --interactive         force the program to prompt for refinements
  --peak-picker PEAKPICKER
                        Uses the 'massif', 'blob' or 'watershed' peak-picker
                        algorithm (default: blob)

This tool has been developed for ESRF MX-beamlines where an acceptable
calibration is usually present is the header of the image. PyFAI reads it and
does a "recalib" on each of them before exporting a linear regression of all
parameters versus this distance.

Example:
--------


.. command-output:: MX-calibrate --help
    :nostderr:
