Calibration tool: check_calib
=============================

Purpose
-------

Check_calib is a research tool aiming at validating both the geometric
calibration and everything else like flat-field correction, distortion 
correction. Maybe the future lies over there ...

Usage: 
------

check_calib [options] -p param.poni image.edf

Options:
--------

  -h, --help            show help message and exit
  -V, --version         print version of the program and quit
  -v, --verbose         switch to debug mode
  -d FILE, --dark=FILE  file containing the dark images to subtract
  -f FILE, --flat=FILE  file containing the flat images to divide
  -m FILE, --mask=FILE  file containing the mask
  -p FILE, --poni=FILE  file containing the diffraction parameter (poni-file)
  -e ENERGY, --energy=ENERGY
                        energy of the X-Ray beam in keV (hc=12.398419292keV.A)
  -w WAVELENGTH, --wavelength=WAVELENGTH
                        wavelength of the X-Ray beam in Angstrom
