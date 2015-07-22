Preprocessing tool: detector2nexus
==================================

Convert a complex detector definition (multiple modules, possibly in 3D) into
a single NeXus detector definition together with the mask (and much more in
the future)


Purpose
-------

Convert a detector to NeXus detector definition for pyFAI.

Usage:
------

detector2nexus [options] [options] -o nxs.h5


options:
--------

  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -o OUTPUT, --output OUTPUT
                        Output nexus file, unless detector_name.h5
  -n NAME, --name NAME  name of the detector
  -m MASK, --mask MASK  mask corresponding to the detector
  -D DETECTOR, --detector DETECTOR
                        Base detector name (see documentation of
                        pyFAI.detectors
  -s SPLINEFILE, --splinefile SPLINEFILE
                        Geometric distortion file from FIT2D
  -dx DX, --x-corr DX   Geometric correction for pilatus
  -dy DY, --y-corr DY   Geometric correction for pilatus
  -p PIXEL, --pixel PIXEL
                        pixel size (comma separated): x,y
  -S SHAPE, --shape SHAPE
                        shape of the detector (comma separated): x,y
  -d DARK, --dark DARK  Dark noise to be subtracted
  -f FLAT, --flat FLAT  Flat field correction
  -v, --verbose         switch to verbose/debug mode

.. command-output:: detector2nexus --help
    :nostderr: