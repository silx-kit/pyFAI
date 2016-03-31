#!/usr/bin/env python
import sys
import numpy

import fabio
from pyFAI.calibration import calib, get_detector
from pyFAI.calibrant import ALL_CALIBRANTS

def get_calibrant(calibrant_name):
    return ALL_CALIBRANTS[calibrant_name]

def calibration(img, calibrant_name, detector_name, wavelength):
    calibrant = get_calibrant(calibrant_name)
    calibrant.wavelength = wavelength
    detector = get_detector(detector_name)
    calib(img, calibrant, detector)

if __name__ == "__main__":
    img = fabio.open(sys.argv[1]).data
    print(sys.argv)
    calibration(img, *sys.argv[2:])
