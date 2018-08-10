#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
pyFAI-calib2

A tool for determining the geometry of a detector using a reference sample.


"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/08/2018"
__status__ = "production"

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger("pyFAI-calib2")
logger_uncaught = logging.getLogger("pyFAI-calib2.UNCAUGHT")

from silx.gui import qt
# Make sure matplotlib is loaded first by silx
import silx.gui.plot.matplotlib

from pyFAI.gui.calibration.CalibrationWindow import CalibrationWindow

import pyFAI.resources
import pyFAI.calibrant
import pyFAI.detectors
# TODO: This should be removed
import pyFAI.gui.cli_calibration
import fabio

from pyFAI.third_party.argparse import ArgumentParser

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    logger.debug("No socket opened for debugging. Please install rfoo")


def configure_parser_arguments(parser):
    # From AbstractCalibration

    parser.add_argument("args", metavar="FILE", help="List of files to calibrate", nargs='*')

    # Not yet used
    parser.add_argument("-o", "--out", dest="outfile",
                        help="Filename where processed image is saved", metavar="FILE",
                        default=None)
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="debug", default=False,
                        help="switch to debug/verbose mode")
    parser.add_argument('--debug',
                        dest="debug",
                        action="store_true",
                        default=False,
                        help='Set logging system in debug mode')

    # Settings
    parser.add_argument("-c", "--calibrant", dest="spacing", metavar="FILE",
                        help="Calibrant name or file containing d-spacing of the reference sample (MANDATORY, case sensitive !)",
                        default=None)
    parser.add_argument("-w", "--wavelength", dest="wavelength", type=float,
                        help="wavelength of the X-Ray beam in Angstrom. Mandatory ", default=None)
    parser.add_argument("-e", "--energy", dest="energy", type=float,
                        help="energy of the X-Ray beam in keV (hc=%skeV.A)." % pyFAI.units.hc, default=None)
    parser.add_argument("-P", "--polarization", dest="polarization_factor",
                        type=float, default=None,
                        help="polarization factor, from -1 (vertical) to +1 (horizontal)," +
                        " default is None (no correction), synchrotrons are around 0.95")
    parser.add_argument("-D", "--detector", dest="detector_name",
                        help="Detector name (instead of pixel size+spline)", default=None)
    parser.add_argument("-m", "--mask", dest="mask",
                        help="file containing the mask (for image reconstruction)", default=None)
    parser.add_argument("-p", "--pixel", dest="pixel",
                        help="size of the pixel in micron", default=None)

    # Not yet used
    parser.add_argument("-i", "--poni", dest="poni", metavar="FILE",
                        help="file containing the diffraction parameter (poni-file). MANDATORY for pyFAI-recalib!",
                        default=None)
    # Not yet used
    parser.add_argument("-b", "--background", dest="background",
                        help="Automatic background subtraction if no value are provided",
                        default=None)
    # Not yet used
    parser.add_argument("-d", "--dark", dest="dark",
                        help="list of comma separated dark images to average and subtract", default=None)
    # Not yet used
    parser.add_argument("-f", "--flat", dest="flat",
                        help="list of comma separated flat images to average and divide", default=None)
    # Not yet used
    parser.add_argument("-s", "--spline", dest="spline",
                        help="spline file describing the detector distortion", default=None)
    # Not yet used
    parser.add_argument("-n", "--pt", dest="npt",
                        help="file with datapoints saved. Default: basename.npt", default=None)
    # Not yet used
    parser.add_argument("--filter", dest="filter",
                        help="select the filter, either mean(default), max or median",
                        default=None)

    # Geometry
    parser.add_argument("-l", "--distance", dest="distance", type=float,
                        help="sample-detector distance in millimeter. Default: 100mm", default=None)
    parser.add_argument("--dist", dest="dist", type=float,
                        help="sample-detector distance in meter. Default: 0.1m", default=None)
    parser.add_argument("--poni1", dest="poni1", type=float,
                        help="poni1 coordinate in meter. Default: center of detector", default=None)
    parser.add_argument("--poni2", dest="poni2", type=float,
                        help="poni2 coordinate in meter. Default: center of detector", default=None)
    parser.add_argument("--rot1", dest="rot1", type=float,
                        help="rot1 in radians. default: 0", default=None)
    parser.add_argument("--rot2", dest="rot2", type=float,
                        help="rot2 in radians. default: 0", default=None)
    parser.add_argument("--rot3", dest="rot3", type=float,
                        help="rot3 in radians. default: 0", default=None)
    # Constraints
    parser.add_argument("--fix-wavelength", dest="fix_wavelength",
                        help="fix the wavelength parameter. Default: Activated", default=True, action="store_true")
    parser.add_argument("--free-wavelength", dest="fix_wavelength",
                        help="free the wavelength parameter. Default: Deactivated ", default=True, action="store_false")
    parser.add_argument("--fix-dist", dest="fix_dist",
                        help="fix the distance parameter", default=None, action="store_true")
    parser.add_argument("--free-dist", dest="fix_dist",
                        help="free the distance parameter. Default: Activated", default=None, action="store_false")
    parser.add_argument("--fix-poni1", dest="fix_poni1",
                        help="fix the poni1 parameter", default=None, action="store_true")
    parser.add_argument("--free-poni1", dest="fix_poni1",
                        help="free the poni1 parameter. Default: Activated", default=None, action="store_false")
    parser.add_argument("--fix-poni2", dest="fix_poni2",
                        help="fix the poni2 parameter", default=None, action="store_true")
    parser.add_argument("--free-poni2", dest="fix_poni2",
                        help="free the poni2 parameter. Default: Activated", default=None, action="store_false")
    parser.add_argument("--fix-rot1", dest="fix_rot1",
                        help="fix the rot1 parameter", default=None, action="store_true")
    parser.add_argument("--free-rot1", dest="fix_rot1",
                        help="free the rot1 parameter. Default: Activated", default=None, action="store_false")
    parser.add_argument("--fix-rot2", dest="fix_rot2",
                        help="fix the rot2 parameter", default=None, action="store_true")
    parser.add_argument("--free-rot2", dest="fix_rot2",
                        help="free the rot2 parameter. Default: Activated", default=None, action="store_false")
    parser.add_argument("--fix-rot3", dest="fix_rot3",
                        help="fix the rot3 parameter", default=None, action="store_true")
    parser.add_argument("--free-rot3", dest="fix_rot3",
                        help="free the rot3 parameter. Default: Activated", default=None, action="store_false")

    # Not yet used
    parser.add_argument("--tilt", dest="tilt",
                        help="Allow initially detector tilt to be refined (rot1, rot2, rot3). Default: Activated", default=None, action="store_true")
    # Not yet used
    parser.add_argument("--no-tilt", dest="tilt",
                        help="Deactivated tilt refinement and set all rotation to 0", default=None, action="store_false")
    # Not yet used
    parser.add_argument("--saturation", dest="saturation",
                        help="consider all pixel>max*(1-saturation) as saturated and "
                        "reconstruct them, default: 0 (deactivated)",
                        default=0, type=float)
    # Not yet used
    parser.add_argument("--weighted", dest="weighted",
                        help="weight fit by intensity, by default not.",
                        default=False, action="store_true")
    # Not yet used
    parser.add_argument("--npt", dest="nPt_1D",
                        help="Number of point in 1D integrated pattern, Default: 1024", type=int,
                        default=None)
    # Not yet used
    parser.add_argument("--npt-azim", dest="nPt_2D_azim",
                        help="Number of azimuthal sectors in 2D integrated images. Default: 360", type=int,
                        default=None)
    # Not yet used
    parser.add_argument("--npt-rad", dest="nPt_2D_rad",
                        help="Number of radial bins in 2D integrated images. Default: 400", type=int,
                        default=None)
    # Not yet used
    parser.add_argument("--unit", dest="unit",
                        help="Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1,"
                        " q_A^-1, r_mm. Default: 2th_deg", type=str, default="2th_deg")
    # Not yet used
    parser.add_argument("--no-gui", dest="gui",
                        help="force the program to run without a Graphical interface",
                        default=True, action="store_false")
    # Not yet used
    parser.add_argument("--no-interactive", dest="interactive",
                        help="force the program to run and exit without prompting"
                        " for refinements", default=True, action="store_false")

    # From Calibration
    # Not yet used
    parser.add_argument("-r", "--reconstruct", dest="reconstruct",
                        help="Reconstruct image where data are masked or <0  (for Pilatus " +
                        "detectors or detectors with modules)",
                        action="store_true", default=False)
    # Not yet used
    parser.add_argument("-g", "--gaussian", dest="gaussian",
                        help="""Size of the gaussian kernel.
Size of the gap (in pixels) between two consecutive rings, by default 100
Increase the value if the arc is not complete;
decrease the value if arcs are mixed together.""", default=None)
    # Not yet used
    parser.add_argument("--square", dest="square", action="store_true",
                        help="Use square kernel shape for neighbor search instead of diamond shape",
                        default=False)


description = """Calibrate the diffraction setup geometry based on
Debye-Sherrer rings images without a priori knowledge of your setup.
You will need to provide a calibrant or a "d-spacing" file containing the
spacing of Miller plans in Angstrom (in decreasing order).
%s or search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php""" % str(pyFAI.calibrant.ALL_CALIBRANTS)

epilog = """The output of this program is a "PONI" file containing the
detector description and the 6 refined parameters (distance, center, rotation)
and wavelength. An 1D and 2D diffraction patterns are also produced.
(.dat and .azim files)"""


def parse_pixel_size(pixel_size):
    """Convert a comma separated sting into pixel size

    :param str pixel_size: String containing pixel size in micron
    :rtype: Tuple[float,float]
    :returns: Returns floating point pixel size in meter
    """
    sp = pixel_size.split(",")
    if len(sp) >= 2:
        try:
            result = [float(i) * 1e-6 for i in sp[:2]]
        except Exception:
            logger.error("Error in reading pixel size_2")
            raise ValueError("Not a valid pixel size")
    elif len(sp) == 1:
        px = sp[0]
        try:
            result = [float(px) * 1e-6, float(px) * 1e-6]
        except Exception:
            logger.error("Error in reading pixel size_1")
            raise ValueError("Not a valid pixel size")
    else:
        logger.error("Error in reading pixel size_0")
        raise ValueError("Not a valid pixel size")
    return result


def setup(model):
    usage = "pyFAI-calib2 [options] input_image.edf"
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    version = "calibration from pyFAI  version %s: %s" % (pyFAI.version, pyFAI.date)
    parser.add_argument("-V", "--version", action='version', version=version)
    configure_parser_arguments(parser)

    # Analyse aruments and options
    options = parser.parse_args()
    args = options.args

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    # Settings
    settings = model.experimentSettingsModel()
    if options.spacing:
        calibrant = None
        if options.spacing in pyFAI.calibrant.CALIBRANT_FACTORY:
            calibrant = pyFAI.calibrant.CALIBRANT_FACTORY(options.spacing)
        elif os.path.isfile(options.spacing):
            calibrant = pyFAI.calibrant.Calibrant(options.spacing)
        else:
            logger.error("No such Calibrant / d-Spacing file: %s", options.spacing)
        if calibrant:
            settings.calibrantModel().setCalibrant(calibrant)

    if options.wavelength:
        settings.wavelength().setValue(options.wavelength)

    if options.energy:
        settings.wavelength().setValue(pyFAI.units.hc / options.energy)

    if options.polarization_factor:
        settings.polarizationFactor(options.polarization_factor)

    if options.detector_name:
        detector = pyFAI.gui.cli_calibration.get_detector(options.detector_name, args)
        if options.pixel:
            logger.warning("Detector model already specified. Pixel size argument ignored.")
    elif options.pixel:
        pixel_size = parse_pixel_size(options.pixel)
        detector = pyFAI.detectors.Detector(pixel1=pixel_size[0], pixel2=pixel_size[0])
    else:
        detector = None

    if options.spline:
        if detector is None:
            detector = pyFAI.detectors.Detector(splineFile=options.spline)
        elif detector.__class__ is pyFAI.detectors.Detector or detector.HAVE_TAPER:
            detector.set_splineFile(options.spline)
        else:
            logger.warning("Spline file not supported with this kind of detector. Argument ignored.")

    settings.detectorModel().setDetector(detector)

    if options.mask:
        settings.maskFile().setValue(options.mask)
        with fabio.open(options.mask) as mask:
            settings.mask().setValue(mask.data)

    if len(args) == 0:
        pass
    elif len(args) == 1:
        image_file = args[0]
        settings.imageFile().setValue(image_file)
        with fabio.open(image_file) as image:
            settings.image().setValue(image.data)
    else:
        logger.error("Too much images provided. Only one is expected")

    # Geometry
    # FIXME it will not be used cause the fitted geometry will be overwrited
    geometry = model.fittedGeometry()
    if options.distance:
        geometry.distance().setValue(1e-3 * options.distance)
    if options.dist:
        geometry.distance().setValue(options.dist)
    if options.dist:
        geometry.poni1().setValue(options.poni1)
    if options.dist:
        geometry.poni2().setValue(options.poni2)
    if options.dist:
        geometry.rotation1().setValue(options.rot1)
    if options.dist:
        geometry.rotation2().setValue(options.rot2)
    if options.dist:
        geometry.rotation3().setValue(options.rot3)

    # Constraints
    constraints = model.geometryConstraintsModel()
    if options.fix_wavelength is not None:
        constraints.wavelength().setFixed(options.fix_wavelength)
    if options.fix_dist is not None:
        constraints.distance().setFixed(options.fix_dist)
    if options.fix_poni1 is not None:
        constraints.poni1().setFixed(options.fix_poni1)
    if options.fix_poni2 is not None:
        constraints.poni2().setFixed(options.fix_poni2)
    if options.fix_rot1 is not None:
        constraints.rotation1().setFixed(options.fix_rot1)
    if options.fix_rot2 is not None:
        constraints.rotation2().setFixed(options.fix_rot2)
    if options.fix_rot3 is not None:
        constraints.rotation3().setFixed(options.fix_rot3)

    # Integration
    if options.unit:
        unit = pyFAI.units.to_unit(options.unit)
        model.integrationSettingsModel().radialUnit().setValue(unit)

    if options.outfile:
        logger.error("outfile option not supported")
    if options.debug:
        logger.error("debug option not supported")

    if options.reconstruct:
        logger.error("reconstruct option not supported")
    if options.gaussian:
        logger.error("gaussian option not supported")
    if options.square:
        logger.error("square option not supported")
    if options.pixel:
        logger.error("pixel option not supported")

    # FIXME poni file should be supported
    if options.poni:
        logger.error("poni option not supported")
    if options.background:
        logger.error("background option not supported")
    if options.dark:
        logger.error("dark option not supported")
    if options.flat:
        logger.error("flat option not supported")
    if options.npt:
        logger.error("npt option not supported")
    if options.filter:
        logger.error("filter option not supported")

    if options.tilt:
        logger.error("tilt option not supported")
    if options.saturation:
        logger.error("saturation option not supported")
    if options.weighted:
        logger.error("weighted option not supported")
    if options.nPt_1D:
        logger.error("nPt_1D option not supported")
    if options.nPt_2D_azim:
        logger.error("nPt_2D_azim option not supported")
    if options.nPt_2D_rad:
        logger.error("nPt_2D_rad option not supported")

    if options.gui is not True:
        logger.error("gui option not supported")
    if options.interactive is not True:
        logger.error("interactive option not supported")


def logUncaughtExceptions(exceptionClass, exception, stack):
    try:
        import traceback
        if stack is not None:
            # Mimic the syntax of the default Python exception
            message = (''.join(traceback.format_tb(stack)))
            message = '{1}\nTraceback (most recent call last):\n{2}{0}: {1}'.format(exceptionClass.__name__, exception, message)
        else:
            # There is no backtrace
            message = '{0}: {1}'.format(exceptionClass.__name__, exception)
        logger_uncaught.error(message)
    except Exception:
        # Make sure there is no problem at all in this function
        try:
            logger_uncaught.error(exception)
        except Exception:
            print("Error:" + str(exception))


def main():
    sys.excepthook = logUncaughtExceptions
    app = qt.QApplication([])
    pyFAI.resources.silx_integration()
    widget = CalibrationWindow()
    setup(widget.model())
    widget.setVisible(True)
    app.exec_()


if __name__ == "__main__":
    main()
