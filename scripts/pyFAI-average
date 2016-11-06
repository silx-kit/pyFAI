#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
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
pyFAI-average is a small utility that averages out a serie of files,
for example for dark, or flat or calibration images
"""
__author__ = "Jerome Kieffer, Picca Frédéric-Emmanuel"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/09/2016"
__status__ = "production"

import os
import sys
import logging

# Must be defined before libraries which define there own basicConfig

class PreEmitStreamHandler(logging.StreamHandler):
    """Add a hook before emit function"""

    def emit(self, record):
        """
        @type record: logging.LogRecord
        """
        self.pre_emit()
        super(PreEmitStreamHandler, self).emit(record)

    def pre_emit(self):
        pass

# Same as basicConfig with a custom handler but portable Python 2 and 3
log_root = logging.getLogger()
log_handler = PreEmitStreamHandler()
log_root.addHandler(log_handler)
log_root.setLevel(logging.INFO)

logger = logging.getLogger("average")

import fabio
import pyFAI
import pyFAI.utils
import pyFAI.utils.shell
import pyFAI.utils.stringutil
from pyFAI import average

try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser


def parse_algorithms(options):
    """Return a list of initilized algorithms from the command line"""

    methods = options.method.split(",")
    methods = set(methods) - set([""])

    if options.quantiles:
        for sep in "-,":
            if sep in options.quantiles:
                q1, q2 = options.quantiles.split(sep, 1)
                break
        else:
            q1 = float(options.quantiles)
            q2 = 1.0 - q1
        quantiles = (float(q1), float(q2))

        if "quanties" not in methods:
            logger.warning("Add quantiles to the set of methods as quantiles parameters is defined.")
            methods.add("quantiles")
    else:
        quantiles = None
        if "quantiles" in methods:
            logger.warning("Quantiles method defined but no parameters set. Method skipped.")
            methods.remove("quantiles")

    if len(methods) == 0:
        logger.warning("No method defined. Add default mean method")
        methods.add("mean")

    result = []
    for method in methods:
        if not average.is_algorithm_name_exists(method):
            logger.warning("Method name '%s' unknown. Method skipped.")
            continue

        try:
            algorithm = average.create_algorithm(method, options.cutoff, quantiles)
        except average.AlgorithmCreationError as e:
            logger.warning("Method skipped: %s", e)
            continue

        result.append(algorithm)

    return result


def parse_writer(input_images, options, algorithms):
    """Return a writer by using information from the command line"""
    output = options.output
    file_format = options.format

    if output:
        template = output
        if len(algorithms) > 1 and "{method_name}" not in template:
            # make sure the template will create multi files
            base, ext = os.path.splitext(template)
            template = base + "_{method_name}" + ext
    else:
        prefix = "{common_prefix}"
        suffix = ""
        if options.cutoff:
            suffix += "_cutoff_{cutoff}_std" % options.cutoff
        suffix += "_{image_count}_files.{file_format}"
        template = prefix + "{method_name}" + suffix

    formats = {
        "common_prefix": os.path.commonprefix(input_images),
        "image_count": len(input_images),
        "cutoff": options.cutoff,
        "file_format": file_format,
    }

    output = pyFAI.utils.stringutil.safe_format(template, formats)
    return average.MultiFilesAverageWriter(output, file_format)


class ShellAverageObserver(average.AverageObserver):
    """Display average processing using a shell progress bar"""

    def __init__(self):
        self.__bar = None
        self.__size = 40

    def image_loaded(self, fabio_image, image_index, images_count):
        if self.__bar is None:
            self.__bar = pyFAI.utils.shell.ProgressBar("Loading", images_count, self.__size)
        self.__bar.update(image_index, fabio_image.filename)

    def process_started(self):
        if self.__bar is not None:
            self.__bar.clear()
            self.__bar = None

    def algorithm_started(self, algorithm):
        if self.__bar is not None:
            self.__bar.clear()
            self.__bar = None

    def frame_processed(self, algorithm, frame_index, frames_count):
        if self.__bar is None:
            title = "Process %s" % algorithm.name
            self.__frames_count = frames_count + 1
            self.__bar = pyFAI.utils.shell.ProgressBar(title, self.__frames_count, self.__size)
        self.__bar.update(frame_index, "Feeding frames")

    def result_processing(self, algorithm):
        self.__bar.update(self.__frames_count - 1, "Computing result")

    def algorithm_finished(self, algorithm):
        self.__bar.clear()
        self.__bar = None
        print("%s reduction finished" % algorithm.name.capitalize())

    def process_finished(self):
        pass

    def clear(self):
        if self.__bar is not None:
            self.__bar.clear()

def main():
    usage = "pyFAI-average [options] [options] -o output.edf file1.edf file2.edf ..."
    version = "pyFAI-average version %s from %s" % (pyFAI.version, pyFAI.date)
    description = """
    This tool can be used to average out a set of dark current images using
    mean or median filter (along the image stack). One can also reject outliers
    be specifying a cutoff (remove cosmic rays / zingers from dark)
    """
    epilog = """It can also be used to merge many images from the same sample when using a small beam
    and reduce the spotty-ness of Debye-Sherrer rings. In this case the "max-filter" is usually
    recommended.
    """
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-o", "--output", dest="output",
                      type=str, default=None,
                      help="Output/ destination of average image")
    parser.add_argument("-m", "--method", dest="method",
                      type=str, default="",
                      help="Method used for averaging, can be 'mean' \
                      (default) or 'min', 'max', 'median', 'sum', 'quantiles'\
                      , 'cutoff', 'std'. Multiple filters can be defined with \
                      ',' separator.")
    parser.add_argument("-c", "--cutoff", dest="cutoff", type=float, default=None,
                  help="Take the mean of the average +/- cutoff * std_dev.")
    parser.add_argument("-F", "--format", dest="format", type=str, default="edf",
                  help="Output file/image format (by default EDF)")
    parser.add_argument("-d", "--dark", dest="dark", type=str, default=None,
                  help="Dark noise to be subtracted")
    parser.add_argument("-f", "--flat", dest="flat", type=str, default=None,
                  help="Flat field correction")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=None,
                      help="switch to verbose/debug mode")
    parser.add_argument("-q", "--quantiles", dest="quantiles", default=None,
                      help="average out between two quantiles -q 0.20-0.90")
    parser.add_argument("--monitor-name", dest="monitor_key", default=None,
                      help="Name of the monitor in the header of each input \
                      files. If defined the contribution of each input file \
                      is divided by the monitor. If the header does not \
                      contain or contains a wrong value, the contribution of \
                      the input file is ignored.\
                      On EDF files, values from 'counter_pos' can accessed by \
                      using the expected mnemonic. \
                      For example 'counter/bmon'.")
    parser.add_argument("--quiet", dest="verbose", default=None, action="store_false",
                      help="Only error messages are printed out")
    parser.add_argument("args", metavar='FILE', type=str, nargs='+',
                        help="Files to be processed")

    options = parser.parse_args()

    # logging
    if options.verbose is True:
        average.logger.setLevel(logging.DEBUG)
    elif options.verbose is False:
        average.logger.setLevel(logging.ERROR)
    else:
        average.logger.setLevel(logging.WARN)

    # shell output
    if options.verbose is not False:
        observer = ShellAverageObserver()
        # clean up the progress bar before displaying a log
        log_handler.pre_emit = observer.clear
    else:
        observer = None

    # Analyze arguments and options
    images = pyFAI.utils.expand_args(options.args)

    if options.flat:
        flats = pyFAI.utils.expand_args([options.flat])
    else:
        flats = None

    if options.dark:
        darks = pyFAI.utils.expand_args([options.dark])
    else:
        darks = None

    algorithms = parse_algorithms(options)
    if len(algorithms) == 0:
        logger.warning("Configure process with a mean filter")
        algorithms = [average.MeanAveraging()]

    writer = parse_writer(images, options, algorithms)

    if images:
        process = average.Average()
        process.set_observer(observer)
        process.set_images(images)
        process.set_dark(darks)
        process.set_flat(flats)
        #average.set_correct_flat_from_dark(correct_flat_from_dark)
        process.set_monitor_name(options.monitor_key)
        process.set_pixel_filter(threshold=0, minimum=None, maximum=None)
        for algorithm in algorithms:
            process.add_algorithm(algorithm)
        process.set_writer(writer)
        process.process()
    else:
        logger.warning("No input file specified.")

if __name__ == "__main__":
    main()
