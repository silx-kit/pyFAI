#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


"""GUI tool for configuring azimuthal integration on series of files."""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/12/2018"
__satus__ = "production"
import sys
import logging
import time
import os.path
import fabio
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger("pyFAI")
import pyFAI.utils
import pyFAI.worker
from pyFAI.io import DefaultAiWriter
from pyFAI.io import HDF5Writer
from pyFAI.utils.shell import ProgressBar
from pyFAI import average

from argparse import ArgumentParser

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.debug("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")


def integrate_gui(options, args):
    from silx.gui import qt
    from pyFAI.gui.IntegrationDialog import IntegrationDialog
    app = qt.QApplication([])
    window = IntegrationDialog(args, options.output, json_file=options.json)
    window.set_input_data(args)
    window.show()
    return app.exec_()


def get_monitor_value(image, monitor_key):
    """Return the monitor value from an image using an header key.

    :param fabio.fabioimage.FabioImage image: Image containing the header
    :param str monitor_key: Key containing the monitor
    :return: returns the monitor else returns 1.0
    :rtype: float
    """
    if monitor_key is None or monitor_key == "":
        return 1.0
    try:
        monitor = average.get_monitor_value(image, monitor_key)
        return monitor
    except average.MonitorNotFound:
        logger.warning("Monitor %s not found. No normalization applied.", monitor_key)
        return 1.0
    except Exception as e:
        logger.warning("Fail to load monitor. No normalization applied. %s", str(e))
        return 1.0


def integrate_shell(options, args):
    import json
    with open(options.json) as f:
        config = json.load(f)

    worker = pyFAI.worker.Worker()
    worker.set_config(config, consume_keys=True)

    # Check unused keys
    for key in config.keys():
        logger.warning("Configuration key '%s' from json file '%s' is unused", key, options.json)

    worker.safe = False  # all processing are expected to be the same.
    start_time = time.time()

    # Skip unexisting files
    image_filenames = []
    for item in args:
        if os.path.exists(item) and os.path.isfile(item):
            image_filenames.append(item)
        else:
            logger.warning("File %s do not exists. Ignored.", item)
    image_filenames = sorted(image_filenames)

    progress_bar = ProgressBar("Integration", len(image_filenames), 20)

    # Integrate files one by one
    for i, item in enumerate(image_filenames):
        logger.debug("Processing %s", item)

        if len(item) > 100:
            message = os.path.basename(item)
        else:
            message = item
        progress_bar.update(i + 1, message=message)

        img = fabio.open(item)
        multiframe = img.nframes > 1

        custom_ext = True
        if options.output:
            if os.path.isdir(options.output):
                outpath = os.path.join(options.output, os.path.splitext(os.path.basename(item))[0])
            else:
                outpath = os.path.abspath(options.output)
                custom_ext = False
        else:
            outpath = os.path.splitext(item)[0]

        if custom_ext:
            if multiframe:
                outpath = outpath + "_pyFAI.h5"
            else:
                if worker.do_2D():
                    outpath = outpath + ".azim"
                else:
                    outpath = outpath + ".dat"
        if multiframe:
            writer = HDF5Writer(outpath)
            writer.init(config)

            for i in range(img.nframes):
                fimg = img.getframe(i)
                normalization_factor = get_monitor_value(fimg, options.monitor_key)
                data = fimg.data
                res = worker.process(data=data,
                                     metadata=fimg.header,
                                     normalization_factor=normalization_factor)
                if not worker.do_2D():
                    res = res.T[1]
                writer.write(res, index=i)
            writer.close()
        else:
            normalization_factor = get_monitor_value(img, options.monitor_key)
            data = img.data
            writer = DefaultAiWriter(outpath, worker.ai)
            worker.process(data,
                           normalization_factor=normalization_factor,
                           writer=writer)
            writer.close()

    progress_bar.clear()
    logger.info("Processing done in %.3fs !", (time.time() - start_time))
    return 0


def main():
    usage = "pyFAI-integrate [options] file1.edf file2.edf ..."
    version = "pyFAI-integrate version %s from %s" % (pyFAI.version, pyFAI.date)
    description = """
    PyFAI-integrate is a graphical interface (based on Python/Qt4) to perform azimuthal integration
on a set of files. It exposes most of the important options available within pyFAI and allows you
to select a GPU (or an openCL platform) to perform the calculation on."""
    epilog = """PyFAI-integrate saves all parameters in a .azimint.json (hidden) file. This JSON file
is an ascii file which can be edited and used to configure online data analysis using
the LImA plugin of pyFAI.

Nota: there is bug in debian6 making the GUI crash (to be fixed inside pyqt)
http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348"""
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose", default=False,
                        help="switch to verbose/debug mode")
    parser.add_argument("-o", "--output",
                        dest="output", default=None,
                        help="Directory or file where to store the output data")
    parser.add_argument("-f", "--format",
                        dest="format", default=None,
                        help="output data format (can be HDF5)")
    parser.add_argument("-s", "--slow-motor",
                        dest="slow", default=None,
                        help="Dimension of the scan on the slow direction (makes sense only with HDF5)")
    parser.add_argument("-r", "--fast-motor",
                        dest="rapid", default=None,
                        help="Dimension of the scan on the fast direction (makes sense only with HDF5)")
    parser.add_argument("--no-gui",
                        dest="gui", default=True, action="store_false",
                        help="Process the dataset without showing the user interface.")
    parser.add_argument("-j", "--json",
                        dest="json", default=".azimint.json",
                        help="Configuration file containing the processing to be done")
    parser.add_argument("args", metavar='FILE', type=str, nargs='*',
                        help="Files to be integrated")
    parser.add_argument("--monitor-name", dest="monitor_key", default=None,
                        help="Name of the monitor in the header of each input \
                        files. If defined the contribution of each input file \
                        is divided by the monitor. If the header does not \
                        contain or contains a wrong value, the contribution \
                        of the input file is ignored.\
                        On EDF files, values from 'counter_pos' can accessed \
                        by using the expected mnemonic. \
                        For example 'counter/bmon'.")
    options = parser.parse_args()

    # Analysis arguments and options
    args = pyFAI.utils.expand_args(options.args)

    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)

    if options.gui:
        result = integrate_gui(options, args)
    else:
        result = integrate_shell(options, args)
    sys.exit(result)


if __name__ == "__main__":
    main()
