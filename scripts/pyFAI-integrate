#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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
pyFAI-integrate

A graphical tool for performing azimuthal integration on series of files.


"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/11/2016"
__satus__ = "production"
import sys, logging, json, os, time, types, threading
import os.path
import numpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
import pyFAI, fabio
from pyFAI.opencl import ocl
import pyFAI.utils
import pyFAI.worker
from pyFAI.io import DefaultAiWriter
from pyFAI.io import HDF5Writer
from pyFAI.utils.shell import ProgressBar

try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.debug("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")


def integrate_gui(options, args):
    from pyFAI.gui import qt
    from pyFAI.integrate_widget import AIWidget

    app = qt.QApplication([])
    if not args:
        dia = qt.QFileDialog(directory=os.getcwd())
        dia.setFileMode(qt.QFileDialog.ExistingFiles)
        dia.exec_()
        try:
            args = [str(i) for i in  dia.selectedFiles()]
        except UnicodeEncodeError as err:
            logger.error("Problem with the name of some files: %s" % (err))
            args = [unicode(i) for i in  dia.selectedFiles()]

    window = AIWidget(args, options.output, options.format, options.slow, options.rapid, options.json)
    window.set_input_data(args)
    window.show()
    return app.exec_()


def integrate_shell(options, args):
    import json
    config = json.load(open(options.json))

    ai = pyFAI.worker.make_ai(config)
    worker = pyFAI.worker.Worker(azimuthalIntegrator=ai)
    # TODO this will init again the azimuthal integrator, there is a problem on the architecture
    worker.setJsonConfig(options.json)

    start_time = time.time()

    # Skip unexisting files
    image_filenames = []
    for item in args:
        if os.path.exists(item) and os.path.isfile(item):
            image_filenames.append(item)
        else:
            logger.warning("File %s do not exists. Ignored." % item)
    image_filenames = sorted(image_filenames)

    progress_bar = ProgressBar("Integration", len(image_filenames), 20)

    # Integrate files one by one
    for i, item in enumerate(image_filenames):
        logger.debug("Processing %s" % item)

        if len(item) > 100:
            message = op.basename(item)
        else:
            message = item
        progress_bar.update(i + 1, message=message)

        img = fabio.open(item)
        multiframe = img.nframes > 1

        if options.output and os.path.isdir(options.output):
            outpath = os.path.join(options.output, os.path.splitext(os.path.basename(item))[0])
        else:
            outpath = os.path.splitext(item)[0]

        if multiframe:
            writer = HDF5Writer(outpath + "_pyFAI.h5")
            writer.init(config)

            for i in range(img.nframes):
                data = img.getframe(i).data
                if worker.do_2D():
                    res = worker.process(data)
                else:
                    res = worker.process(data)
                    res = res.T[1]
                writer.write(res, index=i)
            writer.close()
        else:
            if worker.do_2D():
                filename = outpath + ".azim"
            else:
                filename = outpath + ".dat"
            data = img.data
            writer = DefaultAiWriter(filename, worker.ai)
            if worker.do_2D():
                worker.process(data, writer=writer)
            else:
                worker.process(data, writer=writer)
            writer.close()

    progress_bar.clear()
    logger.info("Processing done in %.3fs !" % (time.time() - start_time))


if __name__ == "__main__":
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
    options = parser.parse_args()

    # Analyse arguments and options
    args = pyFAI.utils.expand_args(options.args)

    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)

    if options.gui:
        result = integrate_gui(options, args)
    else:
        result = integrate_shell(options, args)
    sys.exit(result)
