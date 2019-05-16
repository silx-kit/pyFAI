# coding: utf-8
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
"bunch of utility function/static classes to handle testing environment"
from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/05/2019"

PACKAGE = "pyFAI"

import os
import sys
import threading
import unittest
import logging
import shutil
import contextlib
import tempfile
import getpass
import functools


from argparse import ArgumentParser
from ..third_party import six
from silx.resources import ExternalResources
from ..directories import testimages

logger = logging.getLogger(__name__)

TEST_HOME = os.path.dirname(os.path.abspath(__file__))


def copy(infile, outfile):
    "link or copy file according to the OS"
    if "link" in dir(os):
        os.link(infile, outfile)
    else:
        shutil.copy(infile, outfile)


class TestOptions(object):
    """
    Class providing useful stuff for preparing tests.
    """
    def __init__(self):
        self.WITH_QT_TEST = True
        """Qt tests are included"""

        self.WITH_QT_TEST_REASON = ""
        """Reason for Qt tests are disabled if any"""

        self.WITH_OPENCL_TEST = True
        """OpenCL tests are included"""

        self.WITH_GL_TEST = True
        """OpenGL tests are included"""

        self.WITH_GL_TEST_REASON = ""
        """Reason for OpenGL tests are disabled if any"""

        self.TEST_LOW_MEM = False
        """Skip tests using too much memory"""

        self.options = None
        self.timeout = 60  # timeout in seconds for downloading images
        # url_base = "http://forge.epn-campus.eu/attachments/download"
        self.url_base = "http://ftp.edna-site.org/pyFAI/testimages"
        self.resources = ExternalResources(PACKAGE,
                                           timeout=self.timeout,
                                           env_key=testimages,
                                           url_base=self.url_base)
        self.sem = threading.Semaphore()
        self.recompiled = False
        self.reloaded = False
        self.name = PACKAGE
        self.script_dir = None
        self.installed = False

        self.download_images = self.resources.download_all
        self.getimage = self.resources.getfile

        self._tempdir = None

    @property
    def low_mem(self):
        """For compatibility"""
        return self.TEST_LOW_MEM

    @property
    def opencl(self):
        """For compatibility"""
        return self.WITH_OPENCL_TEST

    def deep_reload(self):
        self.pyFAI = __import__(self.name)
        logger.info("%s loaded from %s", self.name, self.pyFAI.__file__)
        sys.modules[self.name] = self.pyFAI
        self.reloaded = True
        import pyFAI.utils.decorators
        pyFAI.utils.decorators.depreclog.setLevel(logging.ERROR)
        return self.pyFAI

    def forceBuild(self, remove_first=True):
        """
        Force the recompilation of pyFAI

        Nonesense, kept for legacy reasons
        """
        return

    def configure(self, parsed_options=None):
        """Configure the TestOptions class from the command line arguments and the
        environment variables
        """
        if parsed_options is not None and not parsed_options.gui:
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "Skipped by command line"
        elif os.environ.get('WITH_QT_TEST', 'True') == 'False':
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "Skipped by WITH_QT_TEST env var"
        elif sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "DISPLAY env variable not set"

        if parsed_options is not None and not parsed_options.opencl:
            self.WITH_OPENCL_TEST = False
            # That's an easy way to skip OpenCL tests
            # It disable the use of OpenCL on the full silx project
            os.environ['PYFAI_OPENCL'] = "False"
        elif os.environ.get('PYFAI_OPENCL', 'True') == 'False':
            self.WITH_OPENCL_TEST = False
            # That's an easy way to skip OpenCL tests
            # It disable the use of OpenCL on the full silx project
            os.environ['PYFAI_OPENCL'] = "False"

        if parsed_options is not None and not parsed_options.opengl:
            self.WITH_GL_TEST = False
            self.WITH_GL_TEST_REASON = "Skipped by command line"
        elif os.environ.get('WITH_GL_TEST', 'True') == 'False':
            self.WITH_GL_TEST = False
            self.WITH_GL_TEST_REASON = "Skipped by WITH_GL_TEST env var"

        if parsed_options is not None and parsed_options.low_mem:
            self.TEST_LOW_MEM = True
        elif os.environ.get('PYFAI_LOW_MEM', 'True') == 'False':
            self.TEST_LOW_MEM = True

    def add_parser_argument(self, parser):
        """Add extrat arguments to the test argument parser

        :param ArgumentParser parser: An argument parser
        """

        parser.add_argument("-x", "--no-gui", dest="gui", default=True,
                            action="store_false",
                            help="Disable the test of the graphical use interface")
        parser.add_argument("-g", "--no-opengl", dest="opengl", default=True,
                            action="store_false",
                            help="Disable tests using OpenGL")
        parser.add_argument("-o", "--no-opencl", dest="opencl", default=True,
                            action="store_false",
                            help="Disable the test of the OpenCL part")
        parser.add_argument("-l", "--low-mem", dest="low_mem", default=False,
                            action="store_true",
                            help="Disable test with large memory consumption (>100Mbyte")

    def get_test_env(self):
        """
        Returns an associated environment with a working project.
        """
        env = dict((str(k), str(v)) for k, v in os.environ.items())
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        return env

    def script_path(self, script_name, module_name):
        """Returns the script path according to it's location"""
        if self.installed:
            script = self.get_installed_script_path(script_name)
        else:
            import importlib
            module = importlib.import_module(module_name)
            script = module.__file__
        return script

    def get_installed_script_path(self, script):
        """
        Returns the path of the executable and the associated environment

        In Windows, it checks availability of script using .py .bat, and .exe
        file extensions.
        """
        if (sys.platform == "win32"):
            available_extensions = [".py", ".bat", ".exe"]
        else:
            available_extensions = [""]

        paths = os.environ.get("PATH", "").split(os.pathsep)
        for base in paths:
            # clean up extra quotes from paths
            if base.startswith('"') and base.endswith('"'):
                base = base[1:-1]
            for file_extension in available_extensions:
                script_path = os.path.join(base, script + file_extension)
                print(script_path)
                if os.path.exists(script_path):
                    # script found
                    return script_path
        # script not found
        logger.warning("Script '%s' not found in paths: %s", script, ":".join(paths))
        script_path = script
        return script_path

    def _initialize_tmpdir(self):
        """Initialize the temporary directory"""
        if not self._tempdir:
            with self.sem:
                if not self._tempdir:
                    self._tempdir = tempfile.mkdtemp("_" + getpass.getuser(),
                                                     self.name + "_")

    @property
    def tempdir(self):
        if not self._tempdir:
            self._initialize_tmpdir()
        return self._tempdir

    def clean_up(self):
        """Removes the temporary directory (and all its content !)"""
        with self.sem:
            if not self._tempdir:
                return
            if not os.path.isdir(self._tempdir):
                return
            for root, dirs, files in os.walk(self._tempdir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self._tempdir)
            self._tempdir = None


test_options = TestOptions()
"""Singleton containing util context of whole the tests"""

UtilsTest = test_options
"""For compatibility"""


def diff_img(ref, obt, comment=""):
    """
    Highlight the difference in images
    """
    assert ref.shape == obt.shape
    delta = abs(obt - ref)
    if delta.max() > 0:
        from ..gui.matplotlib import pyplot
        fig = pyplot.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        im_ref = ax1.imshow(ref)
        pyplot.colorbar(im_ref)
        ax1.set_title("%s ref" % comment)
        im_obt = ax2.imshow(obt)
        pyplot.colorbar(im_obt)
        ax2.set_title("%s obt" % comment)
        im_delta = ax3.imshow(delta)
        pyplot.colorbar(im_delta)
        ax3.set_title("delta")
        imax = delta.argmax()
        x = imax % ref.shape[-1]
        y = imax // ref.shape[-1]
        ax3.plot([x], [y], "o", scalex=False, scaley=False)
        fig.show()
        six.moves.input()


def diff_crv(ref, obt, comment=""):
    """
    Highlight the difference in vectors
    """
    assert ref.shape == obt.shape
    delta = abs(obt - ref)
    if delta.max() > 0:
        from ..gui.matplotlib import pyplot
        fig = pyplot.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        _im_ref = ax1.plot(ref, label="%s ref" % comment)
        _im_obt = ax1.plot(obt, label="%s obt" % comment)
        _im_delta = ax2.plot(delta, label="delta")
        fig.show()
        six.moves.input()


if sys.hexversion >= 0x030400F0:  # Python >= 3.4
    class ParametricTestCase(unittest.TestCase):
        pass

else:
    class ParametricTestCase(unittest.TestCase):
        """TestCase with subTest support for Python < 3.4.

        Add subTest method to support parametric tests.
        API is the same, but behavior differs:
        If a subTest fails, the following ones are not run.
        """

        _subtest_msg = None  # Class attribute to provide a default value

        @contextlib.contextmanager
        def subTest(self, msg=None, **params):
            """Use as unittest.TestCase.subTest method in Python >= 3.4."""
            # Format arguments as: '[msg] (key=value, ...)'
            param_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items()])
            self._subtest_msg = '[%s] (%s)' % (msg or '', param_str)
            yield
            self._subtest_msg = None

        def shortDescription(self):
            short_desc = super(ParametricTestCase, self).shortDescription()
            if self._subtest_msg is not None:
                # Append subTest message to shortDescription
                short_desc = ' '.join(
                    [msg for msg in (short_desc, self._subtest_msg) if msg])

            return short_desc if short_desc else None


# Test logging messages #######################################################

class TestLogging(logging.Handler):
    """Context checking the number of logging messages from a specified Logger.

    It disables propagation of logging message while running.

    This is meant to be used as a with statement, for example:

    >>> with TestLogging(logger, error=2, warning=0):
    >>>     pass  # Run tests here expecting 2 ERROR and no WARNING from logger
    ...

    :param logger: Name or instance of the logger to test.
                   (Default: root logger)
    :type logger: str or :class:`logging.Logger`
    :param int critical: Expected number of CRITICAL messages.
                         Default: Do not check.
    :param int error: Expected number of ERROR messages.
                      Default: Do not check.
    :param int warning: Expected number of WARNING messages.
                        Default: Do not check.
    :param int info: Expected number of INFO messages.
                     Default: Do not check.
    :param int debug: Expected number of DEBUG messages.
                      Default: Do not check.
    :param int notset: Expected number of NOTSET messages.
                       Default: Do not check.
    :raises RuntimeError: If the message counts are the expected ones.
    """

    def __init__(self, logger=None, critical=None, error=None,
                 warning=None, info=None, debug=None, notset=None):
        if logger is None:
            logger = logging.getLogger()
        elif not isinstance(logger, logging.Logger):
            logger = logging.getLogger(logger)
        self.logger = logger

        self.records = []

        self.count_by_level = {
            logging.CRITICAL: critical,
            logging.ERROR: error,
            logging.WARNING: warning,
            logging.INFO: info,
            logging.DEBUG: debug,
            logging.NOTSET: notset
        }

        super(TestLogging, self).__init__()

    def __enter__(self):
        """Context (i.e., with) support"""
        self.records = []  # Reset recorded LogRecords
        self.logger.addHandler(self)
        self.logger.propagate = False
        # ensure no log message is ignored
        self.entry_level = self.logger.level * 1
        self.logger.setLevel(logging.DEBUG)

    def __exit__(self, exc_type, exc_value, traceback):
        """Context (i.e., with) support"""
        self.logger.removeHandler(self)
        self.logger.propagate = True
        self.logger.setLevel(self.entry_level)

        for level, expected_count in self.count_by_level.items():
            if expected_count is None:
                continue

            # Number of records for the specified level_str
            count = len([r for r in self.records if r.levelno == level])
            if count != expected_count:  # That's an error
                # Resend record logs through logger as they where masked
                # to help debug
                for record in self.records:
                    self.logger.handle(record)
                raise RuntimeError(
                    'Expected %d %s logging messages, got %d' % (
                        expected_count, logging.getLevelName(level), count))

    def emit(self, record):
        """Override :meth:`logging.Handler.emit`"""
        self.records.append(record)


def test_logging(logger=None, critical=None, error=None,
                 warning=None, info=None, debug=None, notset=None):
    """Decorator checking number of logging messages.

    Propagation of logging messages is disabled by this decorator.

    In case the expected number of logging messages is not found, it raises
    a RuntimeError.

    >>> class Test(unittest.TestCase):
    ...     @test_logging('module_logger_name', error=2, warning=0)
    ...     def test(self):
    ...         pass  # Test expecting 2 ERROR and 0 WARNING messages

    :param logger: Name or instance of the logger to test.
                   (Default: root logger)
    :type logger: str or :class:`logging.Logger`
    :param int critical: Expected number of CRITICAL messages.
                         Default: Do not check.
    :param int error: Expected number of ERROR messages.
                      Default: Do not check.
    :param int warning: Expected number of WARNING messages.
                        Default: Do not check.
    :param int info: Expected number of INFO messages.
                     Default: Do not check.
    :param int debug: Expected number of DEBUG messages.
                      Default: Do not check.
    :param int notset: Expected number of NOTSET messages.
                       Default: Do not check.
    """
    def decorator(func):
        test_context = TestLogging(logger, critical, error,
                                   warning, info, debug, notset)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with test_context:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def create_fake_data(dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
                     detector="Pilatus300k", wavelength=1.54e-10,
                     calibrant="AgBh", Imax=1000, poissonian=True, offset=10):
    """Simulate a SAXS image with a small detector by default
    :return: image, azimuthalIngtegrator 
    """
    from .. import calibrant as pyFAI_calibrant
    from .. import azimuthalIntegrator
    import numpy
    cal = pyFAI_calibrant.get_calibrant(calibrant)
    cal.wavelength = wavelength
    ai = azimuthalIntegrator.AzimuthalIntegrator(dist, poni1, poni2,
                                                 rot1, rot2, rot3,
                                                 detector=detector, wavelength=wavelength)
    img = cal.fake_calibration_image(ai, Imax=Imax) + offset
    if poissonian:
        return numpy.random.poisson(img), ai
    else:
        return img, ai

