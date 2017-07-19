# coding: utf-8
#
#    Copyright (C) 2012-2016 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "18/07/2017"

PACKAGE = "pyFAI"
DATA_KEY = "PYFAI_DATA"

if __name__ == "__main__":
    __name__ = "pyFAI.test"

import os
import sys
import threading
import unittest
import logging
import numpy
import shutil
from argparse import ArgumentParser
from ..utils import six
from silx.resources import ExternalResources

logger = logging.getLogger(__name__)

TEST_HOME = os.path.dirname(os.path.abspath(__file__))


def copy(infile, outfile):
    "link or copy file according to the OS"
    if "link" in dir(os):
        os.link(infile, outfile)
    else:
        shutil.copy(infile, outfile)


class UtilsTest(object):
    """
    Static class providing useful stuff for preparing tests.
    """
    options = None
    timeout = 60  # timeout in seconds for downloading images
    # url_base = "http://forge.epn-campus.eu/attachments/download"
    url_base = "http://ftp.edna-site.org/pyFAI/testimages"
    resources = ExternalResources(PACKAGE, timeout=timeout, env_key=DATA_KEY,
                                  url_base=url_base)
    sem = threading.Semaphore()
    recompiled = False
    reloaded = False
    name = PACKAGE
    script_dir = None

    tempdir = resources.tempdir
    download_images = resources.download_all
    clean_up = resources.clean_up
    getimage = resources.getfile

    @classmethod
    def deep_reload(cls):
        cls.pyFAI = __import__(cls.name)
        logger.info("%s loaded from %s", cls.name, cls.pyFAI.__file__)
        sys.modules[cls.name] = cls.pyFAI
        cls.reloaded = True
        import pyFAI.decorators
        pyFAI.decorators.depreclog.setLevel(logging.ERROR)
        return cls.pyFAI

    @classmethod
    def forceBuild(cls, remove_first=True):
        """
        Force the recompilation of pyFAI

        Nonesense, kept for legacy reasons
        """
        return

    @classmethod
    def get_options(cls):
        """
        Parse the command line to analyse options ... returns options
        """
        if cls.options is None:
            parser = ArgumentParser(usage="Tests for %s" % cls.name)
            parser.add_argument("-d", "--debug", dest="debug", help="run in debugging mode",
                                default=False, action="store_true")
            parser.add_argument("-i", "--info", dest="info", help="run in more verbose mode ",
                                default=False, action="store_true")
            parser.add_argument("-f", "--force", dest="force", help="force the build of the library",
                                default=False, action="store_true")
            parser.add_argument("-r", "--really-force", dest="remove",
                                help="remove existing build and force the build of the library",
                                default=False, action="store_true")
            parser.add_argument(dest="args", type=str, nargs='*')
            cls.options = parser.parse_args([])
        return cls.options

    @classmethod
    def get_logger(cls, filename=__file__):
        """
        small helper function that initialized the logger and returns it
        """
        _dirname, basename = os.path.split(os.path.abspath(filename))
        basename = os.path.splitext(basename)[0]
        level = logging.root.level
        mylogger = logging.getLogger(basename)
        logger.setLevel(level)
        mylogger.setLevel(level)
        mylogger.debug("tests loaded from file: %s", basename)
        return mylogger

    @classmethod
    def script_path(cls, script):
        """
        Returns the path of the executable and the associated environment

        In Windows, it checks availability of script using .py .bat, and .exe
        file extensions.
        """
        if (sys.platform == "win32"):
            available_extensions = [".py", ".bat", ".exe"]
        else:
            available_extensions = [""]

        env = dict((str(k), str(v)) for k, v in os.environ.items())
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        paths = os.environ.get("PATH", "").split(os.pathsep)
        if cls.script_dir is not None:
            paths.insert(0, cls.script_dir)

        for base in paths:
            # clean up extra quotes from paths
            if base.startswith('"') and base.endswith('"'):
                base = base[1:-1]
            for file_extension in available_extensions:
                script_path = os.path.join(base, script + file_extension)
                print(script_path)
                if os.path.exists(script_path):
                    # script found
                    return script_path, env
        # script not found
        logger.warning("Script '%s' not found in paths: %s", script, ":".join(paths))
        script_path = script
        return script_path, env


def Rwp(obt, ref, comment="Rwp"):
    """          ___________________________
    Calculate  \/     4 ( obt - ref)²
               V Sum( --------------- )
                        (obt + ref)²

    This is done for symmetry reason between obt and ref

    :param obt: obtained data
    :type obt: 2-list of array of the same size
    :param obt: reference data
    :type obt: 2-list of array of the same size
    :return:  Rwp value, lineary interpolated
    """
    ref0, ref1 = ref
    obt0, obt1 = obt
    big0 = numpy.concatenate((obt0, ref0))
    big0.sort()
    big0 = numpy.unique(big0)
    big_ref = numpy.interp(big0, ref0, ref1, 0.0, 0.0)
    big_obt = numpy.interp(big0, obt0, obt1, 0.0, 0.0)
    big_mean = (big_ref + big_obt) / 2.0
    big_delta = (big_ref - big_obt)
    non_null = abs(big_mean) > 1e-10
    return numpy.sqrt(((big_delta[non_null]) ** 2 / ((big_mean[non_null]) ** 2)).sum())


def recursive_delete(dirname):
    """
    Delete everything reachable from the directory named in "top",
    assuming there are no symbolic links.
    CAUTION:  This is dangerous!  For example, if top == '/', it
    could delete all your disk files.

    :param dirname: top directory to delete
    :type dirname: string
    """
    if not os.path.isdir(dirname):
        return
    for root, dirs, files in os.walk(dirname, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(dirname)

getLogger = UtilsTest.get_logger


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


class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, testcase_method=None, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()

        if testcase_method:
            suite.addTest(testcase_klass(testcase_method, param=param))
        else:
            for name in testnames:
                suite.addTest(testcase_klass(name, param=param))
        return suite
