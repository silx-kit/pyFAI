#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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
from __future__ import print_function, division, absolute_import, with_statement
__doc__ = "bunch of utility function/static classes to handle testing environment"
__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/04/2016"

PACKAGE = "pyFAI"
DATA_KEY = "PYFAI_DATA"

if __name__ == "__main__":
    __name__ = "pyFAI.test"

import os
import imp
import sys
import getpass
import subprocess
import threading
import distutils.util
import unittest
import logging
try:  # Python3
    from urllib.request import urlopen, ProxyHandler, build_opener
except ImportError:  # Python2
    from urllib2 import urlopen, ProxyHandler, build_opener
# import urllib2
import numpy
import shutil
import json
import tempfile
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("%s.utilstest" % PACKAGE)

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
    sem = threading.Semaphore()
    recompiled = False
    reloaded = False
    name = PACKAGE
    script_dir = None
    try:
        pyFAI = __import__("%s.directories" % name)
    except Exception as error:
        logger.warning("Unable to loading %s %s" % (name, error))
        image_home = None
    else:
        image_home = pyFAI.directories.testimages
        pyFAI.depreclog.setLevel(logging.ERROR)

    if image_home is None:
        image_home = os.path.join(tempfile.gettempdir(), "%s_testimages_%s" % (name, getpass.getuser()))
        if not os.path.exists(image_home):
            os.makedirs(image_home)

    testimages = os.path.join(image_home, "all_testimages.json")
    if os.path.exists(testimages):
        with open(testimages) as f:
            ALL_DOWNLOADED_FILES = set(json.load(f))
    else:
        ALL_DOWNLOADED_FILES = set()

    tempdir = tempfile.mkdtemp("_" + getpass.getuser(), name + "_")

    @classmethod
    def clean_up(cls):
        recursive_delete(cls.tempdir)

    @classmethod
    def deep_reload(cls):
        cls.pyFAI = __import__(cls.name)
        logger.info("%s loaded from %s" % (cls.name, cls.pyFAI.__file__))
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
    def timeoutDuringDownload(cls, imagename=None):
            """
            Function called after a timeout in the download part ...
            just raise an Exception.
            """
            if imagename is None:
                imagename = "2252/testimages.tar.bz2 unzip it "
            raise RuntimeError("Could not automatically \
                download test images!\n \ If you are behind a firewall, \
                please set both environment variable http_proxy and https_proxy.\
                This even works under windows ! \n \
                Otherwise please try to download the images manually from \n %s/%s and put it in in test/testimages." % (cls.url_base, imagename))

    @classmethod
    def getimage(cls, imagename):
        """
        Downloads the requested image from Forge.EPN-campus.eu

        @param: name of the image.
        For the RedMine forge, the filename contains a directory name that is removed
        @return: full path of the locally saved file
        """
        imagename = os.path.basename(imagename)
        if imagename not in cls.ALL_DOWNLOADED_FILES:
            cls.ALL_DOWNLOADED_FILES.add(imagename)
            image_list = list(cls.ALL_DOWNLOADED_FILES)
            image_list.sort()
            try:
                with open(cls.testimages, "w") as fp:
                    json.dump(image_list, fp, indent=4)
            except IOError:
                logger.debug("Unable to save JSON list")
        logger.info("UtilsTest.getimage('%s')" % imagename)
        if not os.path.exists(cls.image_home):
            os.makedirs(cls.image_home)

        fullimagename = os.path.abspath(os.path.join(cls.image_home, imagename))
        if not os.path.isfile(fullimagename):
            logger.info("Trying to download image %s, timeout set to %ss",
                        imagename, cls.timeout)
            dictProxies = {}
            if "http_proxy" in os.environ:
                dictProxies['http'] = os.environ["http_proxy"]
                dictProxies['https'] = os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                dictProxies['https'] = os.environ["https_proxy"]
            if dictProxies:
                proxy_handler = ProxyHandler(dictProxies)
                opener = build_opener(proxy_handler).open
            else:
                opener = urlopen

            logger.info("wget %s/%s" % (cls.url_base, imagename))
            data = opener("%s/%s" % (cls.url_base, imagename),
                          data=None, timeout=cls.timeout).read()
            logger.info("Image %s successfully downloaded." % imagename)

            try:
                with open(fullimagename, "wb") as outfile:
                    outfile.write(data)
            except IOError:
                raise IOError("unable to write downloaded \
                    data to disk at %s" % cls.image_home)

            if not os.path.isfile(fullimagename):
                raise RuntimeError("Could not automatically \
                download test images %s!\n \ If you are behind a firewall, \
                please set both environment variable http_proxy and https_proxy.\
                This even works under windows ! \n \
                Otherwise please try to download the images manually from \n%s/%s" % (imagename, cls.url_base, imagename))

        return fullimagename

    @classmethod
    def download_images(cls, imgs=None):
        """
        Download all images needed for the test/benchmarks

        @param imgs: list of files to download
        """
        if not imgs:
            imgs = cls.ALL_DOWNLOADED_FILES
        for fn in imgs:
            print("Downloading from internet: %s" % fn)
            cls.getimage(fn)

    @classmethod
    def get_options(cls):
        """
        Parse the command line to analyse options ... returns options
        """
        if cls.options is None:
            try:
                from argparse import ArgumentParser
            except:
                from pyFAI.third_party.argparse import ArgumentParser

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
        dirname, basename = os.path.split(os.path.abspath(filename))
        basename = os.path.splitext(basename)[0]
        level = logging.root.level
        mylogger = logging.getLogger(basename)
        logger.setLevel(level)
        mylogger.setLevel(level)
        mylogger.debug("tests loaded from file: %s" % basename)
        return mylogger

    @classmethod
    def script_path(cls, script):
        """
        Return the path of the executable and the associated environment
        """
        if (sys.platform == "win32") and not script.endswith(".py"):
                script += ".py"
        env = dict((str(k), str(v)) for k, v in os.environ.items())
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        paths = os.environ.get("PATH", "").split(os.pathsep)
        if cls.script_dir is not None:
            paths.insert(0, cls.script_dir)
        for i in paths:
            script_path = os.path.join(i, script)
            if os.path.exists(script_path):
                break
        else:
            logger.warning("No scipt %s found in path: %s", script, paths)
        return script_path, env


def Rwp(obt, ref, comment="Rwp"):
    """          ___________________________
    Calculate  \/     4 ( obt - ref)²
               V Sum( --------------- )
                        (obt + ref)²

    This is done for symmetry reason between obt and ref

    @param obt: obtained data
    @type obt: 2-list of array of the same size
    @param obt: reference data
    @type obt: 2-list of array of the same size
    @return:  Rwp value, lineary interpolated
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

    @param dirname: top directory to delete
    @type dirname: string
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
        from pyFAI.gui_utils import pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        im_ref = ax1.imshow(ref)
        plt.colorbar(im_ref)
        ax1.set_title("%s ref" % comment)
        im_obt = ax2.imshow(obt)
        plt.colorbar(im_obt)
        ax2.set_title("%s obt" % comment)
        im_delta = ax3.imshow(delta)
        plt.colorbar(im_delta)
        ax3.set_title("delta")
        imax = delta.argmax()
        x = imax % ref.shape[-1]
        y = imax // ref.shape[-1]
        ax3.plot([x], [y], "o", scalex=False, scaley=False)
        fig.show()
        from pyFAI.utils import input
        input()


def diff_crv(ref, obt, comment=""):
    """
    Highlight the difference in vectors
    """
    assert ref.shape == obt.shape
    delta = abs(obt - ref)
    if delta.max() > 0:
        from pyFAI.gui_utils import pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        im_ref = ax1.plot(ref, label="%s ref" % comment)
        im_obt = ax1.plot(obt, label="%s obt" % comment)
        im_delta = ax2.plot(delta, label="delta")
        fig.show()
        from pyFAI.utils import input
        input()


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
