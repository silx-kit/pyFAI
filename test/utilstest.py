#!/usr/bin/env python
# coding: utf-8
#
#    Project: pyFAI tests class utilities
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2010-2014 European Synchrotron Radiation Facility
#                       Grenoble, France
#
#    Principal authors: Jérôme KIEFFER (jerome.kieffer@esrf.fr)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, division, absolute_import

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2014-11-11"


import os
import imp
import sys
import subprocess
import threading
import distutils.util
import logging
try: #Python3
    from urllib.request import urlopen, ProxyHandler, build_opener
except ImportError: #Python2
    from urllib2 import urlopen, ProxyHandler, build_opener
#import urllib2
import numpy
import shutil
import json
import tempfile
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("pyFAI.utilstest")

TEST_HOME = os.path.dirname(os.path.abspath(__file__))
IN_SOURCES = "pyFAI-src" in os.listdir(os.path.dirname(TEST_HOME))

if IN_SOURCES:
    os.environ["PYFAI_DATA"] = os.path.dirname(TEST_HOME)


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
    timeout = 60        # timeout in seconds for downloading images
    url_base = "http://forge.epn-campus.eu/attachments/download"

    # Nota https crashes with error 501 under windows.
#    url_base = "https://forge.epn-campus.eu/attachments/download"
    sem = threading.Semaphore()
    recompiled = False
    reloaded = False
    name = "pyFAI"
    if IN_SOURCES:
        image_home = os.path.join(TEST_HOME, "testimages")
        if not os.path.isdir(image_home):
            os.makedirs(image_home)
        testimages = os.path.join(TEST_HOME, "all_testimages.json")
        if os.path.exists(testimages):
            with open(testimages) as f:
                ALL_DOWNLOADED_FILES = set(json.load(f))
        else:
            ALL_DOWNLOADED_FILES = set()
        platform = distutils.util.get_platform()
        architecture = "lib.%s-%i.%i" % (platform,
                                         sys.version_info[0], sys.version_info[1])
        if os.environ.get("BUILDPYTHONPATH"):
            pyFAI_home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
        else:
            pyFAI_home = os.path.join(os.path.dirname(TEST_HOME),
                                      "build", architecture)
        logger.info("pyFAI Home is: " + pyFAI_home)
        if "pyFAI" in sys.modules:
            logger.info("pyFAI module was already loaded from  %s" % sys.modules["pyFAI"])
            pyFAI = None
            sys.modules.pop("pyFAI")
            for key in sys.modules.copy():
                if key.startswith("pyFAI."):
                    sys.modules.pop(key)

        if not os.path.isdir(pyFAI_home):
            with sem:
                if not os.path.isdir(pyFAI_home):
                    logger.warning("Building pyFAI to %s" % pyFAI_home)
                    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                                         shell=False, cwd=os.path.dirname(TEST_HOME))
                    logger.info("subprocess ended with rc= %s" % p.wait())
                    recompiled = True
        logger.info("Loading pyFAI")
        try:
            pyFAI = imp.load_module(*((name,) + imp.find_module(name, [pyFAI_home])))
        except Exception as error:
            logger.warning("Unable to loading pyFAI %s" % error)
            if "-r" not in sys.argv:
                logger.warning("Remove build and start from scratch %s" % error)
                sys.argv.append("-r")
    else:
        image_home = os.path.join(tempfile.gettempdir(),"pyFAI_testimages_%s"%os.getlogin())
        if not os.path.exists(image_home):
            os.makedirs(image_home)
        testimages = os.path.join(image_home, "all_testimages.json")
        if os.path.exists(testimages):
            with open(testimages) as f:
                ALL_DOWNLOADED_FILES = set(json.load(f))
        else:
            ALL_DOWNLOADED_FILES = set()

    @classmethod
    def deep_reload(cls):
        if not IN_SOURCES:
            cls.pyFAI = __import__(cls.name)
            return cls.pyFAI
        if cls.reloaded:
            return cls.pyFAI
        logger.info("Loading pyFAI")
        cls.pyFAI = None
        pyFAI = None
        sys.path.insert(0, cls.pyFAI_home)
        for key in sys.modules.copy():
            if key.startswith("pyFAI"):
                sys.modules.pop(key)
        cls.pyFAI = __import__(cls.name)
        logger.info("pyFAI loaded from %s" % cls.pyFAI.__file__)
        sys.modules[cls.name] = cls.pyFAI
        cls.reloaded = True
        return cls.pyFAI

    @classmethod
    def forceBuild(cls, remove_first=True):
        """
        force the recompilation of pyFAI
        """
        if not IN_SOURCES:
            return
        if not cls.recompiled:
            with cls.sem:
                if not cls.recompiled:
                    logger.info("Building pyFAI to %s" % cls.pyFAI_home)
                    if "pyFAI" in sys.modules:
                        logger.info("pyFAI module was already loaded from  %s" % sys.modules["pyFAI"])
                        cls.pyFAI = None
                        sys.modules.pop("pyFAI")
                    if remove_first:
                        recursive_delete(cls.pyFAI_home)
                    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                                         shell=False, cwd=os.path.dirname(TEST_HOME))
                    logger.info("subprocess ended with rc= %s" % p.wait())
                    cls.pyFAI = cls.deep_reload()
                    cls.recompiled = True

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
        if imagename not in cls.ALL_DOWNLOADED_FILES:
            cls.ALL_DOWNLOADED_FILES.add(imagename)
            with open(cls.testimages, "w") as fp:
                json.dump(list(cls.ALL_DOWNLOADED_FILES), fp)

        baseimage = os.path.basename(imagename)
        logger.info("UtilsTest.getimage('%s')" % baseimage)
        fullimagename = os.path.abspath(os.path.join(cls.image_home, baseimage))
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
                opener = urllib2.build_opener(proxy_handler).open
            else:
                opener = urlopen

#>>> url = 'http://docs.python.org/library/'
#>>> parts = urlparse(url)
#>>> parts = parts._replace(path='/3.0'+parts.path)
#>>> page = urlopen(parts.geturl())

#           Nota: since python2.6 there is a timeout in the urllib2
            timer = threading.Timer(cls.timeout + 1, cls.timeoutDuringDownload, args=[imagename])
            timer.start()
            logger.info("wget %s/%s" % (cls.url_base, imagename))
            if sys.version > (2, 6):
                data = opener("%s/%s" % (cls.url_base, imagename),
                              data=None, timeout=cls.timeout).read()
            else:
                data = opener("%s/%s" % (cls.url_base, imagename),
                              data=None).read()
            timer.cancel()
            logger.info("Image %s successfully downloaded." % baseimage)

            try:
                open(fullimagename, "wb").write(data)
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
                from pyFAI.argparse import ArgumentParser

            parser = ArgumentParser(usage="Tests for PyFAI")
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
            if IN_SOURCES:
                cls.options = parser.parse_args()
            else:
                cls.options = parser.parse_args([])
        return cls.options

    @classmethod
    def get_logger(cls, filename=__file__):
        """
        small helper function that initialized the logger and returns it
        """
        options = cls.get_options()
        dirname, basename = os.path.split(os.path.abspath(filename))
        basename = os.path.splitext(basename)[0]
        force_build = False
        force_remove = False
        level = logging.WARN
        if options.debug:
            level = logging.DEBUG
        elif options.info:
            level = logging.INFO
        if options.force:
            force_build = True
        if options.remove:
            force_remove = True
            force_build = True
        mylogger = logging.getLogger(basename)
        logger.setLevel(level)
        mylogger.setLevel(level)
        mylogger.debug("tests loaded from file: %s" % basename)
        if force_build:
            UtilsTest.forceBuild(force_remove)
        return mylogger


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


def recursive_delete(strDirname):
    """
    Delete everything reachable from the directory named in "top",
    assuming there are no symbolic links.
    CAUTION:  This is dangerous!  For example, if top == '/', it
    could delete all your disk files.
    @param strDirname: top directory to delete
    @type strDirname: string
    """
    for root, dirs, files in os.walk(strDirname, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(strDirname)

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

