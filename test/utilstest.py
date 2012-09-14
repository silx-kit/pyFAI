#!/usr/bin/env python
# coding: utf8
#
#    Project: pyFAI tests class utilities
#             http://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id:$"
#
#    Copyright (C) 2010 European Synchrotron Radiation Facility
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

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__data__ = "2011-10-17"
import os, imp, sys, subprocess, threading
import distutils.util
import logging
import urllib2
import bz2
import gzip
import numpy
logger = logging.getLogger("pyFAI.utilstest")

class UtilsTest(object):
    """
    Static class providing useful stuff for preparing tests.
    """
    timeout = 60        #timeout in seconds for downloading images
    url_base = "http://forge.epn-campus.eu/attachments/download"
    test_home = os.path.dirname(__file__)
    name = "pyFAI"
    image_home = os.path.join(test_home, "testimages")
    if not os.path.isdir(image_home):
        os.makedirs(image_home)
    platform = distutils.util.get_platform()
    architecture = "lib.%s-%i.%i" % (platform,
                                    sys.version_info[0], sys.version_info[1])
    pyFAI_home = os.path.join(os.path.dirname(test_home),
                                        "build", architecture)
    logger.info("pyFAI Home is: " + pyFAI_home)
    if "pyFAI" in sys.modules:
        logger.info("pyFAI module was already loaded from  %s" % sys.modules["pyFAI"])
        pyFAI = None
        sys.modules.pop("pyFAI")
    if not os.path.isdir(pyFAI_home):
        logger.warning("Building pyFAI to %s" % pyFAI_home)
        p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=os.path.dirname(test_home))
        logger.info("subprocess ended with rc= %s" % p.wait())
    opencl = os.path.join(os.path.dirname(test_home), "openCL")
    for clf in os.listdir(opencl):
        if clf.endswith(".cl") and clf not in os.listdir(os.path.join(pyFAI_home, "pyFAI")):
            print "I wanted to link",os.path.join(opencl, clf), os.path.join(pyFAI_home, "pyFAI", clf)
            import shutil
            shutil.copy(os.path.join(opencl, clf), os.path.join(pyFAI_home, "pyFAI", clf))
    pyFAI = imp.load_module(*((name,) + imp.find_module(name, [pyFAI_home])))
    sys.modules[name] = pyFAI
    logger.info("pyFAI loaded from %s" % pyFAI.__file__)


    @classmethod
    def forceBuild(cls):
        """
        force the recompilation of pyFAI
        """
        logger.info("Building pyFAI to %s" % cls.pyFAI_home)
        if "pyFAI" in sys.modules:
            logger.info("pyFAI module was already loaded from  %s" % sys.modules["pyFAI"])
            pyFAI = None
            sys.modules.pop("pyFAI")
        recursive_delete(cls.pyFAI_home)
        p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=os.path.dirname(cls.test_home))
        logger.info("subprocess ended with rc= %s" % p.wait())
        pyFAI = imp.load_module(*((cls.name,) + imp.find_module(cls.name, [cls.pyFAI_home])))
        sys.modules[cls.name] = pyFAI
        logger.info("pyFAI loaded from %s" % pyFAI.__file__)




    @classmethod
    def timeoutDuringDownload(cls):
            """
            Function called after a timeout in the download part ...
            just raise an Exception.
            """
            raise RuntimeError("""Could not automatically download test images!
If you are behind a firewall, please set the environment variable http_proxy.
Otherwise please try to download the images manually from
""" + cls.url_base)


    @classmethod
    def getimage(cls, imagename):
        """
        Downloads the requested image from Forge.EPN-campus.eu
        @param: name of the image.
        For the RedMine forge, the filename contains a directory name that is removed
        @return: full path of the locally saved file
        """
        baseimage = os.path.basename(imagename)
        logger.info("UtilsTest.getimage('%s')" % baseimage)
        fullimagename = os.path.join(cls.image_home, baseimage)
        if not os.path.isfile(fullimagename):
            logger.info("Trying to download image %s, timeout set to %ss"
                          % (imagename, cls.timeout))
            if "http_proxy" in os.environ:
                dictProxies = {'http': os.environ["http_proxy"]}
                proxy_handler = urllib2.ProxyHandler(dictProxies)
                opener = urllib2.build_opener(proxy_handler).open
            else:
                opener = urllib2.urlopen

#           Nota: since python2.6 there is a timeout in the urllib2
            timer = threading.Timer(cls.timeout + 1, cls.timeoutDuringDownload)
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
                please set the environment variable http_proxy.\n \
                Otherwise please try to download the images manually from \n \
                %s" % (cls.url_base, imagename))

        return fullimagename


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
    big0 = (list(obt0) + list(ref0))
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

def getLogger(filename=__file__):
    """
    small helper function that initialized the logger and returns it
    """
    dirname, basename = os.path.split(os.path.abspath(filename))
    basename = os.path.splitext(basename)[0]
    force_build = False
    level = logging.WARN
    for opts in sys.argv[1:]:
        if opts in ["-d", "--debug"]:
            level = logging.DEBUG
#            sys.argv.pop(sys.argv.index(opts))
        elif opts in ["-i", "--info"]:
            level = logging.INFO
#            sys.argv.pop(sys.argv.index(opts))
        elif opts in ["-f", "--force"]:
            force_build = True
#            sys.argv.pop(sys.argv.index(opts))
    logger = logging.getLogger(basename)
    logger.setLevel(level)
    logger.debug("tests loaded from file: %s" % basename)
    if force_build:
        UtilsTest.forceBuild()
    return logger
