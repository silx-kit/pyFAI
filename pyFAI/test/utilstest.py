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
__date__ = "12/01/2018"

PACKAGE = "pyFAI"
DATA_KEY = "PYFAI_DATA"

import os
import sys
import threading
import unittest
import logging
import shutil
import contextlib
import tempfile
import getpass


from pyFAI.third_party.argparse import ArgumentParser
from ..third_party import six
from silx.resources import ExternalResources

logger = logging.getLogger(__name__)

TEST_HOME = os.path.dirname(os.path.abspath(__file__))


def copy(infile, outfile):
    "link or copy file according to the OS"
    if "link" in dir(os):
        os.link(infile, outfile)
    else:
        shutil.copy(infile, outfile)


class TestContext(object):
    """
    Class providing useful stuff for preparing tests.
    """
    def __init__(self):
        self.options = None
        self.timeout = 60  # timeout in seconds for downloading images
        # url_base = "http://forge.epn-campus.eu/attachments/download"
        self.url_base = "http://ftp.edna-site.org/pyFAI/testimages"
        self.resources = ExternalResources(PACKAGE,
                                           timeout=self.timeout,
                                           env_key=DATA_KEY,
                                           url_base=self.url_base)
        self.sem = threading.Semaphore()
        self.recompiled = False
        self.reloaded = False
        self.name = PACKAGE
        self.script_dir = None

        self.download_images = self.resources.download_all
        self.getimage = self.resources.getfile
        self.low_mem = bool(os.environ.get("PYFAI_LOW_MEM"))
        self.opencl = bool(os.environ.get("PYFAI_OPENCL", True))

        self._tempdir = None

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

    def get_options(self):
        """
        Parse the command line to analyse options ... returns options
        """
        if self.options is None:
            parser = ArgumentParser(usage="Tests for %s" % self.name)
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
            self.options = parser.parse_args([])
        return self.options

    def script_path(self, script):
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
        if self.script_dir is not None:
            paths.insert(0, self.script_dir)

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


UtilsTest = TestContext()
"""Singleton containing util context of whole the tests"""


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
