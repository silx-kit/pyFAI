#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Run the tests of the project.

This script expects a suite function in <project_package>.test,
which returns a unittest.TestSuite.

Test coverage dependencies: coverage, lxml.
"""

__authors__ = ["Jérôme Kieffer", "Thomas Vincent"]
__date__ = "18/05/2017"
__license__ = "MIT"

import distutils.util
import distutils.dir_util
import logging
import os
import subprocess
import sys
import time
import unittest
if os.name == "posix":
    import resource
else:
    resource = None
try:
    import importlib
except:
    importer = __import__
    old_importer = True
else:
    importer = importlib.import_module
    old_importer = False


class StreamHandlerUnittestReady(logging.StreamHandler):
    """The unittest class TestResult redefine sys.stdout/err to capture
    stdout/err from tests and to display them only when a test fail.

    This class allow to use unittest stdout-capture by using the last sys.stdout
    and not a cached one.
    """

    def emit(self, record):
        """
        :type record: logging.LogRecord
        """
        self.stream = sys.stderr
        super(StreamHandlerUnittestReady, self).emit(record)

    def flush(self):
        pass

# Same as basicConfig with a custom handler but portable Python 2 and 3
root = logging.getLogger()
root.addHandler(StreamHandlerUnittestReady())
root.setLevel(logging.WARNING)

logger = logging.getLogger("run_tests")
logger.setLevel(logging.WARNING)

logger.info("Python %s %s", sys.version, tuple.__itemsize__ * 8)

try:
    import numpy
except:
    logger.warning("numpy missing")
else:
    print("numpy %s from %s" % (numpy.version.version, numpy.__path__))
try:
    import scipy
except:
    logger.warning("Scipy missing")
else:
    print("Scipy %s from %s" % (scipy.version.version, scipy.__path__))

try:
    import fabio
except:
    logger.warning("FabIO missing")
else:
    print("FabIO %s" % fabio.version)

try:
    import h5py
except Exception as error:
    logger.warning("h5py missing: %s", error)
else:
    print("h5py %s" % h5py.version.version)

try:
    import Cython
except:
    print("Cython missing")
else:
    print("Cython %s" % Cython.__version__)


def get_project_name(root_dir):
    """Retrieve project name by running python setup.py --name in root_dir.

    :param str root_dir: Directory where to run the command.
    :return: The name of the project stored in root_dir
    """
    logger.debug("Getting project name in %s", root_dir)
    p = subprocess.Popen([sys.executable, "setup.py", "--name"],
                         shell=False, cwd=root_dir, stdout=subprocess.PIPE)
    name, _stderr_data = p.communicate()
    logger.debug("subprocess ended with rc= %s", p.returncode)
    return name.split()[-1].decode('ascii')


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = get_project_name(PROJECT_DIR)
logger.info("Project name: %s", PROJECT_NAME)


class ProfileTextTestResult(unittest.TextTestRunner.resultclass):

    def __init__(self, *arg, **kwarg):
        super(ProfileTextTestResult, self).__init__(*arg, **kwarg)
        self.logger = logging.getLogger("memProf")
        self.logger.setLevel(min(logging.INFO, logging.root.level))
        self.logger.handlers.append(logging.FileHandler("memprofile.log"))

    def startTest(self, test):
        if resource:
            self.__mem_start = \
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.logger.debug("Start %s", test.id())
        self.__time_start = time.time()
        super(ProfileTextTestResult, self).startTest(test)

    def stopTest(self, test):
        super(ProfileTextTestResult, self).stopTest(test)
        # see issue 311. For other platform, get size of ru_maxrss in "man getrusage"
        if sys.platform == "darwin":
            ratio = 1e-6
        else:
            ratio = 1e-3
        if resource:
            memusage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss -
                        self.__mem_start) * ratio
        else:
            memusage = 0
        self.logger.info("Time: %.3fs \t RAM: %.3f Mb\t%s",
                         time.time() - self.__time_start, memusage, test.id())


def report_rst(cov, package="fabio", version="0.0.0", base=""):
    """
    Generate a report of test coverage in RST (for Sphinx inclusion)

    :param cov: test coverage instance
    :param str package: Name of the package
    :param str base: base directory of modules to include in the report
    :return: RST string
    """
    import tempfile
    fd, fn = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    cov.xml_report(outfile=fn)

    from lxml import etree
    xml = etree.parse(fn)
    classes = xml.xpath("//class")

    line0 = "Test coverage report for %s" % package
    res = [line0, "=" * len(line0), ""]
    res.append("Measured on *%s* version %s, %s" %
               (package, version, time.strftime("%d/%m/%Y")))
    res += ["",
            ".. csv-table:: Test suite coverage",
            '   :header: "Name", "Stmts", "Exec", "Cover"',
            '   :widths: 35, 8, 8, 8',
            '']
    tot_sum_lines = 0
    tot_sum_hits = 0

    for cl in classes:
        name = cl.get("name")
        fname = cl.get("filename")
        if not os.path.abspath(fname).startswith(base):
            continue
        lines = cl.find("lines").getchildren()
        hits = [int(i.get("hits")) for i in lines]

        sum_hits = sum(hits)
        sum_lines = len(lines)

        cover = 100.0 * sum_hits / sum_lines if sum_lines else 0

        if base:
            name = os.path.relpath(fname, base)

        res.append('   "%s", "%s", "%s", "%.1f %%"' %
                   (name, sum_lines, sum_hits, cover))
        tot_sum_lines += sum_lines
        tot_sum_hits += sum_hits
    res.append("")
    res.append('   "%s total", "%s", "%s", "%.1f %%"' %
               (package, tot_sum_lines, tot_sum_hits,
                100.0 * tot_sum_hits / tot_sum_lines if tot_sum_lines else 0))
    res.append("")
    return os.linesep.join(res)


def build_project(name, root_dir):
    """Run python setup.py build for the project.
    and copy data files to run the tests

    Build directory can be modified by environment variables.

    :param str name: Name of the project.
    :param str root_dir: Root directory of the project
    :return: The path to the directory were build was performed
    """
    platform = distutils.util.get_platform()
    architecture = "lib.%s-%i.%i" % (platform,
                                     sys.version_info[0], sys.version_info[1])

    if os.environ.get("PYBUILD_NAME") == name:
        # we are in the debian packaging way
        home = os.environ.get("PYTHONPATH", "").split(os.pathsep)[-1]
    elif os.environ.get("BUILDPYTHONPATH"):
        home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
    else:
        home = os.path.join(root_dir, "build", architecture)

    logger.warning("Building %s to %s", name, home)
    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=root_dir)
    logger.debug("subprocess ended with rc= %s", p.wait())

    distutils.dir_util.copy_tree("pyFAI/resources", os.path.join(home, PROJECT_NAME, "resources"), update=1)

    return home


try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

epilog = """Environment variables: None"""
# WITH_QT_TEST=False to disable graphical tests,
# SILX_OPENCL=False to disable OpenCL tests.
# SILX_TEST_LOW_MEM=True to disable tests taking large amount of memory
# GPU=False to disable the use of a GPU with OpenCL test
# """
parser = ArgumentParser(description='Run the tests.',
                        epilog=epilog)

parser.add_argument("-i", "--insource",
                    action="store_true", dest="insource", default=False,
                    help="Use the build source and not the installed version")
parser.add_argument("-c", "--coverage", dest="coverage",
                    action="store_true", default=False,
                    help=("Report code coverage" +
                          "(requires 'coverage' and 'lxml' module)"))
parser.add_argument("-m", "--memprofile", dest="memprofile",
                    action="store_true", default=False,
                    help="Report memory profiling")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="Increase verbosity. Option -v prints additional " +
                         "INFO messages. Use -vv for full verbosity, " +
                         "including debug messages and test help strings.")
parser.add_argument("-n", "--noisy", default=True,
                    action="store_false", dest="display_buffer",
                    help="Display all warnings from the system")

# parser.add_argument("-x", "--no-gui", dest="gui", default=True,
#                    action="store_false",
#                    help="Disable the test of the graphical use interface")
# parser.add_argument("-o", "--no-opencl", dest="opencl", default=True,
#                    action="store_false",
#                    help="Disable the test of the OpenCL part")
# parser.add_argument("-l", "--low-mem", dest="low_mem", default=False,
#                    action="store_true",
#                    help="Disable test with large memory consumption (>100Mbyte")
# parser.add_argument("--qt-binding", dest="qt_binding", default=None,
#                    help="Force using a Qt binding, from 'PyQt4', 'PyQt5', or 'PySide'")

default_test_name = "%s.test.suite" % PROJECT_NAME
parser.add_argument("test_name", nargs='*',
                    default=(default_test_name,),
                    help="Test names to run (Default: %s)" % default_test_name)
options = parser.parse_args()
sys.argv = [sys.argv[0]]


test_verbosity = 1
if options.verbose == 1:
    logging.root.setLevel(logging.INFO)
    logger.info("Set log level: INFO")
elif options.verbose > 1:
    logging.root.setLevel(logging.DEBUG)
    logger.info("Set log level: DEBUG")
    test_verbosity = 2

# if not options.gui:
#    os.environ["WITH_QT_TEST"] = "False"
#
# if not options.opencl:
#    os.environ["SILX_OPENCL"] = "False"
#
# if options.low_mem:isy
#    os.environ["SILX_TEST_LOW_MEM"] = "True"


if options.coverage:
    logger.info("Running test-coverage")
    import coverage
    try:
        cov = coverage.Coverage(omit=["*test*", "*third_party*", "*/setup.py"])
    except AttributeError:
        cov = coverage.coverage(omit=["*test*", "*third_party*", "*/setup.py"])
    cov.start()


# Prevent importing from source directory
if (os.path.dirname(os.path.abspath(__file__)) ==
        os.path.abspath(sys.path[0])):
    removed_from_sys_path = sys.path.pop(0)
    logger.info("Patched sys.path, removed: '%s'", removed_from_sys_path)


# import module
if not options.insource:
    try:
        module = importer(PROJECT_NAME)
    except:
        logger.warning(
            "%s missing, using built (i.e. not installed) version",
            PROJECT_NAME)
        options.insource = True

if options.insource:
    build_dir = build_project(PROJECT_NAME, PROJECT_DIR)

    sys.path.insert(0, build_dir)
    logger.warning("Patched sys.path, added: '%s'", build_dir)
    module = importer(PROJECT_NAME)


PROJECT_VERSION = getattr(module, 'version', '')
PROJECT_PATH = module.__path__[0]


# Run the tests
runnerArgs = {}
runnerArgs["verbosity"] = test_verbosity
runnerArgs["buffer"] = options.display_buffer
if options.memprofile:
    runnerArgs["resultclass"] = ProfileTextTestResult
runner = unittest.TextTestRunner(**runnerArgs)

logger.warning("Test %s %s from %s",
               PROJECT_NAME, PROJECT_VERSION, PROJECT_PATH)

test_module_name = PROJECT_NAME + '.test'
logger.info('Import %s', test_module_name)
test_module = importer(test_module_name)
utilstest = importer(test_module_name + ".utilstest")
if old_importer:
    test_module = getattr(test_module, "test")
    print(dir(test_module))
    utilstest = getattr(test_module, "utilstest")
UtilsTest = getattr(utilstest, "UtilsTest")
UtilsTest.image_home = os.path.join(PROJECT_DIR, 'testimages')
UtilsTest.testimages = os.path.join(UtilsTest.image_home, "all_testimages.json")
UtilsTest.script_dir = os.path.join(PROJECT_DIR, "scripts")

test_suite = unittest.TestSuite()

if not options.test_name:
    # Do not use test loader to avoid cryptic exception
    # when an error occur during import
    project_test_suite = getattr(test_module, 'suite')
    test_suite.addTest(project_test_suite())
else:
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromNames(options.test_name))


result = runner.run(test_suite)
for test, reason in result.skipped:
    logger.warning('Skipped %s (%s): %s',
                   test.id(), test.shortDescription() or '', reason)

if result.wasSuccessful():
    logger.info("Test suite succeeded")
    exit_status = 0
else:
    logger.warning("Test suite failed")
    exit_status = 1


if options.coverage:
    cov.stop()
    cov.save()
    with open("coverage.rst", "w") as fn:
        fn.write(report_rst(cov, PROJECT_NAME, PROJECT_VERSION, PROJECT_PATH))
    print(cov.report())

sys.exit(exit_status)
