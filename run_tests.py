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
__date__ = "26/05/2016"
__license__ = "MIT"

import distutils.util
import logging
import os
import shutil
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


logging.basicConfig(level=logging.WARNING)
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
    logger.warning("h5py missing: %s" % error)
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
    logger.debug("Getting project name in %s" % root_dir)
    p = subprocess.Popen([sys.executable, "setup.py", "--name"],
                         shell=False, cwd=root_dir, stdout=subprocess.PIPE)
    name, stderr_data = p.communicate()
    logger.debug("subprocess ended with rc= %s" % p.returncode)
    return name.split()[-1].decode('ascii')


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = get_project_name(PROJECT_DIR)
logger.info('Project name: %s' % PROJECT_NAME)


def _copy(infile, outfile):
    "link or copy file according to the OS. Nota those are HARD_LINKS"
    if "link" in dir(os):
        os.link(infile, outfile)
    else:
        shutil.copy(infile, outfile)


def _copy_files(source, dest, extn):
    """
    copy all files with a given extension from source to destination
    """
    if not os.path.isdir(dest):
        os.makedirs(dest)
    full_src = os.path.join(PROJECT_DIR, source)
    for clf in os.listdir(full_src):
        if clf.endswith(extn) and clf not in os.listdir(dest):
            _copy(os.path.join(full_src, clf), os.path.join(dest, clf))


class TestResult(unittest.TestResult):
    logger = logging.getLogger("memProf")
    logger.setLevel(logging.DEBUG)
    logger.handlers.append(logging.FileHandler("profile.log"))

    def startTest(self, test):
        if resource:
            self.__mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.__time_start = time.time()
        unittest.TestResult.startTest(self, test)

    def stopTest(self, test):
        unittest.TestResult.stopTest(self, test)
        if resource:
            memusage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - self.__mem_start) * 0.001
        else:
            memusage = 0
        self.logger.info("Time: %.3fs \t RAM: %.3f Mb\t%s" % (
                        time.time() - self.__time_start, memusage, test.id()))


if sys.hexversion < 34013184:  # 2.7
    class ProfileTestRunner(unittest.TextTestRunner):
        def _makeResult(self):
            return TestResult()
else:
    class ProfileTestRunner(unittest.TextTestRunner):
        def _makeResult(self):
            return TestResult(stream=sys.stderr, descriptions=True, verbosity=1)


def report_rst(cov, package="fabio", version="0.0.0", base=""):
    """
    Generate a report of test coverage in RST (for Sphinx inclusion)

    @param cov: test coverage instance
    @return: RST string
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
    res.append("Measured on *%s* version %s, %s" % (package, version, time.strftime("%d/%m/%Y")))
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

        res.append('   "%s", "%s", "%s", "%.1f %%"' % (name, sum_lines, sum_hits, cover))
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

    logger.warning("Building %s to %s" % (name, home))
    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=root_dir)
    logger.debug("subprocess ended with rc= %s" % p.wait())

    _copy_files("openCL", os.path.join(home, PROJECT_NAME, "openCL"), ".cl")
    _copy_files("gui", os.path.join(home, PROJECT_NAME, "gui"), ".ui")
    _copy_files("calibration", os.path.join(home, PROJECT_NAME, "calibration"), ".D")

    return home


try:
    from argparse import ArgumentParser
except ImportError:
    from third_party.argparse import ArgumentParser
parser = ArgumentParser(description='Run the tests.')

parser.add_argument("-i", "--insource",
                    action="store_true", dest="insource", default=False,
                    help="Use the build source and not the installed version")
parser.add_argument("-c", "--coverage", dest="coverage",
                    action="store_true", default=False,
                    help="Report code coverage (requires 'coverage' and 'lxml' module)")
parser.add_argument("-m", "--memprofile", dest="memprofile",
                    action="store_true", default=False,
                    help="Report memory profiling")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="Increase verbosity")
default_test_name = "%s.test.suite" % PROJECT_NAME
parser.add_argument("test_name", nargs='*',
                    default=(default_test_name,),
                    help="Test names to run (Default: %s)" % default_test_name)
options = parser.parse_args()
sys.argv = [sys.argv[0]]


if options.verbose == 1:
    logging.root.setLevel(logging.INFO)
    logger.info("Set log level: INFO")
elif options.verbose > 1:
    logging.root.setLevel(logging.DEBUG)
    logger.info("Set log level: DEBUG")


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
    logger.info("Patched sys.path, removed: '%s'" % removed_from_sys_path)


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
    logger.warning("Patched sys.path, added: '%s'" % build_dir)
    module = importer(PROJECT_NAME)


PROJECT_VERSION = getattr(module, 'version', '')
PROJECT_PATH = module.__path__[0]


# Run the tests
if options.memprofile:
    runner = ProfileTestRunner()
else:
    runner = unittest.TextTestRunner()

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


if runner.run(test_suite).wasSuccessful():
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
