#!/usr/bin/env python3
# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2026-2026 European Synchrotron Radiation Facility
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
"""Run the tests of the project with pytest, in parallel.

Same job as `run_tests.py`, same options, but the suite is collected by pytest
and spread over several worker processes by pytest-xdist, which divides the
wall-clock time by about four:

    python run_pytest.py                     # whole suite, one worker per core
    python run_pytest.py -n 8                # ... on 8 workers
    python run_pytest.py -o -x               # without OpenCL, without the GUI
    python run_pytest.py pyFAI.test.test_csr # one module,
    python run_pytest.py pyFAI.test.test_csr.TestCSR.test_2d_splitbbox  # or one test

The test names are the ones of `run_tests.py` (dotted unittest paths); they are
translated into the `module::Class::method` form pytest expects. Anything after
`--` is handed over to pytest untouched:

    python run_pytest.py -- -k distortion --durations=10

Unlike `run_tests.py`, which builds its suite from the `suite()` functions of
`pyFAI.test.test_all`, pytest discovers every `test_*.py` of the package: a
module missing from `test_all.py` is run here, and not there.

Dependencies: pytest and pytest-xdist (the `test` extra), plus pytest-cov for
the --coverage option.
"""

__authors__ = ["Jérôme Kieffer"]
__date__ = "16/09/2026"
__license__ = "MIT"

import importlib
import logging
import os
import platform
import sys
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path
from xml.etree import ElementTree

logging.basicConfig()
logger = logging.getLogger("run_pytest")
logger.setLevel(logging.WARNING)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
from bootstrap import get_project_name, build_project  # noqa: E402
PROJECT_NAME = get_project_name(PROJECT_DIR)

def thread_per_core():
    """return the number of hyperthreads per core"""
    archi = platform.machine()
    smt = 1
    if "ppc" in archi:
        smt = 4
    elif "arm" in archi:
        smt = 1
    elif "x86" in archi:
        smt = 2
    else:
        logger.warning("Unknown CPU architecture %s, Unable to guess SMT level.", archi)
    return smt

THREADS_PER_CORE = thread_per_core()


def available_cores():
    """Number of CPUs this process is allowed to run on

    Not `os.cpu_count()`: on a shared or partitioned machine, the affinity mask
    is the only one of the two which tells the truth.

    :return: size of the affinity mask, or the number of CPUs of the machine
    """
    try:
        return len(os.sched_getaffinity(0))
    except Exception:  # not available on macos nor windows
        return os.cpu_count()


def default_workers(THREADS_PER_CORE):
    """One worker per physical core, each keeping THREADS_PER_CORE threads"""
    return max(available_cores() // THREADS_PER_CORE, 1)


def build_parser():
    """Command line of run_tests.py, plus the options specific to pytest"""
    epilog = """Environment variables:
WITH_QT_TEST=False to disable graphical tests
PYFAI_OPENCL=False to disable OpenCL tests.
PYFAI_LOW_MEM=True to skip all tests >100Mb
WITH_GL_TEST=False to disable tests using OpenGL
QT_QPA_PLATFORM=offscreen to render all GUI tests offscreen

Arguments after `--` are passed to pytest as-is, for instance:
    python run_pytest.py -- -k distortion --durations=10
"""
    parser = ArgumentParser(description=__doc__.splitlines()[0], epilog=epilog)

    # Options of run_tests.py, kept identical (short options included)
    parser.add_argument("test_name", nargs="*", default=[],
                        help="Test names to run (default: the whole test-suite)")
    parser.add_argument("-i", "--installed",
                        action="store_true", dest="installed", default=False,
                        help="Test the installed version instead of building from the source")
    parser.add_argument("-c", "--coverage", dest="coverage",
                        action="store_true", default=False,
                        help="Report code coverage (requires the 'pytest-cov' module)")
    parser.add_argument("-m", "--memprofile", dest="memprofile",
                        action="store_true", default=False,
                        help="Report the time and the memory used by every test "
                             "(written to profile.json)")
    parser.add_argument("-v", "--verbose", default=0,
                        action="count", dest="verbose",
                        help="Increase verbosity. Option -v prints additional "
                             "INFO messages. Use -vv for full verbosity, "
                             "including debug messages and test help strings.")
    parser.add_argument("--qt-binding", dest="qt_binding", default=None,
                        help="Force using a Qt binding, from 'PyQt5', 'PyQt6' or 'PySide6' (default)")
    parser.add_argument("--qt-screen", dest="qt_screen", default="off",
                        help="Force using a Qt to render 'off'-screen (default) or 'on'-screen. "
                             "use 'vnc' to debug")
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
    parser.add_argument("-r", "--random", dest="random", default=False,
                        action="store_true",
                        help="Enable actual random number to be generated. By default, "
                             "stable seed ensures reproducibility of tests")

    # Specific to this runner
    parser.add_argument("-n", "--workers", dest="workers", type=int, default=0,
                        metavar="N",
                        help="Number of worker processes. 0 (default) uses one worker "
                             f"per physical core, i.e. {default_workers(THREADS_PER_CORE)} here; "
                             "1 runs the whole suite in a single worker.")
    parser.add_argument("-t", "--threads", dest="threads", type=int, default=None,
                        metavar="N",
                        help="Number of threads per worker process. Default guesses the SMT level"
                             f", i.e. {THREADS_PER_CORE} here.")
    return parser


def parse_command_line(argv):
    """Parse the command line, `--` separating our options from pytest's

    argparse cannot do the split itself: the positional test names would
    swallow whatever follows.

    :param argv: arguments, without the name of the program
    :return: the parsed options, carrying the pytest arguments in `pytest_args`
    """
    if "--" in argv:
        index = argv.index("--")
        argv, extra = argv[:index], argv[index + 1:]
    else:
        extra = []
    options = build_parser().parse_args(argv)
    options.pytest_args = extra
    return options


def package_relative(filename, package_dir):
    """Path of a source file, relative to the directory of the package

    Coverage names the file `.../site-packages/pyFAI/utils/mathutil.py`, the
    report expects `utils/mathutil.py`.

    :param filename: file name as written in the XML report
    :param package_dir: directory of the package
    :return: the elided name, or None for a file outside of the package
    """
    path = os.path.abspath(filename)
    if path.startswith(package_dir + os.sep):
        return os.path.relpath(path, package_dir)
    # coverage may report a path relative to another root: elide up to the
    # last directory named after the project
    _head, sep, tail = path.rpartition(os.sep + PROJECT_NAME + os.sep)
    return tail if sep else None


def is_test_module(name):
    """The test-suite is not part of what the coverage of the library measures

    Covers the `test` sub-packages, the `test_*.py` modules and the conftest of
    the package.

    :param name: file name, relative to the package directory
    """
    parts = Path(name).parts
    return ("test" in parts
            or parts[-1].startswith("test_")
            or parts[-1] == "conftest.py")


def coverage_rst(xml_file, package_dir, version):
    """Build the coverage report which goes to `doc/source/coverage.rst`

    One row per module of the package, the modules of the test-suite left out,
    and the file names elided up to the package directory.

    :param xml_file: XML report written by coverage.py
    :param package_dir: directory of the package
    :param version: version of the project, quoted in the header
    :return: the report, as a string
    """
    title = f"Test coverage report for {PROJECT_NAME}"
    res = [title,
           "=" * len(title),
           "",
           f"Measured on *{PROJECT_NAME}* version {version}, {time.strftime('%d/%m/%Y')}",
           "",
           ".. csv-table:: Test suite coverage",
           '   :header: "Name", "Stmts", "Exec", "Cover"',
           "   :widths: 35, 8, 8, 8",
           ""]
    total_stmts = total_exec = 0
    for class_ in ElementTree.parse(xml_file).iterfind(".//class"):
        name = package_relative(class_.get("filename"), package_dir)
        if name is None or is_test_module(name):
            continue
        hits = [int(line.get("hits")) for line in class_.iterfind("lines/line")]
        stmts, executed = len(hits), sum(hits)
        cover = 100.0 * executed / stmts if stmts else 0.0
        res.append(f'   "{name}", "{stmts}", "{executed}", "{cover:.1f} %"')
        total_stmts += stmts
        total_exec += executed
    cover = 100.0 * total_exec / total_stmts if total_stmts else 0.0
    res += ["",
            f'   "{PROJECT_NAME} total", "{total_stmts}", "{total_exec}", "{cover:.1f} %"',
            ""]
    return os.linesep.join(res)


def setup_environment(options):
    """Pass the options down to the workers through the environment.

    pytest-xdist runs the tests in sub-processes, where the parsed options are
    out of reach: `pyFAI.test.utilstest.TestOptions.configure`, called by the
    conftest of the package, reads those variables instead.
    """
    if not options.gui:
        os.environ["WITH_QT_TEST"] = "False"
    if not options.opengl:
        os.environ["WITH_GL_TEST"] = "False"
    if not options.opencl:
        os.environ["PYFAI_OPENCL"] = "False"
    if options.random:
        os.environ["PYFAI_RANDOM"] = "True"
    if options.low_mem:
        os.environ["PYFAI_LOW_MEM"] = "True"

    if options.qt_screen:
        if platform.system() == "Windows":
            off = on = "windows"
        else:
            on, off = "minimal", "offscreen"
        if options.qt_screen.lower() == "on":
            os.environ.setdefault("QT_QPA_PLATFORM", on)
        else:
            value = off if options.qt_screen.lower() == "off" else options.qt_screen
            os.environ["QT_QPA_PLATFORM"] = value

    if options.qt_binding:
        # silx.gui.qt honours QT_API, which the workers inherit
        os.environ["QT_API"] = options.qt_binding
        logger.info("Force using %s", options.qt_binding)


def import_project(installed):
    """Build the project and make the freshly built copy importable

    :param installed: test the installed version instead of building
    :return: the imported project module
    """
    if installed:
        for bad_path in (".", os.getcwd(), PROJECT_DIR):
            if bad_path in sys.path:
                sys.path.remove(bad_path)
        print("Running tests on system-wide installed project")
    else:
        build_dir = build_project(PROJECT_NAME, PROJECT_DIR)
        sys.path.insert(0, build_dir)
        logger.warning("Patched sys.path, added: '%s'", build_dir)
    return importlib.import_module(PROJECT_NAME)


def to_pytest_id(name):
    """Translate a unittest test name into the identifier pytest expects

    `pyFAI.test.test_csr.TestCSR.test_2d_splitbbox` is a module, a class and a
    method for unittest, but `pyFAI.test.test_csr::TestCSR::test_2d_splitbbox`
    for pytest. The split is found by importing the longest prefix which is a
    module.

    :param name: dotted name, as accepted by run_tests.py
    :return: identifier suitable for `pytest --pyargs`
    """
    parts = name.split(".")
    for last in range(len(parts), 0, -1):
        module_name = ".".join(parts[:last])
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return "::".join([module_name] + parts[last:])
    raise ValueError(f"'{name}' does not start with an importable module")


def build_pytest_args(options, coverage_xml=None):
    """Build the command line handed over to pytest

    :param options: parsed command line
    :param coverage_xml: file the XML coverage report is written to
    """
    args = ["--pyargs"]

    if options.test_name:
        args += [to_pytest_id(name) for name in options.test_name]
    else:
        args.append(PROJECT_NAME)

    if options.memprofile:
        # The profile is written by the controller, while the tests run in the
        # workers: measuring in parallel would report an empty profile, and the
        # resident memory of a worker would mix the tests it happens to run.
        logger.warning("--memprofile measures the tests serially, ignoring -n %s",
                       options.workers)
        workers = 0
    else:
        # Keep the libraries from spawning one thread per CPU in every worker:
        # the workers inherit this environment, and oversubscribing slows the
        # suite down (it even crashed a worker of test_multi_geometry).
        if options.threads:
            THREADS_PER_CORE = options.threads
        if available_cores() > THREADS_PER_CORE:
            for key in ("OMP_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMBA_NUM_THREADS"):
                os.environ.setdefault(key, str(THREADS_PER_CORE))
        workers = options.workers or default_workers()
    args += ["-n", str(workers)]

    if options.verbose == 1:
        args.append("-v")
    elif options.verbose > 1:
        args.append("-vv")

    if options.coverage:
        try:
            importlib.import_module("pytest_cov")
        except ImportError:
            raise SystemExit("--coverage needs the 'pytest-cov' module: "
                             "coverage.py alone does not see the xdist workers")
        args += [f"--cov={PROJECT_NAME}",
                 "--cov-report=term",
                 f"--cov-report=xml:{coverage_xml}"]

    if options.memprofile:
        args += ["-p", f"{PROJECT_NAME}.test.profiler", "--profile-out=profile.json"]

    # Everything after `--` goes to pytest untouched
    args += options.pytest_args
    return args


def main():
    options = parse_command_line(sys.argv[1:])

    if options.verbose == 1:
        logging.root.setLevel(logging.INFO)
        logger.info("Set log level: INFO")
    elif options.verbose > 1:
        logging.root.setLevel(logging.DEBUG)
        logger.info("Set log level: DEBUG")

    setup_environment(options)
    module = import_project(options.installed)

    try:
        import pytest
    except ImportError:
        raise SystemExit("pytest is missing: pip install pytest pytest-xdist")
    try:
        importlib.import_module("xdist")
    except ImportError:
        raise SystemExit("pytest-xdist is missing: pip install pytest-xdist")

    logger.warning("Test %s %s from %s", PROJECT_NAME,
                   getattr(module, "version", ""), module.__path__[0])

    coverage_xml = None
    if options.coverage:
        handle, coverage_xml = tempfile.mkstemp(suffix=".xml", prefix="coverage_")
        os.close(handle)

    args = build_pytest_args(options, coverage_xml)
    logger.info("pytest %s", " ".join(args))
    exit_status = pytest.main(args)

    if options.coverage:
        report = coverage_rst(coverage_xml, module.__path__[0],
                              getattr(module, "version", ""))
        with open("coverage.rst", "w", encoding="utf-8") as rst:
            rst.write(report)
        os.remove(coverage_xml)
        print(f"Coverage report written to coverage.rst: "
              f"{report.splitlines()[-1].strip()}")

    if not exit_status:
        from pyFAI.test.utilstest import UtilsTest
        UtilsTest.clean_up()
    return int(exit_status)


if __name__ == "__main__":
    sys.exit(main())
