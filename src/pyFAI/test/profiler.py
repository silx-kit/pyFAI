#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Pytest plugin measuring the time and the memory spent by every test.

It is the pytest counterpart of the `--profile` option of `run_tests.py`, with
the memory reported per test rather than per test-case, and a machine-readable
output::

    python bootstrap.py -m pytest --pyargs pyFAI -n 0 \\
        -p pyFAI.test.profiler --profile-out=profile.json

Three numbers are recorded for each test:

`duration`
    wall-clock time of the whole protocol, split into setup / call / teardown.
    A large `setup` usually means an expensive `setUpClass` attributed to the
    first test of the class.

`rss_kept`
    resident memory still held once the test is over: what the test leaks into
    the interpreter (caches, class attributes, module-level state).

`rss_peak`
    growth of the high-water mark of the process. It only accounts for tests
    allocating more than every test run so far, which is exactly what makes a
    test-suite hit the memory limits of a CI runner.

`--profile-tracemalloc` adds `alloc_peak`, the peak of the Python allocations
(numpy buffers included), which is precise but slows the tests down by a factor
of a few: use it on a selection of tests, not on the whole suite.

The profile must be recorded serially (`-n 0`): with xdist workers the resident
memory of a process mixes the tests it happens to run.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/09/2026"

import gc
import json
import resource
import time
import tracemalloc

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("pyFAI profiling")
    group.addoption("--profile-out", action="store", default="profile.json",
                    metavar="FILE",
                    help="File where the per-test measurements are written (JSON)")
    group.addoption("--profile-top", action="store", type=int, default=25,
                    metavar="N",
                    help="Number of tests listed in the terminal report")
    group.addoption("--profile-tracemalloc", action="store_true", default=False,
                    help="Also record the peak of the Python allocations "
                         "(precise, but several times slower)")


def _rss_kb():
    """Resident set size of the process, in kbytes"""
    with open("/proc/self/statm") as statm:
        pages = int(statm.read().split()[1])
    return pages * resource.getpagesize() // 1024


def _maxrss_kb():
    """High-water mark of the resident set size, in kbytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class Profiler:

    def __init__(self, config):
        self.config = config
        self.tracemalloc = config.getoption("profile_tracemalloc")
        self.records = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        gc.collect()
        rss_before = _rss_kb()
        maxrss_before = _maxrss_kb()
        if self.tracemalloc:
            tracemalloc.start()
        try:
            yield
        finally:
            alloc_peak = 0
            if self.tracemalloc:
                alloc_peak = tracemalloc.get_traced_memory()[1] // 1024
                tracemalloc.stop()
            gc.collect()
            record = self.records.setdefault(item.nodeid, {})
            record["rss_kept"] = _rss_kb() - rss_before
            record["rss_peak"] = _maxrss_kb() - maxrss_before
            record["rss_total"] = _rss_kb()
            if self.tracemalloc:
                record["alloc_peak"] = alloc_peak

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        yield from self._time(item, "setup")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        yield from self._time(item, "call")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        yield from self._time(item, "teardown")

    def _time(self, item, phase):
        start = time.perf_counter()
        try:
            yield
        finally:
            record = self.records.setdefault(item.nodeid, {})
            record[phase] = time.perf_counter() - start
            record["duration"] = sum(record.get(key, 0.0)
                                     for key in ("setup", "call", "teardown"))

    def pytest_sessionfinish(self, session):
        filename = self.config.getoption("profile_out")
        with open(filename, "w") as jsonfile:
            json.dump(self.records, jsonfile, indent=1, sort_keys=True)

    def pytest_terminal_summary(self, terminalreporter):
        top = self.config.getoption("profile_top")
        if not (self.records and top):
            return
        write = terminalreporter.write_line
        keys = ["duration", "rss_peak", "rss_kept"]
        if self.tracemalloc:
            keys.append("alloc_peak")
        for key in keys:
            unit = "s" if key == "duration" else "kB"
            ranked = sorted(self.records.items(),
                            key=lambda kv: kv[1].get(key, 0), reverse=True)
            terminalreporter.write_sep("=", f"{top} most expensive tests: {key}")
            for nodeid, record in ranked[:top]:
                value = record.get(key, 0)
                if key == "duration":
                    detail = " ".join(f"{phase}={record.get(phase, 0):.2f}"
                                      for phase in ("setup", "call", "teardown"))
                    write(f"{value:9.2f} {unit}  {nodeid}  ({detail})")
                else:
                    write(f"{value:9d} {unit}  {nodeid}")
        write(f"Full profile written to {self.config.getoption('profile_out')}")


def pytest_configure(config):
    if config.getoption("profile_out"):
        config.pluginmanager.register(Profiler(config), "pyFAI-profiler")
