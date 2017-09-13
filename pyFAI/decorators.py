# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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
"""Bunch of useful decorators"""

from __future__ import absolute_import, print_function, division

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/09/2017"
__status__ = "development"
__docformat__ = 'restructuredtext'

import sys
import time
import functools
import logging
import traceback


timelog = logging.getLogger("pyFAI.timeit")
depreclog = logging.getLogger("pyFAI.DEPRECATION")

deprecache = set([])


def deprecated(func=None, reason=None, replacement=None, since_version=None, only_once=False):
    """
    Decorator that deprecates the use of a function

    :param str reason: Reason for deprecating this function
        (e.g. "feature no longer provided",
    :param str replacement: Name of replacement function (if the reason for
        deprecating was to rename the function)
    :param str since_version: First *silx* version for which the function was
        deprecated (e.g. "0.5.0").
    :param bool only_once: If true, the deprecation warning will only be
        generated one time. Default is true.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.func_name if sys.version_info[0] < 3 else func.__name__

            deprecated_warning(type_='function',
                               name=name,
                               reason=reason,
                               replacement=replacement,
                               since_version=since_version,
                               only_once=only_once,
                               skip_backtrace_count=1)
            return func(*args, **kwargs)
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator


def deprecated_warning(type_, name, reason=None, replacement=None,
                       since_version=None, only_once=False,
                       skip_backtrace_count=0):
    """
    Decorator that deprecates the use of a function

    :param str type_: Module, function, class ...
    :param str reason: Reason for deprecating this function
        (e.g. "feature no longer provided",
    :param str replacement: Name of replacement function (if the reason for
        deprecating was to rename the function)
    :param str since_version: First *silx* version for which the function was
        deprecated (e.g. "0.5.0").
    :param bool only_once: If true, the deprecation warning will only be
        generated one time. Default is true.
    :param int skip_backtrace_count: Amount of last backtrace to ignore when
        logging the backtrace
    """
    if not depreclog.isEnabledFor(logging.WARNING):
        # Avoid computation when it is not logged
        return

    msg = "%s, %s is deprecated"
    if since_version is not None:
        msg += " since silx version %s" % since_version
    msg += "!"
    if reason is not None:
        msg += " Reason: %s." % reason
    if replacement is not None:
        msg += " Use '%s' instead." % replacement
    msg = msg + "\n%s"
    selection = slice(-2 - skip_backtrace_count, -1 - skip_backtrace_count)
    backtrace = "".join(traceback.format_stack()[selection])
    backtrace = backtrace.rstrip()
    if only_once:
        data = (msg, type_, name, backtrace)
        if data in deprecache:
            return
        else:
            deprecache.add(data)
    depreclog.warning(msg, type_, name, backtrace)


def timeit(func):
    def wrapper(*arg, **kw):
        '''This is the docstring of timeit:
        a decorator that logs the execution time'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        name = func.func_name if sys.version_info[0] < 3 else func.__name__
        timelog.warning("%s took %.3fs", name, t2 - t1)
        return res
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
