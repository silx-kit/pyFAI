#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##########################################################################
#
# Copyright (C) 2015-2018 European Synchrotron Radiation Facility
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
# ###########################################################################
"""
Bootstrap helps you to test scripts without installing them
by patching your PYTHONPATH on the fly

example: ./bootstrap.py ipython
"""

__authors__ = ["Frédéric-Emmanuel Picca", "Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "04/12/2020"

import sys
import os
import distutils.util
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bootstrap")


def _distutils_dir_name(dname="lib"):
    """
    Returns the name of a distutils build directory
    """
    platform = distutils.util.get_platform()
    architecture = "%s.%s-%i.%i" % (dname, platform,
                                    sys.version_info[0], sys.version_info[1])
    return architecture


def _distutils_scripts_name():
    """Return the name of the distrutils scripts sirectory"""
    f = "scripts-{version[0]}.{version[1]}"
    return f.format(version=sys.version_info)


def _get_available_scripts(path):
    res = []
    try:
        res = " ".join([s.rstrip('.py') for s in os.listdir(path)])
    except OSError:
        res = ["no script available, did you ran "
               "'python setup.py build' before bootstrapping ?"]
    return res


if sys.version_info[0] >= 3:  # Python3

    def execfile(fullpath, globals=None, locals=None):
        "Python3 implementation for execfile"
        with open(fullpath) as f:
            try:
                data = f.read()
            except UnicodeDecodeError:
                raise SyntaxError("Not a Python script")
            code = compile(data, fullpath, 'exec')
            exec(code, globals, locals)


def run_file(filename, argv):
    """
    Execute a script trying first to use execfile, then a subprocess

    :param str filename: Script to execute
    :param list[str] argv: Arguments passed to the filename
    """
    full_args = [filename]
    full_args.extend(argv)

    try:
        logger.info("Execute target using exec")
        # execfile is considered as a local call.
        # Providing globals() as locals will force to feed the file into
        # globals() (for examples imports).
        # Without this any function call from the executed file loses imports
        try:
            old_argv = sys.argv
            sys.argv = full_args
            logger.info("Patch the sys.argv: %s", sys.argv)
            logger.info("Executing %s.main()", filename)
            print("########### EXECFILE ###########")
            d = globals()
            d["__file__"] = filename
            execfile(filename, d, d)
        finally:
            sys.argv = old_argv
    except SyntaxError as error:
        logger.error(error)
        logger.info("Execute target using subprocess")
        env = os.environ.copy()
        env.update({"PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    "PATH": os.environ.get("PATH", "")})
        print("########### SUBPROCESS ###########")
        run = subprocess.Popen(full_args, shell=False, env=env)
        run.wait()


def run_entry_point(entry_point, argv):
    """
    Execute an entry_point using the current python context
    (http://setuptools.readthedocs.io/en/latest/setuptools.html#automatic-script-creation)

    :param str entry_point: A string identifying a function from a module
        (NAME = PACKAGE.MODULE:FUNCTION)
    """
    import importlib
    # Remove ending extra dependencies
    entry_point = entry_point.split("[")[0]
    elements = entry_point.split("=")
    target_name = elements[0].strip()
    elements = elements[1].split(":")
    module_name = elements[0].strip()
    function_name = elements[1].strip()

    logger.info("Execute target %s (function %s from module %s) using importlib", target_name, function_name, module_name)
    full_args = [target_name]
    full_args.extend(argv)
    try:
        old_argv = sys.argv
        sys.argv = full_args
        print("########### IMPORTLIB ###########")
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            func()
        else:
            logger.info("Function %s not found", function_name)
    finally:
        sys.argv = old_argv


def find_executable(target):
    """Find a filename from a script name.

    - Check the script name as file path,
    - Then checks if the name is a target of the setup.py
    - Then search the script from the PATH environment variable.

    :param str target: Name of the script
    :returns: Returns a tuple: kind, name.
    """
    if os.path.isfile(target):
        return ("path", os.path.abspath(target))

    # search the file from setup.py
    import setup
    config = setup.get_project_configuration(dry_run=True)
    # scripts from project configuration
    if "scripts" in config:
        for script_name in config["scripts"]:
            if os.path.basename(script) == target:
                return ("path", os.path.abspath(script_name))
    # entry-points from project configuration
    if "entry_points" in config:
        for kind in config["entry_points"]:
            for entry_point in config["entry_points"][kind]:
                elements = entry_point.split("=")
                name = elements[0].strip()
                if name == target:
                    return ("entry_point", entry_point)

    # search the file from env PATH
    for dirname in os.environ.get("PATH", "").split(os.pathsep):
        path = os.path.join(dirname, target)
        if os.path.isfile(path):
            return ("path", path)

    return None, None


home = os.path.dirname(os.path.abspath(__file__))
LIBPATH = os.path.join(home, 'build', _distutils_dir_name('lib'))
cwd = os.getcwd()
os.chdir(home)
build = subprocess.Popen([sys.executable, "setup.py", "build"], shell=False)
logger.info("Build process ended with rc= %s", build.wait())
os.chdir(cwd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.warning("usage: ./bootstrap.py <script>\n")
        script = None
    else:
        script = sys.argv[1]

    if script:
        logger.info("Executing %s from source checkout", script)
    else:
        logging.info("Running iPython by default")
    sys.path.insert(0, LIBPATH)
    logger.info("Patched sys.path with %s", LIBPATH)

    if script:
        argv = sys.argv[2:]
        kind, target = find_executable(script)
        if kind == "path":
            run_file(target, argv)
        elif kind == "entry_point":
            run_entry_point(target, argv)
        else:
            logger.error("Script %s not found", script)
    else:
        logger.info("Patch the sys.argv: %s", sys.argv)
        sys.path.insert(2, "")
        try:
            from IPython import embed
        except Exception as err:
            logger.error("Unable to execute iPython, using normal Python")
            logger.error(err)
            import code
            code.interact()
        else:
            embed()
