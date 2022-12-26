#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap helps you to test scripts without installing them
by patching your PYTHONPATH on the fly

example: ./bootstrap.py ipython
"""

__authors__ = ["Frédéric-Emmanuel Picca", "Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "06/12/2022"

import sys
import os
import subprocess
import logging
import tomli
logging.basicConfig()
logger = logging.getLogger("bootstrap")


def get_project_name(root_dir):
    """Retrieve project name by running python setup.py --name in root_dir.

    :param str root_dir: Directory where to run the command.
    :return: The name of the project stored in root_dir
    """
    logger.debug("Getting project name in %s", root_dir)
    with open("pyproject.toml") as f:
        pyproject = tomli.loads(f.read())
    return pyproject.get("project",{}).get("name")


def build_project(name, root_dir):
    """Build locally the project using meson

    :param str name: Name of the project.
    :param str root_dir: Root directory of the project
    :return: The path to the directory were build was performed
    """
    extra = []
    libdir = "lib"
    if sys.platform == "win32":
        libdir = "Lib"
        extra = ["--buildtype", "plain"]
    
    build = os.path.join(root_dir, "build")
    if not(os.path.isdir(build) and os.path.isdir(os.path.join(build, name))):
        p = subprocess.Popen(["meson", "build"],
                         shell=False, cwd=root_dir, env=os.environ)
        p.wait()
    p = subprocess.Popen(["meson", "configure", "--prefix", "/"] + extra,
                     shell=False, cwd=build, env=os.environ)
    p.wait()
    p = subprocess.Popen(["meson", "install", "--destdir", "."],
                     shell=False, cwd=build, env=os.environ)
    logger.debug("meson install ended with rc= %s", p.wait())
        
    
    if os.environ.get("PYBUILD_NAME") == name:
        # we are in the debian packaging way
        home = os.environ.get("PYTHONPATH", "").split(os.pathsep)[-1]
    elif os.environ.get("BUILDPYTHONPATH"):
        home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
    else:
        if sys.platform == "win32":
            home = os.path.join(build, "Lib", "site-packages")
        else:
            python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            home = os.path.join(build, "lib", python_version, "site-packages")
        home = os.path.abspath(home)

    cnt = 0
    while not os.path.isdir(home):
        cnt += 1
        home = os.path.split(home)[0]
    for _ in range(cnt):
        n = os.listdir(home)[0]
        home = os.path.join(home, n)

    logger.warning("Building %s to %s", name, home)
    
    return home


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
            module_globals = globals().copy()
            module_globals['__file__'] = filename
            execfile(filename, module_globals, module_globals)
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


def run_entry_point(target_name, entry_point, argv):
    """
    Execute an entry_point using the current python context

    :param str entry_point: A string identifying a function from a module
        (NAME = PACKAGE.MODULE:FUNCTION)
    :param argv: list of arguments
    """
    import importlib
    elements = entry_point.split(":")
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

    # search the executable in pyproject.toml
    with open(os.path.join(PROJECT_DIR, "pyproject.toml")) as f:
        pyproject = tomli.loads(f.read())
    for script, entry_point in list(pyproject.get("console_scripts",{}).items())+list(pyproject.get("gui_scripts",{}).items()):
        if script == target:
            print(script, entry_point)
            return ("entry_point", target, entry_point)
    return None, None


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = get_project_name(PROJECT_DIR)
logger.info("Project name: %s", PROJECT_NAME)


if __name__ == "__main__":
    LIBPATH = build_project(PROJECT_NAME, PROJECT_DIR)
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
        res = find_executable(script)
        if res[0]  == "path":
            run_file(res[1], argv)
        elif res[0]  == "entry_point":
            run_entry_point(res[1], res[2], argv)
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
