# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2024 European Synchrotron Radiation Facility
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
"""Access project's data and documentation files.

All access to data and documentation files MUST be made through the functions
of this modules to ensure access across different distribution schemes:

- Installing from source or from wheel
- Installing package as a zip (through the use of pkg_resources)
- Linux packaging willing to install data files (and doc files) in
  alternative folders. In this case, this file must be patched.
- Frozen fat binary application using pyFAI (frozen with cx_Freeze or py2app).
  This needs special care for the resource files in the setup:

- With cx_Freeze, add pyFAI/resources to include_files:

.. code-block:: python

    import pyFAI.resources
    pyFAI_include_files = (os.path.dirname(pyFAI.resources.__file__),
                          os.path.join('pyFAI', 'resources'))
    setup(...,
        options={'build_exe': {'include_files': [pyFAI_include_files]}}
    )

- With py2app, add pyFAI in the packages list of the py2app options:

.. code-block:: python

    setup(...,
        options={'py2app': {'packages': ['pyFAI']}}
    )
"""

__authors__ = ["V.A. Sole", "Thomas Vincent"]
__license__ = "MIT"
__date__ = "07/03/2024"


import os
import sys
import logging

logger = logging.getLogger(__name__)

# importlib_resources is useful when this package is stored in a zip
# When importlib.resources is not available, the resources dir defaults to the
# directory containing this module.
if sys.version_info >= (3,9):
    import importlib.resources as importlib_resources
else:
    try:
        import  importlib_resources
    except ImportError:
        logger.info("Unable to import importlib_resources")
        logger.debug("Backtrace", exc_info=True)
        importlib_resources = None

if importlib_resources is not None:
        import atexit
        from contextlib import ExitStack
        file_manager = ExitStack()
        atexit.register(file_manager.close)


# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/pyFAI/data'
_RESOURCES_DIR = None

# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/pyFAI/doc'
# Not in use, uncomment when functionnality is needed
# _RESOURCES_DOC_DIR = None

# cx_Freeze forzen support
# See http://cx-freeze.readthedocs.io/en/latest/faq.html#using-data-files
if getattr(sys, 'frozen', False):
    # Running in a frozen application:
    # We expect resources to be located either in a pyFAI/resources/ dir
    # relative to the executable or within this package.
    _dir = os.path.join(os.path.dirname(sys.executable), 'pyFAI', 'resources')
    if os.path.isdir(_dir):
        _RESOURCES_DIR = _dir


def resource_filename(resource):
    """Return filename corresponding to resource.

    resource can be the name of either a file or a directory.
    The existence of the resource is not checked.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator.
    :return: Absolute resource path in the file system
    """
    # Not in use, uncomment when functionnality is needed
    # If _RESOURCES_DOC_DIR is set, use it to get resources in doc/ subflodler
    # from an alternative directory.
    # if _RESOURCES_DOC_DIR is not None and (resource is 'doc' or
    #         resource.startswith('doc/')):
    #     # Remove doc folder from resource relative path
    #     return os.path.join(_RESOURCES_DOC_DIR, *resource.split('/')[1:])

    if _RESOURCES_DIR is not None:  # if set, use this directory
        return os.path.join(_RESOURCES_DIR, *resource.split('/'))
    elif importlib_resources is None:  # Fallback if pkg_resources is not available
        return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            *resource.split('/'))
    else:  # Preferred way to get resources as it supports zipfile package
        ref = importlib_resources.files(__name__) / resource
        path = file_manager.enter_context(importlib_resources.as_file(ref))
        return str(path)


_integrated = False


def silx_integration():
    """Provide pyFAI resources accessible throug silx using a prefix."""
    global _integrated
    if _integrated:
        return
    import silx.resources
    silx.resources.register_resource_directory("pyfai",
                                               __name__,
                                               _RESOURCES_DIR)
    _integrated = True
