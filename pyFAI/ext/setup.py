# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/04/2019"

from numpy.distutils.misc_util import Configuration
import platform
import os
import numpy


def create_extension_config(name, extra_sources=None, can_use_openmp=False):
    """
    Util function to create numpy extension from the current pyFAI project.
    Prefer using numpy add_extension without it.
    """
    include_dirs = ['src', numpy.get_include()]

    if can_use_openmp:
        extra_link_args = ['-fopenmp']
        extra_compile_args = ['-fopenmp']
    else:
        extra_link_args = []
        extra_compile_args = []

    sources = ["%s.pyx" % name]
    if extra_sources is not None:
        sources.extend(extra_sources)

    config = dict(
        name=name,
        sources=sources,
        include_dirs=include_dirs,
        language='c',
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args
    )

    return config


def configuration(parent_package='', top_path=None):
    config = Configuration('ext', parent_package, top_path)

    ext_modules = [
        create_extension_config('histogram', can_use_openmp=True),
        create_extension_config("_geometry", can_use_openmp=True),
        create_extension_config("reconstruct", can_use_openmp=True),
        create_extension_config('splitPixel'),
        create_extension_config('splitPixelFull'),
        create_extension_config('splitPixelFullLUT'),
        create_extension_config('splitBBox'),
        create_extension_config('splitBBoxLUT', can_use_openmp=True),
        create_extension_config('splitBBoxCSR', can_use_openmp=True),
        create_extension_config('splitPixelFullCSR', can_use_openmp=True),
        create_extension_config('relabel'),
        create_extension_config("bilinear", can_use_openmp=True),
        # create_extension_config('_distortionCSR', can_use_openmp=True),
        create_extension_config('_bispev', can_use_openmp=True),
        create_extension_config('_convolution', can_use_openmp=True),
        create_extension_config('_blob'),
        create_extension_config('morphology'),
        create_extension_config('watershed'),
        create_extension_config('_tree'),
        create_extension_config('sparse_utils'),
        create_extension_config('preproc', can_use_openmp=True),
        create_extension_config('inpainting'),
        create_extension_config('invert_geometry')
    ]
    if (os.name == "posix") and ("x86" in platform.machine()):
        extra_sources = [os.path.join("src", "crc32.c")]
        ext_config = create_extension_config('fastcrc', extra_sources=extra_sources)
        ext_modules.append(ext_config)

    for ext_config in ext_modules:
        config.add_extension(**ext_config)

    config.add_extension('_distortion',
                         sources=['_distortion.pyx'],
                         include_dirs=[numpy.get_include()],
                         language='c++',
                         extra_link_args=['-fopenmp'],
                         extra_compile_args=['-fopenmp'])

    config.add_extension('sparse_builder',
                         sources=['sparse_builder.pyx'],
                         include_dirs=[numpy.get_include()],
                         language='c++',
                         extra_link_args=['-fopenmp'],
                         extra_compile_args=['-fopenmp'])

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
