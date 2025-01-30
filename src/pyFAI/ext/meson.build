# OpenMP can be used? several ways to disable it:
#  -> use -Duse_openmp=disabled
#  -> or the file PYFAI_WITH_OPENMP should contain False
#  -> or OpenMP can be unsuported on your achitecture/compiler

omp = dependency('openmp', required: get_option('use_openmp').disable_auto_if(openmp_var=='False'))

py.install_sources(['__init__.py'],
    pure: false,   # Will be installed next to binaries
    subdir: 'pyFAI/ext')

py.extension_module('preproc', 'preproc.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('histogram', 'histogram.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('_geometry', '_geometry.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('reconstruct', 'reconstruct.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBox_common', 'splitBBox_common.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitpixel_common', 'splitpixel_common.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBox', 'splitBBox.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixel', 'splitPixel.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBoxCSR', 'splitBBoxCSR.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullCSR', 'splitPixelFullCSR.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBoxLUT', 'splitBBoxLUT.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullLUT', 'splitPixelFullLUT.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBoxCSC', 'splitBBoxCSC.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullCSC', 'splitPixelFullCSC.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('relabel', 'relabel.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('bilinear', 'bilinear.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('_bispev', '_bispev.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('_convolution', '_convolution.pyx',
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('_blob', '_blob.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('morphology', 'morphology.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('watershed', 'watershed.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_tree', '_tree.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('sparse_utils', 'sparse_utils.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('inpainting', 'inpainting.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('invert_geometry', 'invert_geometry.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('dynamic_rectangle', 'dynamic_rectangle.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('sparse_builder', 'sparse_builder.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('_distortion', '_distortion.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [py_dep, omp], install: true, subdir: 'pyFAI/ext')

py.extension_module('fastcrc', ['fastcrc.pyx','src/crc32.c'],
        dependencies : [py_dep], install: true, subdir: 'pyFAI/ext')

py.extension_module('mathutil', ['mathutil.pyx'],
        dependencies : [py_dep], install: true, subdir: 'pyFAI/ext')
