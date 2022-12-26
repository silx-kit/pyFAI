py.install_sources(['__init__.py'],
    pure: false,   # Will be installed next to binaries
    subdir: 'pyFAI/ext')

py.extension_module('preproc', 'preproc.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('histogram', 'histogram.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_geometry', '_geometry.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('reconstruct', 'reconstruct.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

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
        override_options : ['cython_language=cpp'], #TODO: OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullCSR', 'splitPixelFullCSR.pyx',
        override_options : ['cython_language=cpp'], #TODO: OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBoxLUT', 'splitBBoxLUT.pyx',
        override_options : ['cython_language=cpp'], #TODO: OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullLUT', 'splitPixelFullLUT.pyx',
        override_options : ['cython_language=cpp'], #TODO: OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitBBoxCSC', 'splitBBoxCSC.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('splitPixelFullCSC', 'splitPixelFullCSC.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('relabel', 'relabel.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('bilinear', 'bilinear.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_bispev', '_bispev.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_convolution', '_convolution.pyx', #TODO OpenMP
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_blob', '_blob.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('morphology', 'morphology.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('watershed', 'watershed.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_tree', '_tree.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('sparse_utils', 'sparse_utils.pyx', #Deprecated
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('inpainting', 'inpainting.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('invert_geometry', 'invert_geometry.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('dynamic_rectangle', 'dynamic_rectangle.pyx',
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('sparse_builder', 'sparse_builder.pyx',
        override_options : ['cython_language=cpp'], # TODO: parallel
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

py.extension_module('_distortion', '_distortion.pyx',
        override_options : ['cython_language=cpp'], # TODO: parallel
        dependencies : py_dep, install: true, subdir: 'pyFAI/ext')

#    if (os.name == "posix") and ("x86" in platform.machine()):
#        extra_sources = [os.path.join("src", "crc32.c")]
#        ext_config = create_extension_config('fastcrc', extra_sources=extra_sources)
#        ext_modules.append(ext_config)
