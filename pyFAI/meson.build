subdir('app')
subdir('benchmark')
subdir('detectors')
subdir('engines')
subdir('ext')
subdir('geometry')
subdir('gui')
subdir('io')
subdir('opencl')
subdir('resources')
subdir('test')
subdir('third_party')
subdir('utils')

version_py = configure_file(input: '../version.py', output: '_version.py', copy: true)

py.install_sources(
   [ version_py,
    '__init__.py',
    'average.py',
    'azimuthalIntegrator.py',
    'blob_detection.py',
    'calibrant.py',
    'containers.py',
    'control_points.py',
    'diffmap.py',
    'directories.py',
    'distortion.py',
    'geometryRefinement.py',
    'goniometer.py',
    'grid.py',
    'load_integrators.py',
    'massif.py',
    'method_registry.py',
    'multi_geometry.py',
    'parallax.py',
    'spline.py',
    'units.py',
    'worker.py'],
  pure: false,    # Will be installed next to binaries
  subdir: 'pyFAI'  # Folder relative to site-packages to install to
)
