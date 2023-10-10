subdir('test')

py.install_sources(
   ['OCLFullSplit.py',
    '__init__.py',
    'azim_csr.py',
    'azim_hist.py',
    'azim_lut.py',
    'ocl_hist_pixelsplit.py',
    'peak_finder.py',
    'preproc.py',
    'sort.py'],
  pure: false,   # Will be installed next to binaries
  subdir: 'pyFAI/opencl'  # Folder relative to site-packages to install to
)
