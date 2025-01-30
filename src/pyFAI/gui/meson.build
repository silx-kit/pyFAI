subdir('dialog')
subdir('helper')
subdir('jupyter')
subdir('model')
subdir('pilx')
subdir('tasks')
subdir('test')
subdir('utils')
subdir('widgets')

py.install_sources(
  ['ApplicationContext.py',
   'CalibrationContext.py',
   'CalibrationWindow.py',
   'IntegrationDialog.py',
   '__init__.py',
   'cli_calibration.py',
   'diffmap_widget.py',
   'matplotlib.py',
   'mpl_calib.py',
   'mpl_calib_qt.py',
   'peak_picker.py'],
  pure: false,   # Will be installed next to binaries
  subdir: 'pyFAI/gui'  # Folder relative to site-packages to install to
)
