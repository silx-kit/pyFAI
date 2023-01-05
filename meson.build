project('pyFAI',
  'c', 'cython',
  license: 'MIT',
  meson_version: '>= 0.60',
  version: run_command(['version.py', '--wheel'],
                       check:true).stdout().strip(),
  default_options: ['buildtype=release', ],
)


# Seek the backend
if meson.backend() != 'ninja'
  error('Ninja backend required')
endif

# Seek for Cython installation
cython = find_program('cython', required : true)
if not cython.found()
  error('Cython not found.')
endif
cc = meson.get_compiler('c')

m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

# MinGW-W64 #
#if host_machine.system() == 'windows' and meson.get_compiler('c').get_id() == 'gcc'
#  add_project_link_arguments('-lucrt', '-static', language: 'c')
#  add_project_arguments('-mlong-double-64', language: 'c')
#  add_project_arguments('-DMS_WIN64', language: 'c')
#endif


subdir('pyFAI')
