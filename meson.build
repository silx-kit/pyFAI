project('pyFAI',
  'c', 'cpp', 'cython',
  license: 'MIT',
  meson_version: '>= 1.1',
  version: run_command(['version.py', '--wheel'],
                       check:true).stdout().strip(),
  default_options: ['buildtype=plain', ],
)

py_mod = import('python')
py = py_mod.find_installation()
os = import('fs')

# Seek the backend
if meson.backend() != 'ninja'
  error('Ninja backend required')
endif

# How to disable OpenMP:
# The 'PYFAI_WITH_OPENMP' file should contain 'False'
if os.is_file('PYFAI_WITH_OPENMP')
  openmp_var = os.read('PYFAI_WITH_OPENMP').strip()
else
  res = run_command(py, '-c', 'import os; print(os.environ["PYFAI_WITH_OPENMP"])',
                    check:false)
  if res.returncode() == 0
    openmp_var = res.stdout().strip()
  else
    openmp_var = ''
  endif
endif

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
# We need -lm for all C code (assuming it uses math functions, which is safe).
# For C++ it isn't needed, because libstdc++/libc++ is guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
py_dep = py.dependency()


# Detect non-MSVC compilers on Windows for both C and C++
if host_machine.system() == 'windows'
  c_id     = cc.get_id()
  cpp_id   = cpp.get_id()
  # List of ids we accept as “MSVC-compatible”
  acceptable = ['msvc', 'clang-cl']
  warning_msg = '''
    -----------------------------------------------------------------
    Detected @0@ compiler on Windows: @1@
    This project is primarily tested with Microsoft Visual C++.
    If you encounter problems, switch to MSVC or add the needed
    patches for your compiler.
    -----------------------------------------------------------------
  '''
  # C compiler check
  if c_id not in acceptable
    meson.warning(warning_msg.format('C', c_id))
  endif
  # C++ compiler check
  if cpp_id not in acceptable
    meson.warning(warning_msg.format('C++', cpp_id))
  endif
  # make it fatal:
  if c_id not in acceptable or cpp_id not in acceptable
     meson.error('Unsupported compiler on Windows - aborting.')
  endif
endif

py.install_sources([
    'version.py',
],
  pure: false,    # Will be installed next to binaries
  subdir: 'pyFAI'  # Folder relative to site-packages to install to
)

subdir('src/pyFAI')
