---
name: bootstrap
description: Run Python code, a script, a module or a project entry point against the pyFAI source checkout (compiled with meson, no install needed) via bootstrap.py. Use to try the library, reproduce a bug or run a CLI tool on the current sources.
---

# Run code against the development version

`bootstrap.py` (repository root) builds the project with meson into `./build_py<XY>` (the build directory name contains the Python version, e.g. `build_py313` for Python 3.13, so several interpreters can coexist), prepends the result to `sys.path`, then executes the requested target. Always use it rather than the `pyFAI` installed in the environment, which may differ from the sources. Never `import pyFAI` straight from the source tree: the Cython extensions are not built in place.

From the repository root:

- Python commands, like `python -c` (repeatable option: commands run in order and share the same namespace; remaining arguments end up in `sys.argv[1:]`):
  `python bootstrap.py -c "import pyFAI; print(pyFAI.__file__)"`
  `python bootstrap.py -c "import pyFAI.detectors as d" -c "print(d.detector_factory('Pilatus1M').shape)"`
- Python script: `python bootstrap.py my_script.py arg1 arg2`
- Entry point declared in `pyproject.toml`: `python bootstrap.py pyFAI-integrate --help`
- Module: `python bootstrap.py -m some.module args`, e.g. `python bootstrap.py -m pytest --pyargs pyFAI.test.test_detector`
- No argument: interactive IPython console (or standard interpreter); avoid it when driven by an agent.

## Notes

- The first call compiles the Cython extensions (slow); later calls only rebuild what changed.
- `meson` messages are printed before the output of the executed code.
- Set `PYFAI_OPENCL=0` to skip the OpenCL initialisation when it is irrelevant.
- To run the test suite, use the `test` skill.
