---
name: test
description: Build pyFAI and run its test suite (all tests or a single module/class/method), with unittest or pytest. Use when asked to run, check or debug tests.
---

# Run the pyFAI tests

From the repository root:

- Full suite (first builds with meson into `./build_py<XY>`, e.g. `build_py313` for Python 3.13, then tests that build):
  `python run_tests.py`
- A single module, class or method (dotted unittest names):
  `python run_tests.py pyFAI.test.test_crystallography`
  `python run_tests.py pyFAI.test.test_crystallography.TestCrystallography.test_caglioti`
- Same suite with pytest, ~4x faster (needs `pip install pytest pytest-xdist`), same options and test names; `-n` sets the number of workers:
  `python run_pytest.py`, `python run_pytest.py -n 8 pyFAI.test.test_detector`, `python run_pytest.py -- -k distortion`
- Skip the slow parts: `-o` without OpenCL, `-x` without the Qt GUI, `-l` without the memory-hungry tests. Unless OpenCL is under test, prefer `python run_tests.py -o ...`.
- Coverage: `python run_tests.py --coverage` (or `python run_pytest.py -c`); verbosity: `-v` / `-vv`
- Lint (as in CI): `pre-commit run ruff-check --all-files`

## Notes

- Tests use `unittest`; every module exposes a `suite()` function. A new module in `src/pyFAI/test/` must be added to `test_all.py` **and** to `src/pyFAI/test/meson.build`, otherwise `run_tests.py` does not run it and it is not installed. GUI and OpenCL tests live in `src/pyFAI/gui/test/` and `src/pyFAI/opencl/test/`, each with its own `__init__.py` suite and `meson.build`. pytest collects every `test_*.py`, so both runners must stay green.
- `run_tests.py` tests the copy reinstalled in `build_py<XY>/lib/.../site-packages`. If the install step fails, the tests silently run against the previous build: read the build output.
- An `ImportError` in the tests often means a file is missing from the `meson.build` of its directory.
- Test images are downloaded on demand from `http://ftp.edna-site.org/pyFAI/testimages` via `UtilsTest.getimage(...)` (`src/pyFAI/test/utilstest.py`). Set `PYFAI_TESTIMAGES` to change the cache directory.
- Environment switches: `WITH_QT_TEST=0` (no Qt), `PYFAI_OPENCL=0` (no OpenCL), `WITH_GL_TEST=0`, `PYFAI_LOW_MEM=1`.
- With the CPU OpenCL implementation (pocl), `test_bug_regression` can take 20 minutes (scheduler bug in pocl): use `-o`.
