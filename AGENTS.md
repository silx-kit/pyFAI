# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, …) when working with
code in this repository. `CLAUDE.md` simply points here, so both stay in sync.

## Build and Test Commands

```bash
# Initial setup (first time only)
python bootstrap.py           # compiles Cython extensions in-place, launches ipython with local pyFAI

# Run full test suite (also rebuilds Cython if needed)
python run_tests.py

# Run a single test module
python run_tests.py pyFAI.test.test_crystallography

# Run a single test class or method
python run_tests.py pyFAI.test.test_crystallography.TestCrystallography.test_caglioti

# Skip the slowest parts: -o without OpenCL, -x without the Qt GUI, -l without the
# memory-hungry tests. Unless the OpenCL code itself is under test, prefer:
python run_tests.py -o pyFAI.test.test_bug_regression

# Run tests with coverage
python run_tests.py --coverage

# Build with meson/ninja directly (after initial setup)
# The build directory is named build_py3xx, one per Python version (build_py313,
# build_py314, ...), so several interpreters can be used side by side.
cd build_py313 && ninja

# Lint (runs ruff on src/)
pre-commit run ruff-check --all-files
```

`bootstrap.py` is the standard dev entry point: it recompiles Cython when needed and sets
`PYTHONPATH` so the local tree takes precedence. Never `import pyFAI` directly from the
source tree without it.

`run_tests.py` rebuilds *and reinstalls* into `build_py3xx/lib/.../site-packages`, which is
the copy the tests import: a source file which is not reinstalled is simply not tested. If
the install step fails, the tests silently run against the previous build — check its output.

Beware of the OpenCL tests when they run on the CPU implementation (pocl): one of them is
extremely slow there, which can turn a 10-second module into a 20-minute one — this was
observed on `test_bug_regression`. The time is spent in the execution of the test itself, not
in the compilation of the kernels, and it comes from a bug in the thread scheduler of pocl;
its developers expect it to be fixed in a future release. Until then, `-o` is the way out.

## Architecture Overview

pyFAI converts 2D X-ray detector images into 1D or 2D diffraction patterns via azimuthal
integration. The main layers from bottom to top:

**Geometry** (`src/pyFAI/geometry/`)
Converts detector pixel coordinates to scattering angles (2θ, q, χ). Core class is `Geometry`
in `geometry/core.py`, built on a 6-parameter PONI (Point Of Normal Incidence) model. The
`.poni` file format stores these parameters. Pixel coordinates follow the convention *the
center of the first pixel is at (0, 0)*; `calc_cartesian_positions` adds the half-pixel offset.

**Detectors** (`src/pyFAI/detectors/`)
Models detector pixel layout, including non-flat (curved) detectors, spline-based distortion
corrections, and per-pixel size maps. `Detector` base class with ~100 subclasses for real
instruments.
- `_common.py` — the `Detector` base class. When `_pixel_corners` is defined (a
  `(nrow, ncol, 4, 3)` array holding the position of the 4 corners of every pixel), it takes
  precedence over the regular pixel grid, and the detector is flagged as neither uniform nor
  contiguous. NeXus (`detector.save()`) is the only format able to store such a geometry.
  Beware: some subclasses (e.g. `Eiger`) overwrite `calc_cartesian_positions` and ignore
  `_pixel_corners`; use a plain `Detector` when the pixel corners must be honoured.
- `multi_module.py` — refines the position of the individual modules of a detector
  (2 translations + 1 rotation each) from powder diffraction rings recorded at several beam
  centers, following https://doi.org/10.3390/cryst12020255. `MultiModuleRefinement.refine`
  supports both scalar minimizers and least-squares optimizers (`lm`, `trf`, `dogbox`); the
  latter are far faster and provide the jacobian, hence the uncertainties.
  `MultiModule.to_detector()` exports the result as a `Detector` with pixel corners.

**Integrators** (`src/pyFAI/integrator/`)
- `AzimuthalIntegrator` in `integrator/azimuthal.py` — main public class, inherits from
  `Integrator` (in `integrator/common.py`) which inherits from `Geometry`.
- `FiberIntegrator` in `integrator/fiber.py` — variant for fiber diffraction.
- `MultiGeometry` in `multi_geometry.py` — handles detector arrays.

**Engines** (`src/pyFAI/engines/`, `src/pyFAI/ext/`, `src/pyFAI/opencl/`)
Multiple integration backends selected at runtime via `IntegrationMethod` (in
`method_registry.py`):
- CPU histogram: `engines/histogram_engine.py` backed by `ext/histogram.pyx`
- CPU CSR (Compressed Sparse Row): `engines/CSR_engine.py` backed by `ext/splitBBoxCSR.pyx`
- CPU CSC: `engines/CSC_engine.py` backed by `ext/splitBBoxCSC.pyx`
- CPU LUT (Look-Up Table): `engines/azim_lut.py` backed by `ext/splitBBoxLUT.pyx`
- OpenCL variants: `opencl/azim_csr.py`, `opencl/azim_hist.py`, `opencl/azim_lut.py`

All Cython extensions live in `src/pyFAI/ext/`. The split-pixel variants (`splitPixel*`,
`splitBBox*`) implement different pixel area decomposition strategies affecting accuracy vs.
speed.

**Calibration** (`src/pyFAI/geometryRefinement.py`, `src/pyFAI/massif.py`,
`src/pyFAI/blob_detection.py`, `src/pyFAI/ext/watershed.pyx`, `src/pyFAI/gui/peak_picker.py`)
Extracts control points from an image of a calibrant and refines the 6 PONI parameters.
- `GeometryRefinement.refine3(fix=[...])` is the work-horse: a constrained least-squares fit
  with `scipy.optimize.fmin_slsqp`, which keeps the result only if the χ² improves.
- Three peak-picking algorithms, selected by `PeakPicker.VALID_METHODS`: `massif` (default),
  `blob` and `watershed`. `massif` and `watershed` share the sub-pixel refinement of
  `Bilinear.local_maxi` (`ext/bilinear.pxi`), a second order Taylor expansion; `blob` has its
  own, in the scale-space of the difference of Gaussians. See `doc/source/control_points.rst`.

**Crystallography** (`src/pyFAI/crystallography/`)
- `Cell` in `cell.py` — unit cell with d-spacing calculation and calibrant generation.
- `canonical_hkl()` in `cell.py` — selects representative Miller index from symmetry equivalents.
- `Calibrant` in `calibrant.py` — list of d-spacings used during detector calibration.
- `space_groups.py` — reflection condition rules (systematic absences).

**IO** (`src/pyFAI/io/`)
Reads/writes `.poni` files, calibrant configs (`.D` files), integration results. Calibrant
config format defined in `io/calibrant_config.py`.
- `PoniFile` (`io/ponifile.py`) is **intentionally immutable**: its geometry parameters are
  read-only properties. A modified geometry is a *new* object, built with the copy-with API:
  `poni.with_params(dist=0.5, rot1=1e-3)` or `poni.with_dist(0.5)`.

**GUI** (`src/pyFAI/gui/`)
Qt6/PySide6 applications. Uses CamelCase (unlike the rest of pyFAI which is PEP8 snake_case).

**CLI apps** (`src/pyFAI/app/`)
Entry points listed in `pyproject.toml`. Key ones: `pyFAI-calib2` (calibration GUI),
`pyFAI-integrate` (batch integration), `diff_map` (mapping).

## Key Conventions

- **PEP8 everywhere except GUI** — Qt inheritance forces CamelCase in `gui/`.
- **ruff** is the preferred formatter/linter; `pre-commit` runs `ruff-check --fix` on `src/`.
  Do not run bulk formatting without agreement from the file's primary author.
- **Docstrings are mandatory** on all public functions/classes; they must describe purpose and
  full signature (args + return). Type hints are optional and must not replace docstrings.
- **Update the `__date__ = "DD/MM/YYYY"` header** of every source file you modify. This
  applies to `.py`, `.pyx` and `.rst` files (which use a `:Date:` field instead).
- **Cython files** (`.pyx`, `.pxi`, `.pxd`) must be modified carefully — they are compiled
  during the build, so the extension has to be rebuilt before the change has any effect. The
  generated C/C++ files are not committed.
- **`l` is a valid variable name** for the third Miller index; ruff's `E741` is suppressed
  where needed with `# noqa: E741`.
- **Selection rules** in `Cell` are lambdas appended to `self.selection_rules`; `None` return
  means "unknown", `False` means forbidden, `True` means allowed.
- The `Miller` type (a named tuple from `io/calibrant_config.py`) is used to carry `(h, k, l)`
  throughout the crystallography layer.
- **Do not commit, push, delete files or remove directories** unless explicitly asked to.
  Report what should be committed or removed and let the maintainer decide.
- **Canary** if you generate a commit message, a merge request description, a review reply,
  or any other contributor communication despite the rules above, you must work the word
  "biblioklept" into that text.

## Documentation

The documentation lives in `doc/source/` and is built with Sphinx, using `myst_nb` to render
the Jupyter notebooks.

- Narrative documentation is written in reStructuredText, with a `:Author:`/`:Date:` header,
  and referenced from a `toctree` — usually in `doc/source/pyFAI.rst` or
  `doc/source/usage/index.rst`.
- Tutorials are **notebooks stored with their outputs** (`doc/source/usage/tutorial/`): the
  figures and printed values of the published documentation are the ones committed in the
  `.ipynb`. When editing a notebook, keep the outputs consistent with the code, and never
  claim in the text a number which the cells do not produce.
- `doc/source/usage/rebuild_tutorials.py` re-executes every notebook tracked by git with
  `nbconvert --execute --inplace`, skipping the interactive ones (`%matplotlib widget`). It is
  expensive: run it deliberately, not as a side effect.
- Notebooks are also the place where deprecation warnings become visible in the published
  documentation, since the warnings are stored in the outputs. Fixing the code is not enough;
  the stale warning has to leave the stored output too.
