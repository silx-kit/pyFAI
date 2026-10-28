---
name: prepare-fix
description: Prepare the resolution of a pyFAI GitHub issue number XXX - make sure the work happens on a branch/worktree named XXX_SHORT_DESCRIPTION, add a failing regression test, create the ISSUE_XXX.md work file (checklist + agent notes, later used to write the pull request text) and add a one-line changelog entry. Does not fix the bug itself. Use when asked to start working on, or get ready to fix, an issue.
---

# Prepare the resolution of issue XXX

Input: the issue number `XXX` of https://github.com/silx-kit/pyFAI/issues.
Result: a correctly named branch, a regression test that currently fails, an `ISSUE_XXX.md` work file, and a changelog line. **The fix itself is out of scope**: do not modify the library code to make the test pass.

## 1. Understand the issue

Use the `issue` skill to read the issue and get the minimal reproducer and the suspected location. If it cannot be reproduced, stop and report it instead of inventing a test.

## 2. Branch or worktree

The work must happen on a git branch or worktree named `XXX_SHORT_DESCRIPTION` (issue number, underscore, a few lowercase words separated by underscores, e.g. `2956_multimodule_display`).

1. Check the current branch (`git branch --show-current`) and worktrees (`git worktree list`).
2. If the current branch/worktree already starts with `XXX_`, keep it.
3. Otherwise, if a local branch or worktree for `XXX_*` already exists, switch to it (or use that worktree).
4. Otherwise create one from an up-to-date upstream `main` (`git fetch origin`, then branch from `origin/main`), either as a branch (`git switch -c XXX_short_description origin/main`) or as a sibling worktree (`git worktree add ../pyFAI-XXX_short_description -b XXX_short_description origin/main`). Ask the user which one they prefer if it is not clear from the context.
5. Never discard or carry over uncommitted work silently: if the working tree is dirty, tell the user before switching. Do not push, and do not touch other branches.

All following steps are done on that branch/worktree.

## 3. Regression test

- Add the test to the existing test module of the affected feature (`src/pyFAI/test/test_*.py`; `test_bug_regression.py` hosts the bug-specific tests; `src/pyFAI/gui/test/` and `src/pyFAI/opencl/test/` for GUI/OpenCL). Create a new module only if none fits; then register it in `src/pyFAI/test/meson.build` **and** in `src/pyFAI/test/test_all.py` (or in the `__init__.py` of the gui/opencl test directory) **and** perform a `git add` of the new module.
- Follow the style of the surrounding tests (`unittest`, `suite()` registration, `UtilsTest` helpers). Build the data in the test (numpy arrays, simulated calibrant image, temporary directory) rather than adding sample files; if a sample file is unavoidable, say so in `ISSUE_XXX.md` instead of committing a binary.
- Name the test after the issue, e.g. `test_bug_XXX`, and add a short docstring/comment mentioning `#XXX`.
- Derive it from the minimal reproducer of the `issue` skill: assert the **expected** behaviour, so the test fails today and passes once the bug is fixed.
- Run it with the `test` skill (single test; with `-o` unless OpenCL is involved) and confirm it **fails for the reason described in the issue** (not because of a typo or an import error). Run the surrounding module too, to make sure nothing else was broken.
- For a feature request, write the test that the requested feature must satisfy, and confirm it fails because the feature is missing.

## 4. `ISSUE_XXX.md`

Create `ISSUE_XXX.md` at the repository root. Do not commit or stage it: it is a scratch file, deleted once the issue is resolved.

Layout:

```markdown
# Issue #XXX: <title>

## Actions
- [ ] Issue read and reproduced
- [ ] Branch `XXX_short_description` in use
- [ ] Regression test `<path>::<test name>` added (fails before the fix)
- [ ] Changelog entry added
- [ ] Fix implemented
- [ ] `__date__` of the modified files updated
- [ ] Regression test passes
- [ ] Full test suite and ruff pass
- [ ] Pull request text written from this file
- [ ] This file deleted

## Summary
<2-3 lines: what is wrong, or what is requested>

## Reproducer
<the minimal snippet and its actual vs. expected output>

## Analysis
<suspected `path:line`, likely cause, confidence>

## Agents
< each agent should describe itself to know which model was used for each part of the work, one per line in a list>
- agent_name: LLM model used

## Notes
<comments left by whoever works on an item: date/agent name, the action concerned, what was done or found, dead ends. Append, do not rewrite others' notes.>
```

Add further checkbox items to the list when the analysis reveals more work (other engines affected, documentation or notebook to update, ...). Tick items as they are completed. Any agent working on the issue later reads this file first and records its progress in **Notes**.

## 5. Changelog

Add **one line** to `doc/source/changelog.rst` (the root `CHANGELOG.txt` only points to it), in the topmost section (`202X.YY`, the version under development), under the matching heading (`New features`, `Bug fixes`, `Documentation`, `Code quality`). If that section is missing, create it above the latest release with the same layout, and update the `:Date:` header.

- Entry format, as for the other lines: `  * Fix <what was wrong> (#XXX)` under `Bug fixes`, or `  * <Feature description> (#XXX)` under `New features`. A single line, no sub-bullets.
- Word it from the user's point of view. Re-read it when the fix is finished and adjust it if the real fix differs from what was planned.

## 6. Hand-over

Report: the branch/worktree, the test (path and failure message), the `ISSUE_XXX.md` content summary and the changelog line. Do not attempt the fix unless asked.

## When the issue is resolved (for whoever does the fix)

- Update the `__date__` header of every modified source file (see `AGENTS.md`).
- Tick the remaining items, with the `test` skill run the full suite and `pre-commit run ruff-check --all-files`.
- Write the pull request text from `ISSUE_XXX.md`, **as short as possible**: a one-line title, one to three lines of description (cause and fix), and `Closes #XXX`. No checklist, no agent notes, no test log.
- Then delete `ISSUE_XXX.md`.
