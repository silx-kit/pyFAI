# AGENT.md

## Project structure

This Python project has to be compiled before being tested or executed.
One should work in a virtual environment:
```
python -m venv myenv
. myenv/bin/activate
pip install . --upgrade
```
One cannot simply append `src` to the PYTHONPATH.
For simple tries, it is possible to run:
`python bootstrap.py` to launch a python interpreter with the PATH properly setup

## Testing
Tests can be run with `python ./run-tests.py` which will compile a local version of extensions, and run all tests within this local installation.

## Docstrings
All functions/classes require a proper docstring. 
Typing is a bonus but not mandatory.

## Role
You are a senior developer. 
Any modification you are requesting should be adequately tested and validated. 
