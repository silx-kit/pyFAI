# Contributing

When contributing to this repository, please first open an issue to discuss the change you wish to make.
Mention you wish to help us improving this specific part of the project, we will be able to provide you guidance.

If you have questions, the best is to subscribe to the associated mailing list. Direct contact with maintainers is
discouraged as we are few and may be unavailable (holidays, ...).

## Prepare your local working environment

1. Clone the master `git clone https://github.com/silx-kit/pyFAI -o upstream; cd pyFAI`
2. Add your github clone `git remote add origin git@github.com:account/pyFAI`
3. Create a virtual environment `python3 -m venv ~/.py3`
4. Activate your environment `source ~/.py3/bin/activate`
5. Install the dependencies `pip install --upgrade -r requirements.txt --only-binary :all:`
6. Build and test `python run_tests.py`

This should take a few minutes and ensures everything is ready for developping within pyFAI.
Later-on no recompilation will be needed unless you modify cython code.
In this case, recompilation can be accelerating by installing `ccache`.

## Pull Request Process

1. Ensure the idea is described in an issue before starting to code
2. Fork the main branch into a meaningful name (relative to the issue number or name)
3. Ensure your contributed code has an associated test, documentation and does not create regression (see: test locally your code)
4. Push your code and create a pull-request. This will trigger CI on all operating systems
5. A core developer may read your code and review it. The PR comments are used to discuss technically your PR's code

## Debug your code

You can easily test your code without installing it, thanks to the `bootstrap.py` tool.
By default, `./bootstrap.py` will launch an ipython console where `import pyFAI` will import your local pyFAI, modified by you.
`bootstrap.py` can also be used to launch any application provided in pyFAI, like `./bootstart pyFAI-benchmark`
or to run any third party application using pyFAI (when the full path is provided).
In one word, `bootstrap` is a great tool to help debugging, it re-compiles the code when needed but it is not perfect and corner cases probably exist.
Note: it is forbidden to import pyFAI from the sources, to avoid bugs as many files will be missing or mis-placed.

## Test locally your code

The test suite of pyFAI can simply be triggered by running `./run_tests.py` which
takes care of re-building what is needed.
This helper script has many options about coverage, selecting tests, debugging mode ...
use `./run_tests.py -h` to visualize them all.

Please note we have not yet decided for a code of conduct.
