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

# Few common rules to make common development simpler:
* Code formatting is generally PEP8, except for the GUI section where the CamelCase is used (due to inheritance of Qt classes).
* Doc-strings are mandatory. They should contain the purpose of the function and the description of its signature, both input and output, for human beings. Discussion about the implementation are best done in comments.
* Typing is accepted but not enforced. Typing is NOT an excuse for the absence of documentation. It should neither reduce readability nor break tests on any platform/version of python.
* Minimal code formatters are in place as part of the CI procedure using `pre-commit`. More aggressive formatters (`black`, `blue`, `pyink`, `ruff` to cite a few) are only allowed when used by the main author of a file or as part of a refactoring duly accepted. In this case, `ruff` is the preferred tool. This means a pull request with just running `black` on the entire repository will be rejected.

If you encounter any issue at this level, please contact the upstream authors for guidance.
The code of conduct of the project is described in the CODE_OF_CONDUCT.md file.

Above all, the primary goal of this project is to ensure numerically correct data reduction. We are committed to upholding the highest standards of scientific and technical accuracy in our codebase. At the same time, we recognize the importance of fostering a collaborative environment where junior developers are encouraged and supported to grow their skills. We actively invest in mentoring and training, enabling contributors at all levels to reach the expertise required to maintain and advance the quality of our software.
