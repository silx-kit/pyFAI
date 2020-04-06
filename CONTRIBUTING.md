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
6. Build and test `python setup.py build test`

This should take a couple of minutes and ensures everything is ready for developping within pyFAI.
Compilation steps can be accelerating by installing and configuring `ccache`.

## Pull Request Process

1. Ensure the idea is described in an issue before starting to code
2. Fork the master branch into a meaningfull name (relative to the issue number or name)
3. Ensure your contributed code has an associated test, documentation and does not create regression (see: test locally your code)
4. Push your code and create pull-request. This will trigger CI on all operating systems
5. A core developer may read your code and review it. The PR comments are used to discuss technically your PR's code

## Debug your code

You can easily test your code without installing it thanks to the `./bootstrap.py` tool that 
will launch a ipython console where an `import pyFAI` will import the local pyFAI.
Note: it is forbidden to import pyFAI from the sources, to avoid bugs.
It can can also launch any application related to pyFAI, like `./bootstart pyFAI-benchmark`.
`bootstrap.py` is a great tool to help debugging but it is not perfect and corner cases probably exists.

## Test locally your code 

The test suite of pyFAI can simply be triggered by running `./run_tests.py`. 
This helper script has many option about coverage, selecting tests, debuging mode ...
use `./run_tests.py -h` to 


Please note we have not yet decided for a code of conduct.

