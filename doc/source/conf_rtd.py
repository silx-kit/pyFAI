#!/usr/bin/env python
# coding: utf-8
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

""

import os
import glob
import sys
root_dir = os.path.abspath("../..")
build_dir = glob.glob('../../build/lib*')


# Build pyFAI if it is not yet built (especially for readthedocs)
if (not build_dir) or ("__init__.py" not in os.listdir(os.path.join(build_dir[0], "pyFAI"))):
    import subprocess
    curr_dir = os.getcwd()
    os.chdir(root_dir)
    errno = subprocess.call([sys.executable, 'setup.py', 'build'])
    if errno != 0:
        raise SystemExit(errno)
    else:
        os.chdir(curr_dir)
    build_dir = glob.glob('../../build/lib*')

if "" not in sys.path:
    sys.path.insert(0, "")

from conf import *
