#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""This is a simple module to help searching for segmentation fault.
It works on any operating system but I needed it on MacOS-X as I was not
able to use GDB as on linux.

Usage python -m mactrace test.py

it prints all line number for any executed statement
"""

import sys, os
from optparse import OptionParser

class TraceWriter(object):
    def __init__(self, myFile=sys.stdout):
        self.file = myFile
    def trace(self, frame, event, arg):
        self.file.write("%s, %s:%d%s" % (event, frame.f_code.co_filename, frame.f_lineno, os.linesep))
        self.file.flush()
        return self.trace

def main():
    usage = "mactrace.py [-o output_file_path] scriptfile [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
                      help="Save trace to <outfile>", default=None)
    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args
    if options.outfile:
        twriter = TraceWriter(open(options.outfile, "w"))
    else:
        twriter = TraceWriter()
    sys.settrace(twriter.trace)
    if len(args) > 0:
        progname = args[0]
        sys.path.insert(0, os.path.dirname(progname))
        with open(progname, 'rb') as fp:
            code = compile(fp.read(), progname, 'exec')
        globs = {
            '__file__': progname,
            '__name__': '__main__',
            '__package__': None,
        }
        eval(code, globs)
    else:
        parser.print_usage()
    return parser

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()
