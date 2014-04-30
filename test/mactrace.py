# This is a simple module to help searching for segmentation fault.
# It works on any operating system but I needed it on MacOS-X as I was not
# able to use GDB as on linux.
#
# Usage python -m mactrace test.py
#
# it prints all line number for any executed statement
#

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
