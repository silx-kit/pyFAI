#!/usr/bin/env python
import sys
import os
import logging
package = "fabio"

print("Python %s %s" % (sys.version, tuple.__itemsize__ * 8))

try:
    import numpy
except:
    print("Numpy missing")
else:
    print("Numpy %s" % numpy.version.version)

try:
    import scipy
except:
    print("Scipy missing")
else:
    print("Scipy %s" % scipy.version.version)

try:
    import fabio
except:
    print("FabIO missing")
else:
    print("FabIO %s" % fabio.version)

try:
    import h5py
except Exception as error:
    print("h5py missing: %s" % error)
else:
    print("h5py %s" % h5py.version.version)

try:
    import Cython
except:
    print("Cython missing")
else:
    print("Cython %s" % Cython.__version__)


def report_rst(cov, package="fabio", version="0.0.0", base=""):
    """
    Generate a report of test coverage in RST (for Sphinx includion)
    
    @param cov: test coverage instance
    @return: RST string
    """
    import tempfile
    fd, fn = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    cov.xml_report(outfile=fn)
    from lxml import etree
    xml = etree.parse(fn)
    classes = xml.xpath("//class")
    import time
    line0 = "Test coverage report for %s" % package
    res = [line0, "=" * len(line0), ""]
    res.append("Measured on *%s* version %s, %s" % (package, version, time.strftime("%d/%m/%Y")))
    res += ["",
            ".. csv-table:: Test suite coverage",
            '   :header: "Name", "Stmts", "Exec", "Cover"',
            '   :widths: 35, 8, 8, 8',
            '']
    tot_sum_lines = 0
    tot_sum_hits = 0
    for cl in classes:
        name = cl.get("name")
        fname = cl.get("filename")
        if not os.path.abspath(fname).startswith(base):
            continue
        lines = cl.find("lines").getchildren()
        hits = [int(i.get("hits")) for i in lines]

        sum_hits = sum(hits)
        sum_lines = len(lines)

        cover = 100.0 * sum_hits / sum_lines if sum_lines else 0

        res.append('   "%s", "%s", "%s", "%.1f %%"' % (name, sum_lines, sum_hits, cover))
        tot_sum_lines += sum_lines
        tot_sum_hits += sum_hits
    res.append("")
    res.append('   "%s total", "%s", "%s", "%.1f %%"' %
               (package, tot_sum_lines, tot_sum_hits, 100.0 * tot_sum_hits / tot_sum_lines))
    res.append("")
    return os.linesep.join(res)


try:
    from argparse import ArgumentParser
except ImportError:
    from fabio.third_party.argparse import ArgumentParser
parser = ArgumentParser(description='Run the tests.')

parser.add_argument("-i", "--insource",
                    action="store_true", dest="insource", default=False,
                    help="Use the build source and not the installed version")
parser.add_argument("-c", "--coverage", dest="coverage",
                    action="store_true", default=False,
                    help="report coverage of fabio code (requires 'coverage' module)")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="increase verbosity")
options = parser.parse_args()
sys.argv = [sys.argv[0]]
if options.verbose == 1:
    logging.root.setLevel(logging.INFO)
    print("Set log level: INFO")
elif options.verbose > 1:
    logging.root.setLevel(logging.DEBUG)
    print("Set log level: DEBUG")

if options.coverage:
    print("Running test-coverage")
    import coverage
    try:
        cov = coverage.Coverage(omit=["*test*", "*third_party*"])
    except AttributeError:
        cov = coverage.coverage(omit=["*test*", "*third_party*"])
    cov.start()


if not options.insource:
    try:
        import pyFAI
    except:
        print("pyFAI missing, using built (i.e. not installed) version")
        options.insource = True
if options.insource:
    home = os.path.abspath(__file__)
    sys.path.insert(0, home)
    from test.utilstest import *
    import pyFAI

print("pyFAI %s from %s" % (pyFAI.version, pyFAI.__path__[0]))
pyFAI.tests()


if options.coverage:
    cov.stop()
    cov.save()
    with open("coverage.rst", "w") as fn:
        fn.write(report_rst(cov, "PyFAI", pyFAI.version, pyFAI.__path__[0]))
    print(cov.report())
