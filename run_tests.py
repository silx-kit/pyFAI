#!/usr/bin/env python
import sys, numpy, scipy, fabio, pyFAI
print("Python %s %s" % (sys.version, tuple.__itemsize__ * 8))
print("Numpy %s" % numpy.version.version)
print("Scipy %s" % scipy.version.version)
print("FabIO %s" % fabio.version)
print("PyFAI %s" % pyFAI.version)
pyFAI.tests()
