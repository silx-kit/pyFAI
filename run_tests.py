#!/usr/bin/env python
import sys
print("Python %s %s" % (sys.version, tuple.__itemsize__ * 8))

try: import numpy
except: print("Numpy missing")
else: print("Numpy %s" % numpy.version.version)

try: import scipy
except: print("Scipy missing")
else: print("Scipy %s" % scipy.version.version)

try: import fabio
except: print("FabIO missing")
else: print("FabIO %s" % fabio.version)

try: import h5py
except as error: print("h5py missing: %s" % error)
else: print("h5py %s" % h5py.version.version)

try: import Cython
except: print("Cython missing")
else: print("Cython %s" % Cython.__version__)

try: import pyFAI
except: print("PyFAI missing")
else: print("PyFAI %s" % pyFAI.version)

import logging
if "-v" in sys.argv:
    logging.root.setLevel(logging.INFO)
if "-d" in sys.argv:
    logging.root.setLevel(logging.DEBUG)

pyFAI.tests()
