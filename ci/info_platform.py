#!/usr/bin/env python
import sys, numpy
print("Python %s bits" % (tuple.__itemsize__ * 8))
print("       maxsize: %s\t maxunicode: %s" % (sys.maxsize, sys.maxunicode))
print(sys.version)
try:
    from distutils.sysconfig import get_config_vars
except:
    from sysconfig import get_config_vars
config = get_config_vars("CONFIG_ARGS")
print(config)
if config:
    print("Config :" + " ".join(config))
else:
    print("Config : None")
print("")
print("Numpy %s" % numpy.version.version)
print("      include %s" % numpy.get_include())
print("      options %s" % numpy.get_printoptions())
print("")
try:
    import pyopencl
except Exception as error:
    print("Unable to import pyopencl: %s" % error)
else:
    print("PyOpenCL platform:")
    for p in pyopencl.get_platforms():
        print("  %s" % p)
        for d in p.get_devices():
            print("    %s" % d)
