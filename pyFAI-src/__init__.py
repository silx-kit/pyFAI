version = "0.9.2"
date = "2014-01"
import sys, logging
logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("pyFAI.__init__")
    logger.error("pyFAI required a python version >= 2.6")
    raise RuntimeError("pyFAI required a python version >= 2.6, now we are running: %s" % sys.version)
from azimuthalIntegrator import AzimuthalIntegrator
load = AzimuthalIntegrator.sload
