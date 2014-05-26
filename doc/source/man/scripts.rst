pyFAI scripts manual
====================

While pyFAI is first and foremost a Python library to be used by developers, a set of scripts is provided to process
a full diffraction experiment on the command line without knowing anything about Python. 
Those scipts can be divided into 3 categories: pre-processing tools which prepare the dataset for the calibration tool.
The calibration is the determination of the geometry of the experimental setup using Debye-Scherrer rings of a reference compound (or calibrant).
Finally a full dataset can be integrated using different tools targeted at different experiments. 

Pre-processing tools:
 * drawMask_pymca: tool for drawing a mask on top of an image
 * pyFAI-average: tool for averaging/median/... filtering images (i.e. for dark current)

Calibration tools:
 * pyFAI-calib: manually select the rings and refine the geometry
 * pyFAI-recalib: automatic ring extraction to refine the geometry
 * MX-calibrate: Calibrate automatically a set of images taken at various detector distances
 * check_calib: checks the calibration of an image at the sub-pixel level

Azimuthal integration tools:
 * pyFAI-integrate: the only graphical interface for integration
 * pyFAI-saxs: command line interface for small-angle scattering
 * pyFAI-waxs: command line interface for powder difration
 * diff_tomo: diffraction mapping&tomography tool
 
.. toctree::
   :maxdepth: 4

   pyFAI-average
   drawMask_pymca
   MX-calibrate
   check_calib
   pyFAI-calib
   pyFAI-recalib
   pyFAI-integrate
   pyFAI-saxs
   pyFAI-waxs
   diff_tomo

