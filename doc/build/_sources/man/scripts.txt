:Author: Jérôme Kieffer
:Date: 31/05/2016
:Keywords: manpage
:Target: Scientists

.. _manpage:

pyFAI scripts manual
====================

While pyFAI is first and foremost a Python library to be used by developers, a set of scripts is provided to process
a full diffraction experiment on the command line without knowing anything about Python.
Those scripts can be divided into 3 categories: pre-processing tools which prepare the dataset for the calibration tool.
The calibration is the determination of the geometry of the experimental setup using Debye-Scherrer rings of a reference compound (or calibrant).
Finally a full dataset can be integrated using different tools targeted at different experiments.

Pre-processing tools:
 * drawMask_pymca: tool for drawing a mask on top of an image
 * pyFAI-average: tool for averaging/median/... filtering images (i.e. for dark current)

Calibration tools:
 * pyFAI-calib: manually select the rings and refine the geometry
 * pyFAI-recalib: automatic ring extraction to refine the geometry (deprecated: see "recalib" option in pyFAI-calib)
 * MX-calibrate: Calibrate automatically a set of images taken at various detector distances
 * check_calib: checks the calibration of an image at the sub-pixel level (deprecated: see "validate" option in pyFAI-calib)

Azimuthal integration tools:
 * pyFAI-integrate: the graphical interface for integration (GUI)
 * pyFAI-saxs: command line interface for small-angle scattering
 * pyFAI-waxs: command line interface for powder difration
 * diff_map: diffraction mapping & tomography tool (command line and GUI)
 * diff_tomo: diffraction tomography tool (command line only)

.. toctree::
   :maxdepth: 1

   pyFAI-average
   drawMask_pymca
   detector2nexus
   pyFAI-calib
   pyFAI-recalib
   check_calib
   MX-calibrate
   pyFAI-integrate
   diff_map
   diff_tomo
   pyFAI-saxs
   pyFAI-waxs

