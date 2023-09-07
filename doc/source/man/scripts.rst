:Author: Jérôme Kieffer
:Date: 30/10/2020
:Keywords: manpage
:Target: Scientists

.. _manpage:

Application manuals
===================

While pyFAI was first developped as a Python library to be used by developers, it evolved
with applications allowing to analyse a full diffraction experiment without knowing anything about Python.
Those scripts can be divided into 3 categories:

 - Pre-processing tools which prepare a dataset for the calibration tool, i.e. produce one image suitable for calibration.

 - Calibration tools which aim at the determination of the geometry of the experimental setup using Debye-Scherrer rings
   of a reference compound (or calibrant). They produce a `PONI-file` which contains this geometry

 - Integration tools which can reduce a full dataset using 1d or 2d integration, optionnally rebuiling images.

Pre-processing tools:
 - ``pyFAI-average``: tool for averaging/median/... filtering stacks of images (i.e. for dark current)

Calibration tools:
 - ``pyFAI-calib2``: manually select the rings and refine the geometry

Azimuthal integration tools:
 - ``pyFAI-integrate``: the graphical interface for integration (GUI)
 - ``diff_map``: diffraction imaging tool (command line and GUI)

.. toctree::
   :maxdepth: 1

   pyFAI-average
   pyFAI-calib2
   pyFAI-integrate
   diff_map


Specific tools and deprecated scripts
-------------------------------------

Those tools are not recommended for general purpose use,
some of them are deprecated, most of them see little maintenance
and remain available only for compatibility reasons.

.. toctree::
   :maxdepth: 1

   pyFAI-drawmask
   detector2nexus
   pyFAI-calib
   pyFAI-recalib
   check_calib
   MX-calibrate
   diff_tomo
   pyFAI-saxs
   pyFAI-waxs
   pyFAI-benchmark
   sparsify-Bragg
   peakfinder
