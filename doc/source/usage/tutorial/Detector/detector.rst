:Author: Jérôme Kieffer
:Date: 10/03/2020
:Keywords: Tutorials
:Target: Advanced users tutorials for dealing with geometry distortion of detectors

.. _detectors:

Detector geometric distortions and corrections
==============================================

Those tutorials deal with distortion of detectors also called geometric distortion.
The first tutorial is about correcting distorted images, which is straight forwards in pyFAI when the detector is known.
The subsequent ones are about measuring this distortion, often based on the image of a regular grid placed in front of the detector.

.. toctree::
   :maxdepth: 1

   Distortion/Distortion
   CCD_Calibration/CCD_calibration
   Pilatus_Calibration/Pilatus_ID15
   Pilatus_Calibration/Pilatus900kw-ID06
   Eiger_Calibration/Eiger2-ID11
