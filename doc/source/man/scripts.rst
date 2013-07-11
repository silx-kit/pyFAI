pyFAI scripts manual
====================

While pyFAI is first and foremost a Python library to be used by developers, a set of scripts is provided to process
a full diffraction experiment on the command line without knowing anything about Python:

 * pyFAI-average to merge a set of files like dark-current files or diffracton images using various filters
 * drawMask_pymca to mask out some region of the detector
 * pyFAI-calib to select the rings and refine the geometry
 * pyFAI-recalib with an automatic ring extraction followed by the refinement.
 * pyFAI-integrate offers a graphical interface to configure the integration of an experiment
 * pyFAI-waxs text interface for integration of an experiment in 2theta
 * pyFAI-saxs text interface for integration of an experiment in q-space

Few other scripts are also available, most of them are very specific to one experiment or are highly experimental.
 * diff_tomo is a tool to generate a 3D sinogram as an HDF5 dataset for a diffracton tomography experiment
 * check-calib is an experimental tool to validate a full calibration of the complete image (not only on the peaks)
 * MX-calibrate refines the calibration from a set of images and exports the parameters interpolated as function of the detector distance

.. toctree::
   :maxdepth: 4

   pyFAI-average
   drawMask_pymca
   pyFAI-calib
   pyFAI-recalib
   pyFAI-integrate
   pyFAI-saxs
   pyFAI-waxs
   diff_tomo
   MX-calibrate
   check_calib

