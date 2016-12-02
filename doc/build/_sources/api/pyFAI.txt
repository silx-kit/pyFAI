
:mod:`pyFAI` Package
--------------------

.. automodule:: pyFAI.__init__
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`average` Module
---------------------

.. automodule:: pyFAI.average
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`azimuthalIntegrator` Module
---------------------------------

.. automodule:: pyFAI.azimuthalIntegrator
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`multi_geometry` Module
----------------------------

.. automodule:: pyFAI.multi_geometry
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`integrate_widget` Module
------------------------------

.. automodule:: pyFAI.integrate_widget
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`geometry` Module
----------------------

.. automodule:: pyFAI.geometry
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`geometryRefinement` Module
--------------------------------

.. automodule:: pyFAI.geometryRefinement
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`detectors` Module
-----------------------

.. automodule:: pyFAI.detectors
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`spline` Module
--------------------

.. automodule:: pyFAI.spline
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`opencl` Module
--------------------

.. automodule:: pyFAI.opencl
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ocl_azim` Module
----------------------

.. automodule:: pyFAI.ocl_azim
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ocl_azim_lut` Module
--------------------------

.. automodule:: pyFAI.ocl_azim_lut
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ocl_azim_csr` Module
--------------------------

.. automodule:: pyFAI.ocl_azim_csr
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ocl_azim_csr_dis` Module
------------------------------

.. automodule:: pyFAI.ocl_azim_csr_dis
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`io` Module
----------------

.. automodule:: pyFAI.io
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`calibration` Module
-------------------------

.. automodule:: pyFAI.calibration
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`peak_picker` Module
-------------------------

.. automodule:: pyFAI.peak_picker
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`massif` Module
--------------------

.. automodule:: pyFAI.massif
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`blob_detection` Module
----------------------------

.. automodule:: pyFAI.blob_detection
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`calibrant` Module
-----------------------

.. automodule:: pyFAI.calibrant
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`distortion` Module
------------------------

.. automodule:: pyFAI.distortion
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`worker` Module
--------------------

.. automodule:: pyFAI.worker
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`units` Module
-------------------

.. automodule:: pyFAI.units
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`utils` Module
-------------------

.. automodule:: pyFAI.utils
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`gui.gui_utils` Module
---------------------------

.. automodule:: pyFAI.gui.utils
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.bilinear` Module
--------------------------

This extension makes a discrete 2D-array appear like a continuous function thanks
to bilinear interpolations.

.. automodule:: pyFAI.ext.bilinear
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`ext._bispev` Module
-------------------------

This extension is a re-implementation of bi-cubic spline evaluation from scipy

.. automodule:: pyFAI.ext._bispev
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`ext._blob` Module
-----------------------

Blob detection is used to find peaks in images by performing subsequent blurs

.. automodule:: pyFAI.ext._blob
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.container` Module
---------------------------

Container are a new uniform storage, optimized for the creation of both LUT and CSR.
It has nothing to do with Docker.

.. automodule:: pyFAI.ext.container
    :members:
    :undoc-members:
    :show-inheritance:

   
:mod:`ext._convolution` Module
------------------------------

Convolutions in real space are used to blurs images, used in blob-detection algorithm

.. automodule:: pyFAI.ext._convolution
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext._distortion` Module
-----------------------------

Distortion correction are correction are applied by Look-up table (or CSR)

.. automodule:: pyFAI.ext._distortion
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext._geometry` Module
---------------------------

This extension is a fast-implementation for calculating the geometry, i.e. where
every pixel of an array stays in space (x,y,z) or its (r, \chi) coordinates.

.. automodule:: pyFAI.ext._geometry
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.histogram` Module
---------------------------

Re-implementation of the numpy.histogram, optimized for azimuthal integration.
Deprecated, will be replaced by silx.math.histogramnd

.. automodule:: pyFAI.ext.histogram
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.marchingsquares` Module
---------------------------------

The marchingsquares algorithm is used for calculating an iso-contour curve (displayed
on the screen while calibrating) but also to seed the points for the "massif" algoritm
during recalib.

.. automodule:: pyFAI.ext.marchingsquares
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.morphology` Module
----------------------------

The morphology extension provides a couple of binary morphology operations on images.
They are also implemented in scipy.ndimage in the general case, but not as fast.

.. automodule:: pyFAI.ext.morphology
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`ext.reconstruct` Module
-----------------------------

Very simple inpainting module for reconstructing the missing part of an image (masked)
to be able to use more common algorithms.

.. automodule:: pyFAI.ext.reconstruct
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`ext.relabel` Module
-------------------------

Relabel regions, used to flag from largest regions to the smallest

.. automodule:: pyFAI.ext.relabel
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`ext.preproc` Module
-------------------------

Contains a preprocessing function in charge of the dark-current subtraction,
flat-field normalization, ... taking care of masked values and normalization.

.. automodule:: pyFAI.ext.preproc
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`ext._tree` Module
-----------------------

The tree is used in file hierarchy tree for the diff_map graphical user interface.

.. automodule:: pyFAI.ext._tree
    :members:
    :undoc-members:
    :show-inheritance:
        
:mod:`ext.watershed` Module
---------------------------

Peak peaking via inverse watershed for connecting region of high intensity

.. automodule:: pyFAI.ext.watershed
    :members:
    :undoc-members:
    :show-inheritance:
        
        