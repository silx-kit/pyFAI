:Author: Marco Cammarata
:Date: 22/11/2022
:Keywords: description of conventions used by pyFAI

Conventions
===========

This page describes the conventions used by pyFAI.
Some useful information (orientation, position of observer, etc.) can be found in :ref:`geometry`

Mask
----

PyFAI considers masks with values equal to zero 0 as valid pixels (mnemonic: non zero pixels are *masked out*).
This is a different convention with respect to other programs used at ESRF like Lima that used no zero pixels as valid.


Pixel Coordinates
-----------------

The origin is set at the corner of a pixel as shown in the figure below for a simple 3×4 pixel matrix.

.. figure:: img/pixel_coordinates.svg
   :align: center
   :alt: Position of the origin with respect to the pixel matrix

Note some specificities:

* Each pixel *n* starts at the coordinated *n* (included) and goes to the coordinate *n+1* (excluded). The center of any pixel is at half integer pixel coordinate. This convention differs by half a pixel from the one used in matplotlib where pixels range from *n-½* to *n-½*. Care must be taken when displaying images with matplotlib (when assessing beam-center for example): there is ½ pixel offset.
* The origin is at the bottom and differs from the *camera* convention where the origin is at the top. As a consequence, the sign of the ᵪ-angle gets  inverted.
* The detector is seen from the sample and differs from the *camera* convention where the observer is behind the camera. As a consequence, the sign of the ᵪ-angle gets  inverted.
* The former 2 inversions cancel each other.
