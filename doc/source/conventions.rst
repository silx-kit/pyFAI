:Author: Marco Cammarata
:Date: 22/11/2022
:Keywords: description of conventions used by pyFAI

Conventions
===========

This page describes the conventions used by pyFAI.
Some useful information (orientation, position of observer, etc.) can be found in :res:`geometry`

Mask
----

PyFAI considers masks with values equal to zero 0 as valid pixels (mnemonic: non zero pixels are *masked out*).
This is a different convention with respect to other programs used at ESRF like Lima that used no zero pixels as valid.


Pixel Coordinates
-----------------

The origin is set at the corner of a pixel as shown in the figure below for a simple 3×4 pixel matrix.
Note that the center of each pixel has fractional pixel coordinate.

.. figure:: img/pixel_coordinates.svg
   :align: center
   :alt: Position of the origin with respect to the pixel matrix
