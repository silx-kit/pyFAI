:Author: Jérôme Kieffer
:Date: 27/10/2015
:Keywords: Tutorials
:Target: Scientists

Calibration of a diffraction setup
==================================

`Associated video <http://www.edna-site.org/pub/calibration/calibration.flv>`_

Review your calibration image
-----------------------------
As viewer, try fabio_viewer from the FabIO package
If you need to pre-process your data, look at pyFAI-average.
In this example we have used a "max filter" over 20 frames
using pyFAI-average.

Get all additional data
-----------------------

 * calibrant used, here LaB6
 * the energy or the wavelength
 * detector geometry
 * masks, ...

Start pyFAI-calib
-----------------

Use the man page (or --help) to see all options

Pick peaks
----------

 * A few (5) points in the most inner
 * Increase the counter to indicate the ring number
 * Pick some extra point in outer ring
 * right click to pick a point !

Review the group of peaks
-------------------------

Press Enter to do so...
and check the ring assignment

The position of the expected rings is overlaid to the image
Un-zoom to view them !

Acquire some more control points
--------------------------------

 * Use the recalib to extract new data-points
 * free/fix/bound then refine

Visualize the integrated patterns
---------------------------------

 * integrate to view the integrated pattern
 * then extract a few extra rings ...
 * the geometry is displayed on the screen

Quit
----

All different geometries have been saved into the .poni file.
It can directly be used for integration

That's all.


