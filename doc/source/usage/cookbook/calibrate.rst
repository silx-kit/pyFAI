Calibration of a diffraction setup
==================================

Author: Jérôme Kieffer

Date: 20/01/2015

Keywords: Installation procedure

Target: System administrators

Associated video:
  http://www.edna-site.org/pub/calibration/calib.flv

1. Review your calibration image
--------------------------------
As viewer, try fabio_viewer from the FabIO package
If you need to pre-process your data, look at pyFAI-average.
In this example we have used a "max filter" over 20 frames
using pyFAI-average.

2. Get all additional data
--------------------------

 * calibrant used, here LaB6
 * the energy or the wavelength
 * detector geometry
 * masks, ...

3. Start pyFAI-calib
--------------------

Use the man page (or --help) to see all options

4. Pick peaks
-------------

 * A few (5) points in the most inner
 * Increase the counter to indicate the ring number
 * Pick some extra point in outer ring
 * right click to pick a point !

5. Review the group of peaks
----------------------------

Press Enter to do so...
and check the ring assignment

The position of the expected rings is overlaid to the image
Unzoom to view them !

6. Acquire some more control points
-----------------------------------

 * Use the recalib to extract new data-points
 * free/fix/bound then refine

7. Visualize the integrated patterns
------------------------------------

 * integrate to view the integrated pattern
 * then extract a few extra rings ...
 * the geometry is displayed on the screen

8. Quit
-------

All different geometries have been saved into the .poni file.
It can directly be used for integration

That's all.


