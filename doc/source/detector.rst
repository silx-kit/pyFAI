:Author: Jérôme Kieffer
:Date: 31/05/2016
:Keywords: detector
:Target: General audiance
Simple detector
===============

Like most other diffraction processing packages, pyFAI allows the definition of
2D detectors with a constant pixel size and recoded in S.I..
Typical pixel size are 50e-6 m (50 microns) and will be used as example in the
numerical application.

Pixels of the detector are indexed from the *origin* located at the **lower left corner**.
The pixel's center is located at half integer index:

* pixel 0 goes from position 0 m to 50e-6m and is centered at 25e-6 m.
* pixel 1 goes from position 50e-6 m to 100e-6 m and is centered at 75e-6 m

**Nota**:
Most of the time you will need to pass the optional argument *origin="lower"* to
matplotlib imshow function when displaying the image to avoid confusion

Complex detectors
=================

The *simple detector* approach reaches its limits
with several detector types, such as multi-module and fiber optic taper coupled
detectors (most CCDs).
Large area pixel detectors are often composed of smaller modules (i.e. Pilatus
from Dectris, Maxipix from ESRF, ...).

By construction, such detectors exhibit gaps between modules along with
pixels of various sizes within a single module, hence they require specific
data masks.
Optically coupled detectors need also to be corrected
for small spatial displacements, often called geometric distortion.
This is why detectors need more complex definitions than just that of a pixel
size.
To avoid complicated and error-prone sets of parameters, two tools have been introduced:
either *detector* classes define programatically detector or Nexus saved detector setup.

Detectors classes
-----------------
They are used to define families of detectors.
In order to take the specificities of each detector into account, pyFAI
contains about 55 detector class definitions (and twice a much with aliases)
which contain a mask (invalid pixels,
gaps, ...) and a method to calculate the pixel positions in Cartesian
coordinates. Available detectors can be printed using:

.. code-block:: python

   import pyFAI
   print(pyFAI.detectors.ALL_DETECTORS)

For optically coupled CCD detectors, the geometrical distortion is often
described by a bi-dimensional cubic spline which can be imported into
the detector instance and be used to calculate the actual pixel position in space.

Nexus Detectors
---------------

Any detector object in pyFAI, can be saved into a HDF5 file following the NeXus
convention (http://nexusformat.org).
Detector objects can subsequently be restored from the disk, making
complex detector definitions less error-prone.
Pixels of an area detector are saved as a 4D dataset: i.e. a 2D
array of vertices pointing to every corner of each pixel, generating
an array of shape: (*Ny*, *Nx*, *Nc*, 3) where *Nx* and *Ny* are the dimensions of the
detector, *Nc* is the number of corners of each pixel, usually 4, and the last
entry contains the coordinates of the vertex itself (z,y,x).
This kind of definitions, while relying on large description files,
can address some of the most complex detector layouts:

* hexagonal pixels (i.e. Pixirad detectors, still under development)
* curved/bent imaging plates (i.e. Rigaku, Aarhus detector)
* pixel detectors with tiled modular (i.e. Xpad detectors from ImXpad)
* semi-cylindrical pixel detectors (i.e. Pilatus12M from Dectris).

The detector instance can be saved as HDF5, either programmatically, either
on the command line.

.. code-block:: python

   from pyFAI import detectors
   frelon = detectors.FReLoN("halfccd.spline")
   print(frelon)
   frelon.save("halfccd.h5")

Using the *detector2nexus* script to convert a complex detector definition
(multiple modules, possibly in 3D) into
a single NeXus detector definition together with the mask:

  detector2nexus -s halfccd.spline -o halfccd.h5

Conclusion
==========

Detector definition in pyFAI is very versatile.
Fortunately, most detectors are already described, making the usage transparent
for most users.
There are a couple of :ref:`tutorials` on detector definition which will help
you understanding the mechanisme :

* Distortion which explains how to correct images for geometric distortion
* CCD-calibration which explains how to calibrate a detector for geometric
  distortion.

