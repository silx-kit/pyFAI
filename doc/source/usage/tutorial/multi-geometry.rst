:Author: Jérôme Kieffer
:Date: 27/10/2015
:Keywords: Tutorials
:Target: Advanced users

====================================
Multi-geometry azimuthal integration
====================================

Or how it is possible to perform an azimuthal regrouping using multiple detectors
or by moving a single (small) detector to cover more solid angle.

Idea
====

Azimuthal integration or azimuthal regrouping is (roughly) the averaging of all
pixel intensities with the same Q value (or 2theta), as described in
`this publication <http://arxiv.org/pdf/1412.6367v1.pdf>`_ chapter 3.2 and 3.3.

By taking multiple images at various places in space one covers more solid angle,
allowing either a better statistics or a larger Q-range coverage.

As described in the publication, the average is calculated by the ratio of the
(intensity-) weighted histogram by the unweighted histogram. By enforcing a same
bin position over multiple geometries, one can create a combined weighted and
unweighted histograms by simply summing all partial histograms from each geometry.

The resulting pattern is obtained as usual by the ration of weighted/unweighted

How it works
============

Lets assume you are able to know where your detector is in space, either
calibrated, either calculated from the goniometer position.
A diffrection image (img_i) has been acquired using a geometry which is stored
in a poni-file (poni_i) useable by pyFAI.

To define a multi-geometry integrator, one needs all poni-files and one needs
to define the output space so that all individual integrators use the same bins.

.. code:: python

   import glob
   import fabio
   from pyFAI.multi_geometry import MultiGeometry

   img_files = glob.glob("*.cbf")
   img_data = [fabio.open(i).data for i in img_files]
   ais = [i[:-4]+".poni" for i in img_files]
   mg = MultiGeometry(ais, unit="q_A^-1", radial_range=(0, 50), wavelength=1e-10)
   q, I = mg.integrate1d(img_data, 10000)

What is automatic
-----------------

* MultiGeometry takes care of defining the same output space with the same bin position,
* It performs the normalization of the solid angle (absolute solid angle, unlike AzimuthalIntegrator !)
* It can handle polarization correction if needed
* It can normalize by a monitor (I1 normalization) or correct for exposure time

What is not
-----------

For PDF measurement, data needs to be properly prepared, especially:

* Dark current subtraction
* Flat-field correction
* Exposure time correction (if all images are not taken with the same exposure time)

Examples
========

.. toctree::
   :maxdepth: 2

   MultiGeometry/index

Conclusion
==========

MultiGeometry is a unique feature of PyFAI ...
While extremely powerful, it need
careful understanding of the numerical treatement going on underneath.
