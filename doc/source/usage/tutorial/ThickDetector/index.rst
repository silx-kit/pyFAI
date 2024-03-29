:Author: Jérôme Kieffer
:Date: 09/06/2021
:Keywords: Tutorial
:Target: Scientists

.. _thick:

Thick detectors
===============

This is a long journey in the depth of the physics of the X-ray detector to
understand the effect of the thickness of the sensor layer on our data.
This study has been sponsored by ESRF's beamlines ID31 and ID22 with the contribution
from the SNBL and from the detector group.
Most images were taken at ESRF-ID28 on a Pilatus 1M, mounted on a goniometer stage.
Thus I would like to thank Veijo Honkimaki for the subject and Thanh Tra Nguyen
for the images.

*Disclaimer:* The Pilatus detector is a photon counting detector with a threshold
in energy.
In this demonstration, we consider only the photo-electric effect and consider
the full absorption of a photon if it interacts with the sensor material.

This document first presents an experiment on ID28 where the detector position
is calibrated on its goniometer stage and highlights some odd behavior at high
incidence angle where rings looks shifted and broadened and could be attributed to
"thickness effect".
In a second time, the detector is modeled as a 2D array of voxels, hence with
some thickness, and the signal is then deconvolved to invert this effect.
Finally the corrected images are used to validate the correction.

Another approach is presented where the *parallax* is modeled as a convolution of the beam profile
with the absorption profile, leading to some numerical correction to apply to the position of absorption (WIP).


.. toctree::
   :maxdepth: 1

   goniometer_id28
   raytracing
   deconvolution
   Parallax_model
