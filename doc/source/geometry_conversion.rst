:Author: Carsten Detlefs
:Date: 16/04/2019
:Keywords: geometry conversion from/to pyFAI
:Target: developers

.. _geometryconversion:

The purpose of this note is to compare how ``pyFAI`` and ``ImageD11``
treat the detector position. In particular, we derive how “PONI”
detector parameters refined with ``pyFAI`` can be transformed into
``ImageD11`` parameters.

In both packages, the transformation from pixel space to 3D laboratory
coordinates is carried out in 4 steps:

-  Transformation from “pixel space” to the “detector coordinate
   system”. The detector coordinate system is a 3D coordinate system
   centered on the (0,0) pixel of the detector.

-  Correction for linear offsets, i.e. the position of the (0,0) pixel
   relative to the beam axis.

-  Correction for the origin/diffractometer-to-detector distance. The
   sample and diffractometer center of rotation are assumed to be
   located at the origin.

-  A series of rotations for the detector coordinate system relative to
   the laboratory coordinates.

Unfortunately, the conventions chosen by ``pyFAI`` and ``ImageD11``
differ. For example, ``pyFAI`` applies the origin-to-detector distance
correction before rotations, whereas ``ImageD11`` applies it after
rotations. Furthermore, they employ different coordinate systems.

Detector
========

We consider a pixelated 2D imaging detector. In “pixel space”, the
position of a given pixel is given by the horizontal and vertical pixel
numbers, :math:`d_H` and :math:`d_V`. We assume that looking along the
beam axis into the detector, :math:`d_H` increases towards the right
(towards the center of the synchrotron) and :math:`d_V` towards the top.
For clarity, we assign the unit :math:`\mathrm{px}` to these
coordinates.

The pixel numbers :math:`d_H` and :math:`d_V` are transformed into 3D
“detector” coordinates by a function :math:`D`:

.. math::

   \begin{aligned}
     \vec{p}
     & =
     D\left(d_H, d_V\right).\end{aligned}

This function will account for the detector’s pixel size and the
orientation and direction of pixel rows and columns relative to the
detector coordinate system. Furthermore it may apply a distortion
correction. This, however, is beyond the scope of this note.

Limiting ourselves to linear functions, :math:`D` takes the form of a
matrix with two columns and three rows. We will see below that the
different choices of laboratory coordinate systems yield different
:math:`D`-matrices for ``pyFAI`` and ``ImageD11``. We assume that the
pixels have a constant horizontal and vertical size,
:math:`\mathrm{pxsize}_H` and :math:`\mathrm{pxsize}_V`. Both are given
in units of length per pixel. ``pyFAI`` specifically defines the unit of
length as meter, we will therefore use pixel sizes in units of
:math:`\mathrm{m}/\mathrm{px}` throughout this note.

The position and orientation of this detector relative to the laboratory
coordinates are described below.

Geometry definition of ``pyFAI``
================================

Coordinates
-----------

``pyFAI`` uses a coordinate system where the first axis (1) is
vertically up (:math:`y`), the second axis (2) is horizontal (:math:`x`)
towards the ring center (starboard), and the third axis (3) along the
beam (:math:`z`). Note that in this order (1, 2, 3) is a right-handed
coordinate system, which makes :math:`xyz` in the usual order a
left-handed coordinate system!

Units
-----

All dimensions in ``pyFAI`` are in meter and all rotation are in
radians.

Parameters
----------

``pyFAI`` describes the position and orientation of the detector by six
variables, collectively called the PONI, for point of normal incidence.
In addition, a detector calibration is provided in the PONI-file to
convert pixel coordinates into real-space coordinates. Here we limit our
discussion to the simplest case, i.e. a pixel size as discussed above.

Rotations:
    :math:`\theta_1`, :math:`\theta_2` and :math:`\theta_3` describe the
    detector’s orientation relative to the laboratory coordinate system.

Offsets:
    :math:`\mathrm{poni}_1` and :math:`\mathrm{poni}_2` describe the
    offsets of pixel (0,0) relative to the “point of normal incidence”.
    In the absence of rotations the point of normal incidence is defined
    by the intersection of the direct beam beam axis with the detector.

Distance:
    :math:`L` describes the distance from the origin of the laboratory
    system to the point of normal incidence.

Detector
--------

The transformation from pixel space to ``pyFAI`` detector coordinates is
given by

.. math::

   \begin{aligned}
     \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix}
     & =
     \begin{bmatrix}
       0 & \mathrm{pxsize}_V \\
       \mathrm{pxsize}_H & 0 \\
       0 & 0
     \end{bmatrix}
     \cdot
     \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     \\
     D_{\mathtt{pyFAI}}
     & = 
     \begin{bmatrix}
       0 & \mathrm{pxsize}_V \\
       \mathrm{pxsize}_H & 0 \\
       0 & 0
     \end{bmatrix}.
     \label{eq-dmatrixpyFAI}\end{aligned}

Offsets
-------

The PONI parameters are: a distance :math:`L`, the vertical (:math:`y`)
and horizontal (:math:`x`) coordinates of the point of normal incidence
in meters, :math:`\mathrm{poni}_1` and :math:`\mathrm{poni}_2`. The
inversion of the :math:`x` and :math:`y` axes is due to the arrangement
of the detector data, with :math:`x`-rows being the “slow” axis and
:math:`y`-columns the “fast” axis. Extra care has to be taken with the
signs of the rotations when converting form this coordinate system to
another.

``pyFAI`` applies both the offset correction and the origin-to-detector
distance after the transformation from pixel space to the detector
system, but before rotations,

Let :math:`L` be the distance from the origin/sample/diffractometer
center of rotation. In the absence of any detector rotations, :math:`L`
is taken along :math:`p_3` (beam axis, :math:`z`), :math:`p_1` along the
:math:`y`-axis (vertical) and :math:`p_2` along the :math:`x`-axis
(horizontal). Then the laboratory coordinates before rotation are

.. math::

   \begin{aligned}
     \begin{bmatrix}
       p_1 \\ p_2 \\ p_3
     \end{bmatrix}
     & =
     D_{\mathtt{pyFAI}} \cdot \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     +
     \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}.\end{aligned}

Rotations
---------

The detector rotations are taken about the origin of the coordinate
system (sample position). We define the following right-handed rotation
matrices:

.. math::

   \begin{aligned}
     \mathrm{R}_1(\theta_1)
     & =
     \begin{bmatrix}
       1 & 0 & 0 \\
       0 & \cos(\theta_1) & -\sin(\theta_1) \\
       0 & \sin(\theta_1) & \cos(\theta_1)
     \end{bmatrix}
     \label{eq-rot1}
     \\
     \mathrm{R}_2(\theta_2)
     & =
     \begin{bmatrix}
       \cos(\theta_2) & 0 & \sin(\theta_2) \\
       0 & 1 & 0 \\
       -\sin(\theta_2) & 0 & \cos(\theta_2)
     \end{bmatrix}
     \label{eq-rot2}
     \\
     \mathrm{R}_3(\theta_3)
     & =
     \begin{bmatrix}
       \cos(\theta_3) & -\sin(\theta_3) & 0\\
       \sin(\theta_3) & \cos(\theta_3) & 0\\
       0 & 0 & 1  
     \end{bmatrix}.
     \label{eq-rot3}\end{aligned}

The rotations 1 and 2 in ``pyFAI`` are left handed, i.e. the sign of
:math:`\theta_1` and :math:`\theta_2` is inverted.

The combined ``pyFAI`` rotation matrix is then

.. math::

   \begin{aligned}
     R_{\mathtt{pyFAI}}(\theta_1, \theta_2, \theta_3)
     & =
     R_3(\theta_3) \cdot R_2(-\theta_2) \cdot R_1(-\theta_1)\end{aligned}

which yields the final laboratory coordinates after rotation

.. math::

   \begin{aligned}
     \begin{bmatrix}
       t_1 \\ t_2 \\ t_3
     \end{bmatrix}
     & =
     R_{\mathtt{pyFAI}}(\theta_1, \theta_2, \theta_3)
     \cdot
     \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix}
     \label{eq-tpyFAI}
     \\
     & =
     R_{\mathtt{pyFAI}}(\theta_1, \theta_2, \theta_3)
     \cdot
     \left(
     D_{\mathtt{pyFAI}} \cdot \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     + \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}
     \right)
     \\
     & =
     R_{\mathtt{pyFAI}}(\theta_1, \theta_2, \theta_3)
     \cdot
     \left(
     \begin{bmatrix}
       0 & \mathrm{pxsize}_V \\
       \mathrm{pxsize}_H & 0 \\
       0 & 0
     \end{bmatrix}
     \cdot \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     + \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}
     \right).\end{aligned}

Inversion: Finding where a scattered beam hits the detector
-----------------------------------------------------------

For a 3DXRD-type simulation, we have to determine the pixel where a
scattered ray intercepts the detector. Let :math:`A` be the scattering
center of a ray within a sample volume (grain, sub-grain or voxel). The
Bragg condition and grain orientation pre-define the direction of the
scattered beam, :math:`\vec{k}`. The coordinates :math:`A_{1,2,3}` and
:math:`k_{1,2,3}` are specified in the laboratory system.

The inversion eq. [eq-tpyFAI] is straight-forward:

.. math::

   \begin{aligned}
     R_1(\theta_1)\cdot R_2(\theta_2) \cdot R_3(-\theta_3) \cdot
     \begin{bmatrix} t_1 \\ t_2 \\ t_3
     \end{bmatrix}
     & =
     \begin{bmatrix} p_1 \\ p_2 \\ L \end{bmatrix}
     \label{eq-find-alpha}
     \\
     \begin{bmatrix}
       t_1 \\ t_2 \\ t_3
     \end{bmatrix}
     & =
     \begin{bmatrix}
       A_1  \\ A_2 \\ A_3 
     \end{bmatrix}
     + \alpha
     \begin{bmatrix}
       k_1 \\ k_2 \\ k_3
     \end{bmatrix}.\end{aligned}

The third line (:math:`\ldots = L`) of eq. [eq-find-alpha] is then used
to determine the free parameter :math:`\alpha`, which in turn is used in
the first and second lines to find :math:`p_{1,2}` and thus
:math:`d_{1,2}`.

As the most trivial example we consider the case of no rotations,
:math:`\theta_1 = \theta_2 = \theta_3 = 0`. Then

.. math::

   \begin{aligned}
     A_3 + \alpha k_3 & = L \\
     \alpha & = \frac{L-A_3}{k_3} \\
     p_1 & = A_1 + (L-A_3) \frac{k_1}{k_3} \\
     p_2 & = A_2 + (L-A_3) \frac{k_2}{k_3}.\end{aligned}

We see also that when all rotations are zero, :math:`(\mathrm{poni}_1,
\mathrm{poni_2})` are the real space coordinates of the direct beam
(:math:`A_{1,2,3}=k_{1,2}=0`) .

Geometry definition of ``ImageD11``
===================================

For maximum convenience, ``ImageD11`` defines almost everything
differently than ``pyFAI``.

Coordinates
-----------

``ImageD11`` uses the ID06 coordinate system with :math:`x` along the
beam, :math:`y` to port (away from the ring center), and :math:`z` up.

Units
-----

As the problem is somewhat scale-invariant, ``ImageD11`` allows a free
choice of the unit of length, which we will call :math:`X` here. The
same unit has to be used for all translations, and for the pixel size of
the detector. The default used in the code appears to be
:math:`X = 1\,\mathrm{\mu m}`, but it might as well be Planck lengths,
millimeters, inches, meters, tlalcuahuitl, furlongs, nautical miles,
light years, kparsec, or whatever else floats your boat. The only
requirement is that you can actually measure and express the detector
pixel size and COR-to-detector distance in your units of choice. Since
we want to compare to ``pyFAI``, we choose :math:`X=1\,\mathrm{m}`.

Rotations are given in radians.

Parameters
----------

``ImageD11`` defines the detector geometry via the following parameters:

Beam center:
    :math:`y_{\mathrm{center}}` and :math:`z_{\mathrm{center}}` define
    the position of the direct beam on the detector. Contrary to
    ``pyFAI``, the beam center is given in pixel space, in units of
    :math:`\mathrm{px}`.

Pixel size:
    The horizontal and vertical pixel size are defined by
    :math:`y_{\mathrm{size}}` and :math:`z_{\mathrm{size}}` in
    :math:`{X}/{\mathrm{px}}`. With the right choice of the unit of
    length :math:`X`, these corresponds directly to the pixel sizes
    :math:`\mathrm{pxsize}_H` and :math:`\mathrm{pxsize}_V` defined
    above.

Detector flip matrix:
    :math:`O = \begin{bmatrix} o_{11} & o_{12} \\ o_{21} & o_{22} \end{bmatrix}`.
    This matrix takes care ofcorrecting typical problems with the way pixel coordinates are
    arranged on the detector. If, e.g., the detector is rotated by
    :math:`90^{\circ}`, then
    :math:`O=\begin{bmatrix} 0 & 1 \\ -1 & 0\end{bmatrix}`.
    If left and right (or up and down) are inverted
    on the detector, then :math:`o_{22} = -1` (:math:`o_{11}=-1`).

Rotations:
    Detector tilts :math:`t_x`, :math:`t_y`, and :math:`t_z`, in
    :math:`\mathrm{rad}`. The center of rotation is the point where the
    direct beam intersects the detector.

Distance:
    :math:`\Delta`, in units :math:`X`, is the distance between the
    origin to the point where the direct beam intersects the detector.
    Note that this is again different from the definition of ``pyFAI``.

It appears that these conventions where defined under the assumption
that the detector is more or less centered in the direct beam, and that
the detector tilts are small.

Transformation
--------------

The implementation in the code ``transform.py`` is using the following
equations:

.. math::

   \begin{aligned}
     R_{\mathtt{ImageD11}}(\theta_x, \theta_y, \theta_z)
     & =
     R_1(\theta_x) \cdot R_2(\theta_y) \cdot R_3(\theta_z)
     \\
     \begin{bmatrix}
       p_z \\ p_y
     \end{bmatrix}
     & =
     \begin{bmatrix}
       o_{11} & o_{12}
       \\ o_{21} & o_{22}
     \end{bmatrix}
     \cdot
     \begin{bmatrix}
       (d_z - z_{\mathrm{center}}) z_{\mathrm{size}} \\
       (d_y - y_{\mathrm{center}}) y_{\mathrm{size}}
     \end{bmatrix}
     \label{eq-p}
     \\
     \begin{bmatrix}
       t_x \\ t_y \\ t_z
     \end{bmatrix}
     & =
     R_{\mathtt{ImageD11}}(\theta_x, \theta_y, \theta_z)
     \cdot
     \begin{bmatrix}
       0 \\ p_y \\ p_z
     \end{bmatrix}
     +
     \begin{bmatrix}
       \Delta \\ 0 \\ 0
     \end{bmatrix}
     \label{eq-tImageD11}\end{aligned}

Note that the order of :math:`y` and :math:`z` is not the same in
eqs. [eq-p] and [eq-tImageD11].

By combining the detector flip matrix :math:`O` and the pixel size into
a detector :math:`D` matrix, this can be written as

.. math::

   \begin{aligned}
     D_{\mathtt{ImageD11}}
     & = 
     \begin{bmatrix}
       0 & 0 \\
       y_{\mathrm{size}} o_{22} & z_{\mathrm{size}} o_{21} \\
       y_{\mathrm{size}} o_{12} & z_{\mathrm{size}} o_{11}
     \end{bmatrix}
     \label{eq-DImageD11}
     \\
     \begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix}
     & =
     D_{\mathtt{ImageD11}} \cdot
     \begin{bmatrix}
       d_H - y_{\mathrm{center}} \\
       d_V - z_{\mathrm{center}}
     \end{bmatrix}\end{aligned}


Conversion
==========

Assume that the same detector geometry is described by the two
notations. How can the parameters be converted from one to the other?

Detector :math:`D`-matrix
-------------------------

The pixel size is the same in both notations, :math:`y_{\mathrm{size}} =
\mathrm{pxsize}_H` and :math:`z_{\mathrm{size}} = \mathrm{pxsize}_V`.

As ``pyFAI`` does not allow for detector flipping, :math:`o_{11}=1`,
:math:`o_{22}=-1` (because the sign of the horizontal axis is inverted
between ``ImageD11`` and ``pyFAI``) and :math:`o_{12}=o_{21}=0`. For the
detector setup described above, with :math:`d_V` increasing to the top
and :math:`d_H` increasing towards the center of the synchrotron
(i.e. opposite to the positive :math:`y`-direction), eq. [eq-DImageD11]
becomes

.. math::

   \begin{aligned}
     D_{\mathtt{ImageD11}}
     & =
     \begin{bmatrix}
       0 & 0 \\ -\mathrm{pxsize}_H & 0 \\ 0 & \mathrm{pxsize}_V
     \end{bmatrix}.
     \label{eq-dmatrixImageD11}\end{aligned}

Coordinates
-----------

Both notations use the same sign for the vertical and beam axes. The
sign of the horizontal transverse axis, however, is inverted.

The transformation between the different coordinate systems is then
achieved by:

.. math::

   \begin{aligned}
     G & =
     \begin{bmatrix}
       0 & 0 & 1 \\ 0 & -1 & 0 \\ 1 & 0 & 0
     \end{bmatrix}
     \\
     t_{\mathtt{ImageD11}}
     & =
     G \cdot
     t_{\mathtt{pyFAI}},
     \label{eq-coordconv}\end{aligned}

where :math:`t_{\mathtt{ImageD11}}`
is given by eq. [eq-tImageD11], and
:math:`t_{\mathtt{pyFAI}}` is given by eq. [eq-tpyFAI]. The matrix
:math:`G` performs the change of axes (:math:`x \leftrightarrow z`,
:math:`y \leftrightarrow -y`) and has the convenient property :math:`G^2 = 1`.

Substituting these equations into eq. [eq-coordconv], one can them
attempt to convert ``pyFAI`` parameters into ``ImageD11`` parameters and
vice versa.

.. math::

   \begin{aligned}
     R_{\mathtt{ImageD11}}
     \cdot
     D_{\mathtt{ImageD11}}
     &
     \cdot
     \begin{bmatrix}
       d_H - y_{\mathrm{center}} \\
       d_V - z_{\mathrm{center}} 
     \end{bmatrix}
     +
     \begin{bmatrix} \Delta \\ 0 \\ 0 \end{bmatrix}
     \nonumber \\
     = &
     G \cdot
     R_{\mathtt{pyFAI}}
     \cdot
     \left(
     D_{\mathtt{pyFAI}}
     \cdot
     \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     +
     \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}
     \right)
     \label{eq-transformation}\end{aligned}

Rotations
---------

Take an arbitrary vector :math:`d` with :math:`d_{\mathtt{ImageD11}}
= \begin{bmatrix} a \\ b \\ c \end{bmatrix}`. We first transform this
into the ``pyFAI`` coordinate system by multiplication with :math:`G`,
and then apply an arbitrary rotation matrix, once in before (in
``pyFAI`` coordinates, :math:`R_{\mathtt{pyFAI}}`) and once after the
transformation (in ``ImageD11`` coordinates,
:math:`R_{\mathtt{ImageD11}}`).

.. math::

   \begin{aligned}
       d_{\mathtt{pyFAI}}
       & =
       G \cdot d_{\mathtt{ImageD11}}
       = \begin{bmatrix} c \\ -b \\ a \end{bmatrix}
       \\
       R_{\mathtt{pyFAI}} \cdot d_{\mathtt{pyFAI}}
       & =
       R_{\mathtt{pyFAI}} \cdot G \cdot d_{\mathtt{ImageD11}}
       \\
       & = G \cdot R_{\mathtt{ImageD11}} \cdot d_{\mathtt{ImageD11}}.\end{aligned}

Comparing the last two lines, we find that with

.. math::

   \begin{aligned}
     R_{\mathtt{pyFAI}} \cdot G
     & =
     G \cdot R_{\mathtt{ImageD11}} \end{aligned}

the transformation is applicable for each and any vector :math:`d`.
Because :math:`G^{-1} = G` this transformation can also be applied to a
series of rotations:
:math:`G \cdot R \cdot R' = (G \cdot R \cdot G) \cdot (G \cdot R'
\cdot G) \cdot G`.

Applying this to the rotations matrices defined in
eqs. [eq-rot1]–[eq-rot3] shows, unsurprisingly, that this coordinate
transformation is an exchange of rotation axes :math:`x` and :math:`y`,
and a change of sign for :math:`y`.

.. math::

   \begin{aligned}
     G \cdot R_1(\theta) \cdot G & = R_3(\theta) \\
     G \cdot R_2(\theta) \cdot G & = R_2(-\theta) \\
     G \cdot R_3(\theta) \cdot G & = R_1(\theta)\end{aligned}

Applying this transformation to the ``pyFAI`` rotation matrix can
comparing to the ``ImageD11`` rotation matrix, we see

.. math::

   \begin{aligned}
     G \cdot R_{\mathtt{pyFAI}}(\theta_1, \theta_2, \theta_3)
     \cdot G
     & =
     G R_3(\theta_3) \cdot R_2(-\theta_2) \cdot R_1(-\theta_1)
     \cdot G
     \\
     & =
     R_1(\theta_3) \cdot R_2(\theta_2) \cdot R_3(-\theta_1)
     \\
     & =
     R_{\mathtt{ImageD11}}(\theta_x, \theta_y, \theta_z)
     \\
     & = 
     R_1(\theta_x) \cdot R_2(\theta_y) \cdot R_3(-\theta_z)\end{aligned}

We find that, by divine intervention [1]_ and despite all the efforts to
choose incompatible conventions, *the effective order of rotations is
actually the same between ``ImageD11`` and ``pyFAI``*. Consequently,
there is a direct correspondence with only a change of sign between
:math:`\theta_z` and :math:`\theta_1`:

.. math::

   \begin{aligned}
     \theta_x & = \theta_3
     \label{eq-thetax}
     \\
     \theta_y & = \theta_2
     \label{eq-thetay}
     \\
     \theta_z & = -\theta_1
     \label{eq-thetaz}\end{aligned}

Translations and offsets
------------------------

Inserting eqs. [eq-thetax]–[eq-thetaz] into [eq-transformation], we find

.. math::

   \begin{aligned}
     \begin{bmatrix} \Delta \\ 0 \\ 0 \end{bmatrix}
     = &
     G \cdot
     R_{\mathtt{pyFAI}}
     \cdot
     \left(
     D_{\mathtt{pyFAI}}
     \cdot
     \begin{bmatrix} d_H \\ d_V \end{bmatrix}
     +
     \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}
     \right)
     \nonumber \\ &
     -
     R_{\mathtt{ImageD11}}
     \cdot
     D_{\mathtt{ImageD11}}
     \cdot
     \begin{bmatrix}
       d_H - y_{\mathrm{center}} \\
       d_V - z_{\mathrm{center}} 
     \end{bmatrix}
     \\
     = &
     R_{\mathtt{ImageD11}}
     \cdot
     G \cdot
     \left(
     \begin{bmatrix}
       \mathrm{pxsize}_V d_V \\ \mathrm{pxsize}_H d_H \\ 0
     \end{bmatrix}
     +
     \begin{bmatrix} -\mathrm{poni}_1 \\ -\mathrm{poni}_2 \\ L \end{bmatrix}
     \right)
     \nonumber \\ &
     -
     R_{\mathtt{ImageD11}}
     \cdot
     \begin{bmatrix}
       0 \\
       -\mathrm{pxsize}_H (d_H - y_{\mathrm{center}}) \\
       \mathrm{pxsize}_V (d_V - z_{\mathrm{center}}) 
     \end{bmatrix}
     \\
     = &
     R_{\mathtt{ImageD11}}
     \cdot
     \left(
     \begin{bmatrix}
       L
       \\
       \mathrm{poni}_2 - \mathrm{pxsize}_H d_H
       \\
       -\mathrm{poni}_1 + \mathrm{pxsize}_V d_V
     \end{bmatrix}
     -
     \begin{bmatrix}
       0 \\
       -\mathrm{pxsize}_H (d_H - y_{\mathrm{center}}) \\
       \mathrm{pxsize}_V (d_V - z_{\mathrm{center}}) 
     \end{bmatrix}
     \right)
     \\
     = &
     R_{\mathtt{ImageD11}}
     \cdot
     \begin{bmatrix}
       L
       \\
       \mathrm{poni}_2 - \mathrm{pxsize}_H y_{\mathrm{center}}
       \\
       -\mathrm{poni}_1 + \mathrm{pxsize}_V z_{\mathrm{center}}
     \end{bmatrix}.\end{aligned}

With a little help from our friend Mathematica, we find for the
conversion from ``pyFAI`` to ``ImageD11``

.. math::

   \begin{aligned}
     \Delta
     & =
     \frac{L}{\cos(\theta_1) \cos(\theta_2)}
     \\
     y_{\mathrm{center}}
     & =
     \frac{1}{\mathrm{pxsize}_H}
     \left(
     \mathrm{poni}_2 - L \tan(\theta_1)
     \right)
     \\
     z_{\mathrm{center}}
     & =
     \frac{1}{\mathrm{pxsize}_V}
     \left(
     \mathrm{poni}_1 + L \frac{\tan(\theta_2)}{\cos(\theta_1)}
     \right),\end{aligned}

and for the conversion from ``ImageD11`` to ``pyFAI``

.. math::

   \begin{aligned}
     L
     & =
     \Delta \cos(\theta_y) \cos(\theta_z)
     \\
     \mathrm{poni}_1
     & =
     -\Delta \sin(\theta_y) + \mathrm{pxsize}_V z_{\mathrm{center}}
     \\
     \mathrm{poni}_2
     & =
     -\Delta \cos(\theta_y) \sin(\theta_z) + \mathrm{pxsize}_H y_{\mathrm{center}}.\end{aligned}

.. [1]
   May his noodly appendages forever touch you!
