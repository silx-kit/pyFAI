:Author: Jérôme Kieffer
:Date: 06/10/2026
:Keywords: Tutorial, parallax, calibration
:Target: Scientists

.. _parallax_and_geometry:

Parallax and the sample-detector distance
=========================================

Enabling the parallax correction on experimental data rarely improves the geometry
refinement the way one would expect, and it always shifts the refined distance. This is
not a defect of the correction: **most of the parallax effect is indistinguishable from
a change of the sample-detector distance**, and a calibration which refines the distance
has already absorbed it. What is left is concentrated at high incidence angle. This page
quantifies both, so that the result of a refinement is read for what it is.

The displacement
----------------

A photon reaching the sensor under an incidence angle :math:`\alpha` deposits its energy
at a mean depth :math:`\langle z \rangle`, hence at a radial position offset by
:math:`\delta r = \langle z \rangle \tan\alpha` from where it entered. For a sensor of
thickness :math:`e` and linear absorption coefficient :math:`\mu`, writing
:math:`\mu' = \mu / \cos\alpha` for the absorption along the slanted path:

.. math::

    \langle z \rangle = \frac{1}{\mu'}\,
        \frac{1-(1+\mu' e)\,e^{-\mu' e}}{1-e^{-\mu' e}}
    \qquad
    \delta r = \frac{\sin\alpha}{\mu}\,
        \frac{1-(1+\mu' e)\,e^{-\mu' e}}{1-e^{-\mu' e}}

This is what ``ThinSensor.measure_displacement_integrate`` returns, and what
``Parallax.correct`` inverts.

Why the distance absorbs it
---------------------------

On a flat detector at a distance :math:`L`, a ring of scattering angle :math:`2\theta`
lands at :math:`r = L \tan 2\theta` and is seen at
:math:`r_{obs} = r + \langle z \rangle \tan\alpha`. Without any tilt
:math:`\alpha = 2\theta`, so:

.. math::

    \frac{r_{obs}}{L} = \tan 2\theta \left(1 + \frac{\langle z \rangle}{L}\right)

If :math:`\langle z \rangle` were a constant, this would be **exactly** the image
produced by a detector sitting at :math:`L + \langle z \rangle`: the sensor would behave
as if its reference plane were its mean absorption depth rather than its entrance
window. A refinement which lets the distance free reproduces such an image perfectly,
with a distance too large by :math:`\langle z \rangle`, and gains nothing from the
correction.

The residual the correction removes comes only from the *variation* of
:math:`\langle z \rangle` with the incidence angle. That variation is slow, because a
slanted ray travels a longer path for the same depth and is therefore absorbed closer to
the surface — for a 450 µm silicon sensor at 13.45 keV:

=========  ============  ==========
Incidence  Displacement  Mean depth
=========  ============  ==========
2°         5.56 µm       159.29 µm
20°        56.63 µm      155.58 µm
40°        119.33 µm     142.22 µm
60°        190.98 µm     110.26 µm
=========  ============  ==========

Over 2–60° the effective distance only drifts from :math:`L + 159` µm to
:math:`L + 110` µm, which is why a single refined distance fits the displaced rings so
well over most of the detector.

How much is left
----------------

The script below displaces the rings of a 100 mm geometry by the parallax effect, then
refines the distance alone — the way a calibration does — and reports what the parallax
still costs, in pixels of a 75 µm detector:

.. code-block:: python

    import numpy
    from scipy.optimize import minimize_scalar
    from pyFAI.detectors.sensors import Si_MATERIAL
    from pyFAI.parallax import Parallax, ThinSensor

    L, pixel = 0.1, 75e-6
    parallax = Parallax(ThinSensor(thickness=450e-6,
                                   mu=Si_MATERIAL.mu(energy=13.45, unit="m")))
    alpha = numpy.radians(numpy.linspace(2, 60, 150))
    r_obs = L * numpy.tan(alpha) + parallax.displace(numpy.sin(alpha))

    # the distance which best reproduces the displaced rings
    distance = minimize_scalar(lambda d: numpy.sum((numpy.arctan(r_obs / d) - alpha) ** 2),
                               bracket=(0.08, 0.12)).x
    print(f"refined distance: {distance * 1e3:.4f} mm")
    for degree in (2, 20, 40, 50, 60):
        a = numpy.radians(degree)
        observed = L * numpy.tan(a) + parallax.displace(numpy.sin(a))
        print(f"{degree:3d}°  displacement {parallax.displace(numpy.sin(a)) / pixel:5.2f} px"
              f"  residual {(observed - distance * numpy.tan(a)) / pixel:+6.2f} px")

The refined distance comes out at 100.1393 mm, i.e. 139 µm too long — close to the mean
absorption depth, as expected:

=========  ============  =========================
Incidence  Displacement  Residual after refinement
=========  ============  =========================
2°         0.07 px       +0.01 px
20°        0.76 px       +0.08 px
40°        1.59 px       +0.03 px
50°        2.06 px       −0.16 px
60°        2.55 px       −0.67 px
=========  ============  =========================

Refining the distance alone absorbs the whole effect up to 40° of incidence and still
93 % of it in rms over the full range (0.18 px rms, 2.9 mdeg). But the residual is not
spread evenly: it is negligible near the beam center and reaches **0.67 px at 60°**,
which a good calibration does resolve. Tilting the detector reduces it further
(0.79 mdeg rms for :math:`rot_1 = 0.25`, :math:`rot_2 = -0.15` rad), because the
incidence angle then spans a narrower range at a given scattering angle.

What to expect in practice
--------------------------

* **The refined distance drops by about** :math:`\langle z \rangle` when the correction
  is switched on — 139 µm in the example above. This is the most visible consequence, and
  it is the *right* distance: it is then measured to the entrance window of the sensor
  rather than to its mean absorption depth.
* **The gain on the residual lives at the edge of the detector.** Judging the correction
  on the global rms of a calibration is misleading, since the control points at low
  incidence dominate it and carry no information on the parallax. Compare the residuals
  of the outermost rings instead.
* **It is worth enabling for a thick sensor used at high incidence**: a 450 µm silicon
  sensor beyond 50°, and sooner for CdTe or for a detector placed close to the sample.
  Below 40° of incidence the correction is reparametrized away by the distance.
* **The correction also matters when the distance is not free**: a geometry transferred
  from another calibration, a detector on a goniometer whose position is known otherwise,
  or a :class:`~pyFAI.multi_geometry.MultiGeometry` where one distance serves several
  detector positions. There even the low-angle part is no longer absorbed.
* **A refinement which gets worse when the correction is enabled** is pointing at
  something else — module positions, distortion, an actual sensor thickness differing
  from the nominal one, planarity — not at the parallax.

Barycenter or maximum
---------------------

By default the correction targets the displacement of the **barycenter** of the energy
deposit. The absorption profile is a decreasing exponential, so its maximum stays at the
entrance of the sensor while its barycenter moves by :math:`\delta r`: a narrow peak
convolved with it moves by markedly less than :math:`\delta r`. Control points, however,
are extracted
by a local maximum search (``Bilinear.local_maxi``), and 1D peaks are usually positioned
by their maximum too. For a ring 2 pixels wide on a 450 µm silicon sensor the maximum
moves by 118 µm at 60° of incidence where the barycenter moves by 191 µm, so the
correction overshoots by about one pixel there — of the same order as the effect it is
removing.

This is what the ``beam`` parameter of
:meth:`~pyFAI.geometry.core.Geometry.enable_parallax` is for: given the width of the
peaks, the correction targets the maximum rather than the barycenter.

.. code-block:: python

    from pyFAI.parallax import Beam

    ai.enable_parallax(True, beam=150e-6)        # FWHM of the peaks, gaussian by default
    ai.enable_parallax(True, beam=Beam(150e-6, "square"))   # or a full Beam instance

Beware of two practical consequences:

* Tabulating the displacement with a beam requires a convolution per angle and takes
  about a second, against a few milliseconds without. Re-activating an identical setup
  is a no-op, so the calibration GUI — which calls ``enable_parallax`` once per
  refinement pass — only pays it once.
* The beam is **not** stored in the poni-file, whose version 3 format only records
  whether the correction is active. A geometry saved with a beam comes back with the
  barycenter model; :meth:`~pyFAI.geometry.core.Geometry.save` warns when this happens.
