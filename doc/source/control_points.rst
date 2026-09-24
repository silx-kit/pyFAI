:Author: Jérôme Kieffer
:Date: 08/09/2026
:Keywords: control points, peak-picking, sub-pixel refinement
:Target: Advanced users

.. _control_points:

Extraction of control points
============================

A *control point* is the position of a peak of the calibrant, measured on the diffraction
image and assigned to a given Debye-Scherrer ring. It is the elementary piece of
information used by the calibration: the refinement minimizes the difference between the
:math:`2\theta` value calculated from the position of every control point and the one
expected from the *d*-spacing of its ring.

Since the geometry is refined from those positions only, the accuracy of the calibration
can never be better than the accuracy with which the peaks are located. This is why all
the algorithms described here provide a **sub-pixel** position, and why the last section is
dedicated to this refinement.

Control points are extracted interactively in ``pyFAI-calib2``, or automatically by the
``recalib`` command and by :meth:`~pyFAI.gui.cli_calibration.AbstractCalibration.extract_cpt`,
which uses the geometry already known to predict where the rings are and to seed the
search. They are stored in ``.npt`` files, together with the calibrant and the wavelength.

Three algorithms are available, selected by the ``method`` argument of
:meth:`~pyFAI.gui.peak_picker.PeakPicker.peaks_from_area`: ``"massif"`` (the default),
``"blob"`` and ``"watershed"``. All of them take a region of interest as a mask, and
return a list of ``(dim1, dim2)`` positions expressed in pixel units, the center of the
first pixel being at ``(0, 0)``.

Massif detection
----------------

Implemented in :class:`pyFAI.massif.Massif`.

The image is smoothed with a Gaussian filter and the smoothed version is subtracted from
the original one. The connected regions of positive values of this difference — the
*massifs* — are labelled: each of them corresponds to a piece of ring. The width of the
Gaussian is the ``valley_size`` attribute (option ``-g`` of ``pyFAI-calib``), documented as
the minimum distance between two massifs and defaulting to ``max(5, max(shape)/50)``: it
sets the scale at which two neighbouring peaks are considered as belonging to the same
massif.

Within a massif, peaks are then found by a *steepest ascent*: starting from a seed pixel,
the algorithm walks towards the brightest of the neighboring pixels until a local maximum
is reached, and the position of this maximum is refined at the sub-pixel level. Two
entry points are available:

* :meth:`~pyFAI.massif.Massif.find_peaks` starts from a single seed, computes the massif
  it belongs to, and extracts the peaks of this massif;
* :meth:`~pyFAI.massif.Massif.peaks_from_area` performs a Monte-Carlo sampling: seeds are
  drawn at random within the region of interest, which makes the sampling of a ring
  reasonably uniform. A list of *good guesses* may be provided as ``seed``, which biases
  the search towards an already known geometry, and ``dmin`` enforces a minimum distance
  between two extracted points.

This is the cheapest method and the one used by default. Because a massif is a connected
region, it works best when the rings are continuous and well separated.

Blob detection
--------------

Implemented in :class:`pyFAI.blob_detection.BlobDetection`.

This approach is based on the difference of Gaussians (DoG), as described in the
blob_detection_ article of Wikipedia. The image is convolved with Gaussian kernels of
increasing width and the successive blurred versions are subtracted from each other, which
builds a scale-space. Keypoints are the maxima of this 3D space :math:`(dim1, dim2,
\sigma)`: each of them comes with its own size, hence the method adapts itself to peaks of
different widths without any tuning.

.. _blob_detection: http://en.wikipedia.org/wiki/Blob_detection

Keypoints are refined in the three dimensions at once, either by a second-order
interpolation — the full 3x3 Hessian of the DoG is inverted, see
:meth:`~pyFAI.blob_detection.BlobDetection.refine_Hessian` — or by a Savitzky-Golay
filter. A keypoint whose refined position is too far from the pixel it was found in is
considered as spurious and discarded, which makes the method fairly robust against noise.

The scale-space makes this method the most expensive of the three, but it is the only one
which measures the size of the peaks and the only one able to work at several scales
simultaneously.

Inverse watershed
-----------------

Implemented in :class:`pyFAI.ext.watershed.InverseWatershed`.

The image is seen as a landscape which is flooded from its summits instead of its valleys:
every pixel is linked to the local maximum reached by a steepest ascent, which partitions
the image into as many regions as there are local maxima. The borders between regions and
the height of the pass joining two neighbouring regions are then computed.

The regions overlapping the area of interest are selected, and the maxima of the retained
regions become the control points, refined at the sub-pixel level when ``refine=True``.
Since the partition is complete, this method separates peaks which are close to each other
better than the massif approach does, at the cost of building the full partition of the
image.

Three merging strategies are implemented — ``merge_singleton`` for one-pixel regions,
``merge_twins`` for regions pointing at each other, and ``merge_intense``, which merges two
regions when the relative height of their pass, :math:`(pass - min)/(max - min)`, exceeds
the ``thres`` parameter. Note that none of them is currently called by
:meth:`~pyFAI.ext.watershed.InverseWatershed.init`: as of today, every local maximum
defines its own region.

.. _subpixel:

Sub-pixel refinement
--------------------

The massif and the watershed methods share the same refinement, implemented in
:meth:`pyFAI.ext.bilinear.Bilinear.local_maxi`; blob detection uses its own, in the
scale-space.

Around the pixel :math:`(i, j)` holding the local maximum, the intensity is described by
its second order Taylor expansion. At the maximum the gradient vanishes, hence the
sub-pixel offset :math:`\delta` is the solution of:

.. math::

    \delta = - H^{-1} \cdot \nabla f

where the gradient :math:`\nabla f` and the Hessian :math:`H` are estimated by centred
finite differences on the 3x3 neighbourhood of the maximum. The offset is accepted only if
it stays inside the neighbourhood; otherwise the position falls back to the centre of mass
of the 3x3 patch, and finally to the position of the pixel itself.

A few consequences are worth keeping in mind when aiming at a precision better than a
tenth of a pixel:

* the expansion is **exact** for a quadratic surface, and remains excellent for a smooth,
  well sampled peak: on a Gaussian of :math:`\sigma = 3` pixels, the position is recovered
  with an error of a few :math:`10^{-3}` pixel;
* it degrades quickly when the peak is **under-sampled**. A Gaussian of :math:`\sigma = 0.5`
  pixel, i.e. a FWHM slightly larger than one pixel, cannot be located better than a few
  tenths of a pixel from three points, whatever the quality of the data;
* it also degrades when the peak is strongly **anisotropic**, which is precisely what a
  narrow ring looks like locally: sharp radially, flat azimuthally;
* the calculation is performed on the raw data, so the intensity of the background is
  taken into account. A high background flattens the curvature and pulls the fallback
  centre of mass towards the centre of the pixel.

In practice, this means that a calibrant should be chosen and an experiment designed so
that the rings are sampled by at least two or three pixels across their width. Sharper
rings look nicer but are located less accurately.

Choosing a method
-----------------

* ``massif`` is the default and is the right choice for a standard calibration on
  continuous, well separated rings. It is also the fastest.
* ``watershed`` is preferable when the rings are close to each other, or at high angle
  where they start to overlap: since every local maximum defines its own region, peaks
  that a single massif would merge stay separated.
* ``blob`` is useful when the peaks have very different sizes or intensities across the
  image, and when spurious keypoints must be filtered out aggressively; it is the slowest.

Whatever the method, the extracted points should always be inspected on the image, and the
residual error per control point, printed at the end of the refinement, compared with the
angle subtended by one pixel: a calibration whose residual is much smaller than a pixel is
either excellent or over-fitted.
