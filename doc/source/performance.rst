:Author: Jérôme Kieffer
:Date: 22/01/2021
:Keywords: Performance analysis
:Target: user

.. _performances:

Performances
============

The performance analysis has to be done on what matters' most for the end user,
which is often the azimuthal integration step.
PyFAI features many integrators, many pixel splitting schemes and the best one is not always obvious.
The :ref:`pyfaibenchmark` tool measures the time to perform the 1D azimuthal integration for images
between 1 and 16 megapixels and helps the user to take this descision.
The pixel splitting scheme used, ``bbox``, is also the default one.
Other splitting schemes have similar behavior.

By default the histogram and CSR algorithm implemented in cython are compared.
All those cython engines accumulate information in `float64` containers while
OpenCL implementations accumulates using error-compensated_arithmetics_ with
10 `float32` operations per addition.

.. _error-compensated_arithmetics: http://www.theses.fr/2017LYSEN036
.. _Kahan_summation: http://en.wikipedia.org/wiki/Kahan_summation_algorithm

The benchmarking tool provides plots like this:

.. figure:: img/benchmark_2024.01.svg
   :align: center
   :alt: Benchmark performed on a 2016 single-socket workstation and two graphics card.

This plot shows the number of images processed per second as function of the image size for various integrators.
The vertical axis is in logarithmic scale, so a small offset can represent a factor two in speed.
Those curves have all hyperbolic shapes, which means that larger images process slower.

By default, the benchmarking tool probes histogram-, CSC- and CSR- based integrators (precisely ``("bbox", "histogram", "cython")``, ``("bbox", "csc", "cython")``
and ``("bbox", "csr", "cython")``.
The 2 formers are single threaded and offers the *worse* performances but it is still much faster than numpy based histograms.
They are usually the lowest curves.

The CSR-integration is usually faster than the histogram thanks to the multi-threading.
On Apple system and other system where multithreading is disabled, it can be that CSR-integrator
is slower than histograms.
One may also appreciate histogram-based integrators for their quicker initialization time or
their lower memory footprint, for example when dealing with multi-geometry objects.

In this plot, one OpenCL device has been added (plotted with dashed lines), it is a high-end GPU.
GPU provides the best performances when it comes to azimuthal integration, it is usually the upper most curve,
with speed up to 1000 or 2000 1Mpixel frames processed per second (on high-end GPU).
The best performances registered with this method is 2.5 GPix/s on recent gaming graphics card.

Since pyFAI version 0.20 introduced a new generation of integrator (in production),
both `integrate1d_legacy` and `integrate1d_ng` are benchmarked together to validate the absence of regression.
Note that `integrate1d_ng` performs 4 sums (on signal, variance, normalization and count) for each bin,
while former version `integrate1d_legacy` used to sum only 2 (signal and count), it is normal that the new version is
slightly slower. A factor larger than 2 should be considered as a bug.
Depending on the system, the new version is sometimes even faster, up to 30% faster for
histograms on Zen2 AMD processor with the histogram (thanks to the large caches available).
