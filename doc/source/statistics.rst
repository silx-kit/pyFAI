:Author: Jérôme Kieffer
:Date: 12/07/2022
:Keywords: average sigma uncertainties standard-deviation standard-error-of-the-mean std sem
:Target: General audiance

Weighted average and uncertainty propagation
============================================

This document explains how pyFAI performs the azimuthal integration and the mathematical formula behind.
While the mean is fairly straight forward, there is a lot discussed on uncertainty propagation since
there are several uncertainties and different models to calculate them.

Naively, most people would expect the azimuthal average to be the average of the signal of each pixel from
the image which position correspon to a given bin.

Usually this pixel needs to be corrected for some dark-current (often in the case of an integrating detector) and for several normalization effects like:

* Flat-Field of the detector: each pixel can respond more or less efficiently compared to its neighbors (:math:`F`)
* Polarization effect: the beam used at synchrotron is heavily polarized and some direction will see more signal than others (:math:`P`)
* Solid-angle: Pixel which see the beam arriving with a heavy inclinasion recieve less photons (:math:`\Omega`)
* Absorption-effect, also refered to as parallax effect or thickness effect: very inclined beam passes through a longer sensor length, thus are likely to have a better detection of photons (:math:`A`)
* Normalization: The user may want to scale the signal, for example to have it in absolute units, directly useable in subsequent analysis (:math:`I_0`). This value being constant among the pixel, it can be moved freele in equations unlike the 4 others.

This can be summarized as:

.. math::

      I_{cor} = \frac{I_{raw} - I_{dark}}{F \cdot \Omega \cdot P \cdot A \cdot I_0} = \frac{signal}{normalization}

To simplify the notation, we will assign :math:`signal=I_{raw} - I_{dark}` and :math:`normalization=F \cdot \Omega \cdot P \cdot A \cdot I_0`.

Azimuthal average
-----------------
As stated in the introduction, the arithmetic average of the normalized pixel values :math:`I_{cor}` is not the proper average since some pixels are weighted more than others, thus a weigted average is needed.

In case of pixel splitting, a single pixel can contribute to serveral azimuthal bins.
Let :math:`c_{i,r}` be the contribution of pixel :math:`i` to the bin :math:`r`.
All those contributions are positive and sum up to one (the pixel is completely taken into account):

.. math::

    \sum_{r} c_{i,r} = 1


The weight for a given pixel :math:`i` in the bin :math:`r` is thus the product of the normalization
with the pixel splitting factor: :math:`\omega_i  = c_{i,r} \cdot  normalization_i`.

So the weighted average in a given by the textbook formula (`wikipedia <https://en.wikipedia.org/wiki/Weighted_arithmetic_mean>`_):

.. math::

    \overline{x} = \frac{\sum \omega \cdot x}{\sum \omega}

which simplifies slightly in our case in:

.. math::

    \overline{I_{r}} = \frac{\sum\limits_{i \in bin_r} c_{i,r} \cdot signal_i}{\sum\limits_{i \in bin_r} c_{i,r} \cdot normalization_i}

Accumulators
------------

In order to perform those calculations efficiently, possibly using multicore processor,
it is important to use a divide-and-conquer approach to reduce the amount of calculation to perform.

Four different accumulators are used in pyFAI for azimuthal integration to simplify those calculations:

.. math::

    V_A = \sum\limits_{i \in A} c_{i,r} \cdot signal_i

    VV_A = \sum \omega^2 = \sum\limits_{i \in A} c_{i,r}^2 \cdot normalization_i^2 \cdot \sigma_i^2

    \Omega_A = \sum\limits_{i \in A} c_{i,r} \cdot normalization_i

    \Omega\Omega_A = \sum\limits_{i \in A} c_{i,r}^2 \cdot normalization_i^2

Those *accumulators* are inspired by `Schubert & Gertz (2018) <https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf>`_
but due to different convention on weights considering the variance, their formula (eq22) for variance propagaion does not match ours.
Those accumulator are summing over an ensemble :math:`A` and are designed for parallel reduction by regrouping ensembles:

.. math::

    V_{A \cup B} = V_A + V_B

    \Omega_{A \cup B} = \Omega_A + \Omega_B

    \Omega\Omega_{A \cup B} = \Omega\Omega_A + \Omega\Omega_B

Uncertainties
-------------

One should distinguish two types of uncertainties: the uncertainties on the mean value, often called *standard error of the mean* and abreviated *sem*,
from the *standard deviation* which is abreviated *std*

Standard deviation
++++++++++++++++++

The standard error correspond to the uncertainty for a pixel value in the ensemble and is calculated this way in pyFAI:

.. math::

    \sigma(I_r) = \sqrt{\frac{VV_{i \in bin_r}}{\Omega\Omega_{i \in bin_r}}}

The standard deviation is rarely used in pyFAI except in the sigma-clipping procedure where it is used to discard pixels.
The numerical value can be retrieved from an azimuthal-integration result with the *std* attribute.

Standard error of the mean
++++++++++++++++++++++++++

As the name states, this uncertainty correspond to the precision with wich one knows the mean value and this is the *sigma* reported by pyFAI by default.

.. math::

    \sigma (\overline{I_r}) = \frac{\sqrt{VV_{i \in bin_r}}}{\Omega_{i \in bin_r}}

The numerical value of the *sem* is always smaller than the *std* by a factor close to :math:`\sqrt{N}`, where N is the number of pixel in the bin (unweighted mean analogy).

The numerical value can be retrieved from an azimuthal-integration result with the *sem* attribute.

Uncertainties propagated from known variance
++++++++++++++++++++++++++++++++++++++++++++

Sometimes variance can be modeled and the array VV can be calculated directly.
Very often the variance formula is based on asumption that the distribution is Poissonian (i.e. variance_i = max(1, signal_i)) which, after normalization, becomes :math:`\sigma_i^2 = max(1, signal_i)/ \cdot normalization_i^2`, thus:

.. math::

    VV_A = \sum\limits_{i \in A} c_{i,r}^2 \cdot max(signal_i, 1)

Uncertainties propagated from the variance in a ring
++++++++++++++++++++++++++++++++++++++++++++++++++++

This is the classical way to evaluate variance:

.. math::

    VV_A = \sum\limits_{i \in A} \omega_i^2\cdot(v_i - \overline{v_A})^2

Note this formula differs from `Schubert & Gertz (2018) <https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf>`_'s
paper with squared weights, but it does match the textbook or the `wikipedia <https://en.wikipedia.org/wiki/Weighted_arithmetic_mean>`_ page on the topic.
Since there is no assumption made on the underlying distribution, this formula should be used when the input data are not Poissonian.
There are several drawbacks, the first is the speed and the second, the noise of extracted uncertainties.

This formula is a classical 2-pass algorithm which is not suitable for parallel reductions, but numerically stable.
The 2-pass version is used in the python-implementation of CSR-sparse matrix multiplication and provided a ground-truth to validate the single pass version.

For accumulating the variance in a single pass, the formula becomes:

.. math::

    VV_{A\cup b} = VV_A + \omega_b^2\cdot(v_b-\frac{V_A}{\Omega_A})(v_b-\frac{V_{A\cup b}}{\Omega_{A\cup b}})

This formula is subject to numerical error accumulation and can be extended when merging two ensemble A and B (with :math:`\Omega_A > \Omega_B`):

.. math::

    VV_{A\cup B} = VV_A + VV_B + \frac{\Omega_B^2\cdot(V_A \cdot \Omega_B-V_B\cdot \Omega_A)^2}{\Omega_{A\cup B} \cdot \Omega_A \cdot \Omega_B^2}


The equivalence of those formula can be checked thanks to a notebook available at `tutorial/Variance/uncertainties <https://github.com/silx-kit/pyFAI/blob/master/doc/source/usage/tutorial/Variance/uncertainties.ipynb>`_.
It is worth noticing error-bars obtained from the azimuthal model are always more noisy (but of similar magnitude) when compared to the ones obtained from the Poisson statistics on a Poissonian signal.

Conclusion
----------

This document described the way azimuthal integration is performed within pyFAI from a mathematical point of view.
It highlights the difference between the *std* and the *sem* and exposes the two main error-models used: Azimuthal and Poisson.
