The determination of the geometry of the experimental setup for the diffraction pattern
of a reference sample is called calibration in pyFAI.
A geometry setup is composed of a detector, the six refined parameters like the distance
and fixed parameters like the wavelength (or the energy of the beam), they are all
saved together into a text files named ".poni" (as a reference to the point of
normal incidence) which is subsequently used for processing the experiment.

 The program *pyFAI-calib* helps calibrating
the experimental setup using a constrained least squares optimization on
the Debye-Scherrer rings of a reference sample (:math:`LaB_6`, silver
behenate, …) and saves the results into a *.poni* file.
Alternatively, geometries calibrated using fit2d\  can be
imported into pyFAI, including geometric distortions (i.e. optical-fiber
tapers distortion) described as *spline-files*.

By storing all parameters together in a single small file, the risk of mixing two
parameters is highly reduced and we believe this helps performing better
science with fewer mistakes.

While entering the geometry of the experiment in a poni-file is possible it is
easier to perform a calibration, using the Debye-Sherrer rings of a reference
sample called calibrant.
About 10 calibrant description files are shipped with the default installation of pyFAI,
like LaB6, silicon, ceria, corrundum or silver behenate.
The user can choose to provide their own calibrant description files which are
simple text-file containing the largest d-spacing (in Angstrom) for a set of
Miller plans. A useful reference is the American Mineralogist database [AMD]_
or the Crystallographic Open database [COD]_.

The calibration is divided into 4 major steps:

Pre-processing of images:
-------------------------
The typical pre-processing consists of the averaging (or median filter) of darks images.
Dark current images are then subtracted from data and corrected for flat.

If saturated pixels exists, the are likely to be treated like peaks but their positions
will be wrong.
It is advised to either mask them out or to desaturate them (pyFAI provides an option,
but it is expensive in calculation time)

Peak-picking
------------


The Peak-picking consists in the identification of peaks and groups of peaks
belonging to same ring. It can be performed by two methods : blob detection or
massif detection.

Massif detection
................

This method consists in making the difference of the original image and a blurred
image. Then we look for a chain of positives values, corresponding to a single group
of peak. The blurring parameter can be adjusted using the "-g" option in pyFAI-calib.

Blob detection
..............

The approach is based on difference of gaussians (DoGs) as described in the blob_detection_ article of wikipedia.

.. _blob_detection: http://en.wikipedia.org/wiki/Blob_detection

It consists in blurring the image by convolution with a 2D gaussian kernel and making
differences between two successive blurs (called Difference Of Gaussian).
In theses DoGs, keypoints are defined as the maxima in the 3D space (y,x,size of
the gaussian). After their localization, keypoints are refined by Savitzky Golay
algorithm or by an interpolation at the second order which is equivalent but uses
less points. At this step, if the estimation of the maximum is too far from the maximum,
the keypoint will be considered as a fake maximum and removed.

Steepest ascent
...............

This is very naive implementation which looks for the nearest local maximum.
Subsequently a sub-pixel optimization is performed based on the local gradient and hessian.

Monte-Carlo sampling
....................

Series of peaks can be extracted using the Steepest Ascent on randomly selected seeds.

Refinement of the parameters
----------------------------

After grouping of peaks, groups of peak are assigned to a Debye-Scherrer ring and
to a d-spacing. PyFAI uses a least-squares refinement of the geometry parameters on
peak position.

The optimization procedure is the Sequential Least SQuares Programming
implemented in scipy.optimize.slsqp.
The cost function is the sum of the square of the difference between the expected and
calculated 2\theta values for the various peaks. This sum is dependent on the number
of control-points.


Validation of the calibration
-----------------------------

Validation by an human being of the geometry is an essential step:
pyFAI will overlay to the diffraction image, the lines corresponding to the various diffraction
rings expected from the calibrant. Those lines should be in pretty good agreement with the rings
of the scattering image.

Once the calibration is finished, one can use the validate option to check the offset between the
input image and the generated on from the diffraction pattern.

