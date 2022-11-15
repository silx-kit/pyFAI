Single crystal tool: peakfinder
===============================

Count the number of Bragg-peaks on an image

Purpose
-------
Bragg peaks are local maxima of the background subtracted signal. Peaks
are integrated and variance propagated. The centroids are reported.

Background is calculated by an iterative sigma-clipping in the polar
space. The number of iteration, the clipping value and the number of
radial bins could be adjusted.

This program requires OpenCL. The device needs be properly selected.


Usage:
------

spasify-Bragg [-h] [-V] [-v] [--debug] [--profile] [-o OUTPUT]
[--save-source] [--grid-size GRID_SIZE GRID_SIZE]
[--zig-zag] [-b BEAMLINE] [-p PONI] [-m MASK] [--dummy DUMMY]
[--delta-dummy DELTA_DUMMY]
[--radial-range RADIAL_RANGE_MIN RADIAL_RANGE_MAX] [-P POLARIZATION] [-A] [--bins BINS] [--unit UNIT]
[--cycle CYCLE] [--cutoff-clip CUTOFF_CLIP]
[--error-model ERROR_MODEL] [--cutoff-pick CUTOFF_PICK] [--noise NOISE]
[--patch-size PATCH_SIZE] [--connected CONNECTED] [--workgroup WORKGROUP] [--device DEVICE DEVICE] [--device-type DEVICE_TYPE]
[IMAGE ...]


Options:
--------

**IMAGE**
   File with input images. All results are concatenated into a single
   HDF5 file.

**-h**, **--help**
   show this help message and exit

**-V**, **--version**
   output version and exit

**-v**, **--verbose**
   show information for each frame

**--debug**
   show debug information

**--profile**
   show profiling information

Main arguments:
---------------

**-o** OUTPUT, **--output** OUTPUT
   Output filename

**--save-source**
   save the path for all source files

Scan options:
-------------

**--grid-size** GRID_SIZE GRID_SIZE
   Grid along which the data was acquired

**--zig-zag**
   The scan was performed with a zig-zag pattern

Experimental setup options:
---------------------------

**-b** BEAMLINE, **--beamline** BEAMLINE
   Name of the instument (for the HDF5 NXinstrument)

**-p** PONI, **--poni** PONI
   geometry description file

**-m** MASK, **--mask** MASK
   mask to be used for invalid pixels

**--dummy** DUMMY
   value of dynamically masked pixels

**--delta-dummy** DELTA_DUMMY
   precision for dummy value

**--radial-range** RADIAL_RANGE RADIAL_RANGE
   radial range as a 2-tuple of number of pixels, by default all
   available range

**-P** POLARIZATION, **--polarization** POLARIZATION
   Polarization factor of the incident beam [-1:1], by default disabled,
   0.99 is a good guess

**-A**, **--solidangle**
   Correct for solid-angle correction (important if the detector is not
   mounted normally to the incident beam, off by default

Sigma-clipping options:
-----------------------

**--bins** BINS
   Number of radial bins to consider

**--unit** UNIT
   radial unit to perform the calculation

**--cycle** CYCLE
   precision for dummy value

**--cutoff-clip** CUTOFF_CLIP
   SNR threshold for considering a pixel outlier when performing the
   sigma-clipping

**--error-model** ERROR_MODEL
   Statistical model for the signal error, may be \`poisson`(default) or
   \`azimuthal\` (slower) or even a simple formula like '5*I+8'

Peak finding options:
---------------------

**--cutoff-pick** CUTOFF_PICK
   SNR threshold for considering a pixel high when searching for peaks

**--noise** NOISE
   Quadratically added noise to the background

**--patch-size** PATCH_SIZE
   size of the neighborhood for integration

**--connected** CONNECTED
   Number of high pixels in neighborhood to be considered as a peak

Opencl setup options:
---------------------

**--workgroup** WORKGROUP
   Enforce the workgroup size for OpenCL kernel. Impacts only on the
   execution speed, not on the result.

**--device** DEVICE DEVICE
   definition of the platform and device identifier: 2 integers. Use
   \`clinfo\` to get a description of your system

**--device-type** DEVICE_TYPE
   device type like \`cpu\` or \`gpu\` or \`acc`. Can help to select the
   proper device.

Current status of the program: Production
