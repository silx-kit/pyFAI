Single crystal tool: sparsify-Bragg
===================================

Compress single crystal datasets by removing the background noise

Purpose
-------

Sparsify 2D single crystal diffraction images by
separating Bragg peaks from background signal.

Positive outlier pixels (i.e. Bragg peaks) are all recorded as they are
without destruction. Peaks are not integrated.

Background is calculated by an iterative sigma-clipping in the polar
space. The number of iteration, the clipping value and the number of
radial bins could be adjusted.

This program requires OpenCL. The device needs be properly selected.


Usage:
------

spasify-Bragg [-h] [-V] [-v] [--debug] [--profile] [-o OUTPUT] [--save-source] [-b BEAMLINE] [-p PONI] [-m MASK]
[--dummy DUMMY] [--delta-dummy DELTA_DUMMY] [--radial-range RADIAL_RANGE RADIAL_RANGE] [-P POLARIZATION] [-A] [--bins BINS]
[--unit UNIT] [--cycle CYCLE] [--cutoff-clip CUTOFF_CLIP]
[--cutoff-pick CUTOFF_PICK] [--error-model ERROR_MODEL] [--noise NOISE] [--workgroup WORKGROUP] [--device DEVICE DEVICE]
[--device-type DEVICE_TYPE] [IMAGE ...]


Options:
--------

**IMAGE**
   File with input images. All frames will be concatenated in a single
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
   Threshold to be used when performing the sigmaclipping

**--cutoff-pick** CUTOFF_PICK
   Threshold to be used when picking the pixels to be saved

**--error-model** ERROR_MODEL
   Statistical model for the signal error, may be \`poisson`(default) or
   \`azimuthal\` (slower) or even a simple formula like '5*I+8'

**--noise** NOISE
   Noise level: quadratically added to the background uncertainty

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
