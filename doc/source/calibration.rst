Calibration
===========

The determination of the geometry of the experimental setup is called 
calibration in pyFAI.
PyFAI assumes this setup does not change during the experiment.
A geometry setup is composed of a detector, refined parameters like the distance 
and fixed parameters like the wavelength (or the energy of the beam), they are all 
saved together into a text files named ".poni" (as a reference to the point of 
normal incidence) which is subsequently used for processing the experiment.

By storing all parameters together in a single small file, the risk of mixing two 
parameters is highly reduced and we believe this helps performing better 
science with fewer mistakes.  

While entering the geometry of the experiment in a poni-file is possible it is 
easier to perform a calibration, using the Debye-Sherrer rings of a reference 
sample called calibrant. 
Many calibrant description files are shipped with the default installation of pyFAI, 
like LaB6, silicon, ceria, corrundum or silver behenate. 
The user can choose to provide their own calibrant description files which are 
simple text-file containing the largest d-spacing (in Angstrom) for a set of 
Miller plans.     

The calibration is divided into 4 major steps:

Pre-processing of images: 
-------------------------


averaging, dark/flat correction, desaturation

Peak-picking
------------


Identification of peaks and groups of peaks belonging to same ring

Refinement of the parameters
----------------------------

Least-squares refinement of the geometry parameters on peak position

Validation of the calibration
-----------------------------


Validation by an human being of the geometry