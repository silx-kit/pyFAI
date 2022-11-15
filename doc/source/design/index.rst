:Author: Jérôme Kieffer
:Date: 30/10/2020
:Keywords: Design
:Target: Developers interested in using the library
:Reference: API documentation

.. _Design:

Design of the library
=====================

PyFAI can be seen as a set of application written in Python or as the underlying library.
I would describe it as a stack of 5 layers, upper ones relying on lower ones:

5. Graphical user interfaced application: ``pyFAI-calib2``, ``diff_map``, ...

4. Command-line scripts: ``pyFAI-average``, ``pyFAI-benchmark``, ...

3. Top level objects: ``AzimuthalIntegrator``, ``Distortion``, ...

2. Helper classes: ``Detector``, ``Calibrant``, ...

1. Rebinning engines: ``CSRIntegrator``, ``OCL_CSR_Integrator``, ...

0. Common basement: Python, Numpy, Cython, PyOpenCL and silx

At level 5. are easy-to use applications, requiring no specific computing knowledge.
Levels 3. and 2. are heavily described in tutorials and should be accesible to
any Pythonistas. We try to keep the API on those levels consistent between versions.
Level 1. is often written in Cython, or OpenCL and requires low-level expertise.

Of course mastering level 1. provides faster analysis, but it is not of general use.

.. toctree::
   :maxdepth: 2

   ai
