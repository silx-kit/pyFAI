Integration tool: pyFAI-integrate
=================================

Purpose
-------

PyFAI-integrate is a graphical interface (based on Python/Qt4) to perform azimuthal integration
on a set of files. It exposes most of the important options available within pyFAI and allows you
to select a GPU (or an openCL platform) to perform the calculation on.

.. figure:: ../img/integrate.png
   :align: center
   :alt: image


Usage
-----

pyFAI-integrate [options] file1.edf file2.edf ...

Options:
--------

  --version             show program's version number and exit
  -h, --help            show help message and exit
  -v, --verbose         switch to verbose/debug mode
  -o OUTPUT, --output=OUTPUT
                        Directory or file where to store the output data

Tips & Tricks:
--------------

PyFAI-integrate saves all parameters in a .azimint.json (hidden) file. This JSON file
is an ascii file which can be edited and used to configure online data analysis using
the LImA plugin of pyFAI.

Nota: there is bug in debian6 making the GUI crash (to be fixed inside pyqt)
http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348

Example:
--------


.. command-output:: pyFAI-integrate --help
    :nostderr:
