Mask generation tool: drawMask_pymca
====================================

Purpose
-------

Draw a mask, i.e. an image containing the list of pixels which are considered invalid (no scintillator, module gap, beam stop shadow, ...).

.. figure:: ../img/drawMask.png
   :align: center
   :alt: image


This will open a PyMca window and let you draw on the first image (provided) with different tools (brush, rectangle selection, ...).
When you are finished, come back to the console and press enter.
The mask image is saved into file1-masked.edf.
Optionally the script will print the number of pixel
masked and the intensity masked (as well on other files provided in input)


Usage: drawMask_pymca [options] file1.edf file2.edf ...

Options:
--------

  --version   show program's version number and exit
  -h, --help  show help message and exit

Optionally the script will print the number of pixel masked and the intensity masked (as well on other files provided in input)

.. command-output:: drawMask_pymca --help
    :nostderr:
