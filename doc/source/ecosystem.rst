Author: Jérôme Kieffer

Date: 05/02/2015

Keywords: Other software related to pyFAI

PyFAI Ecosystem
===============

Software pyFAI is relying on
----------------------------

FabIO
.....
PyFAI is using FabIO everywhere access to a 2D images is needed.
The fabio-viewer is also a lightweight convenient viewer for diffraction images.

PyMca
.....
The X-ray Fluorescence Toolkit provides convenient tools for
HDF5 file browsing and mask drawing.


Program using pyFAI as a library
--------------------------------

Dahu
....
Dahu is a lightweight plugin based framework.
Lighter then EDNA, it is technically a JSON-RPC server over Tango.
Used on TRUSAXS beamline at ESRF (ID02), dahu uses pyFAI to process data
up to the kHz range.

Dioptas
.......
TODO ... Developed at the APS synchrotron by C. Prescher

Dpdak
.....
TODO ... Developed at the Petra III synchrotron by G. Benecke and co-workers

EDNA
....
EDNA is a framework for developing plugin-based applications especially
for online data analysis in the X-ray experiments field (http://edna-site.org)
A EDNA data analysis server is using pyFAI as an integration engine (on the GPU)
on the ESRF BioSaxs beamline, BM29.
The server is running 24x7 with a processing frequency from 0.1 to 10 Hz.

LImA
....
The Library for Image Acquisition is used at many European synchrotrons
to control various types of camera.
A pyFAI plugin is available to integrate images on the fly without saving them.


NanoPeakCell
............
TODO ... Developed at IBS (Grenoble) by N. Coquelle

PySAXS
......
TODO ... Developed at CEA by O. Taché



