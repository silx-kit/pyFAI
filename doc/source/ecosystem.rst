:Author: Jérôme Kieffer
:Date: 31/05/2015
:Keywords: Other software related to pyFAI

PyFAI Ecosystem
===============

Software pyFAI is relying on
----------------------------

PyFAI is relying on the full Python scientific stack which includes [NumPy]_,
[SciPy]_, [Matplotlib]_, [PyOpenCL]_ but also on some ESRF-developped code:

FabIO
.....

PyFAI is using FabIO everywhere access to a 2D images is needed.
The *fabio_viewer* is also a lightweight convenient viewer for diffraction images.
It has been described in `doi:10.1107/S0021889813000150 <http://journals.iucr.org/j/issues/2013/02/00/kk5124/>`_

PyMca
.....

The X-ray Fluorescence Toolkit provides convenient tools for
HDF5 file browsing and mask drawing.
It has been described in `doi:10.1016/j.sab.2006.12.002 <http://www.sciencedirect.com/science/article/pii/S0584854706003764>`_

Silx
....

`The silx toolkit <https://github.com/silx-kit/silx>`_  is currently onging development.
Future releases of pyFAI will use its input/output and graphical visualization capabilities

.. _ecosystem:

Program using pyFAI as a library
--------------------------------

Bubble
......
Client-server program to perform azimuthal integration online.
Developed for the SNBL and Dubble beamlines by Vadim DIADKIN and available from this `mercurial repository <http://www.3lp.cx/>`_.

Dahu
....

Dahu is a lightweight plugin based framework available from this
`git repository <https://github.com/kif/UPBL09a>`_.
Lighter then EDNA, it is technically a JSON-RPC server over Tango.
Used on TRUSAXS beamline at ESRF (ID02), ID15 and ID31,
dahu uses pyFAI to process data up to the kHz range.

Dioptas
.......

Graphical user interface for high-pressure diffraction, developed at the
APS synchrotron by C. Prescher and described in:
`doi:10.1080/08957959.2015.1059835 <http://www.tandfonline.com/doi/full/10.1080/08957959.2015.1059835>`_

The amount of data collected during synchrotron X-ray diffraction (XRD)
experiments is constantly increasing. Most of the time, the data are
collected with image detectors, which necessitates the use of image
reduction/integration routines to extract structural information from measured XRD patterns.
This step turns out to be a bottleneck in the data processing procedure due to a lack of suitable software packages.
In particular, fast-running synchrotron experiments require online data reduction and analysis
in real time so that experimental parameters can be adjusted interactively.
Dioptas is a Python-based program for on-the-fly data processing and exploration of two-dimensional
X-ray diffraction area detector data, specifically designed for the large amount of data collected at
XRD beamlines at synchrotrons. Its fast data reduction algorithm and graphical data exploration capabilities
make it ideal for online data processing during XRD experiments and batch post-processing of large numbers of images.

Dpdak
.....

Graphical user interface for small angle diffusion, developed at the
Petra III synchrotron by G. Benecke and co-workers and described in
`doi:10.1107/S1600576714019773 <http://scripts.iucr.org/cgi-bin/paper?S1600576714019773>`_

X-ray scattering experiments at synchrotron sources are characterized by large and constantly increasing amounts of data.
The great number of files generated during a synchrotron experiment is often a limiting factor in the analysis of the data,
since appropriate software is rarely available to perform fast and tailored data processing.
Furthermore, it is often necessary to perform online data reduction and analysis during the experiment in order
to interactively optimize experimental design.
This article presents an open-source software package developed to process
large amounts of data from synchrotron scattering experiments.
These data reduction processes involve calibration and correction of raw data,
one- or two-dimensional integration, as well as fitting and further analysis of the data,
including the extraction of certain parameters.
The software, DPDAK (directly programmable data analysis kit), is based on
a plug-in structure and allows individual extension in accordance with the
requirements of the user.
The article demonstrates the use of DPDAK for on- and offline analysis of
scanning small-angle X-ray scattering (SAXS) data on biological samples and
microfluidic systems, as well as for a comprehensive analysis of
grazing-incidence SAXS data.
In addition to a comparison with existing software packages,
the structure of DPDAK and the possibilities and limitations are discussed.

EDNA
....

EDNA is a framework for developing plugin-based applications especially
for online data analysis in the X-ray experiments field (http://edna-site.org)
A EDNA data analysis server is using pyFAI as an integration engine (on the GPU)
on the ESRF BioSaxs beamline, BM29.
The server is running 24x7 with a processing frequency from 0.1 to 10 Hz.

LImA
....
The `Library for Image Acquisition <https://github.com/esrf-bliss/Lima>`_,
developped at the European synchrotron is used worldwide to control any types of
cameras.
A pyFAI plugin has been written to integrate images on the fly without saving them.
(no more tested).


NanoPeakCell
............
NanoPeakCell (NPC) is a python-software intended to pre-process your serial
crystallography raw-data into ready-to-be-inedexed images with CrystFEL,
cctbx.xfel and nXDS.
NPC is able to process data recorded at SACLA and LCLS XFELS, as well as data
recorded at any synchrotron beamline.
A graphical interface is deployed to visualize your raw and pre-processed data.

Developed at `IBS (Grenoble) by N. Coquelle <https://github.com/coquellen/NanoPeakCell>`_

pygix
.....

A Python library for reduction of 2D grazing-incidence X-ray scattering
data developped at ESRF (ID13) by Thomas DANE.

Grazing-incidence X-ray scattering techniques (GISAXS, GIWAXS/GID)
allow the study of thin films on surfaces that would otherwise be
unmeasurable in standard transmission geometry experiments. The fixed
incident X-ray angle gives rise to a distortion in the diffraction
patterns, which is extreme at wide-angles. The pygix library provides
routines for projecting 2D detector images into corrected reciprocal
space maps, radial transformations and line profile extraction using
pyFAI's regrouping functions.


PySAXS
......
Python for Small Angle X-ray Scattering data acquisition, treatment and computation
of model SAXS intensities.

Developed at CEA Saclay by O. Taché and available on `PyPI <https://pypi.python.org/pypi/pySAXS>`_.

xPDFsuite
.........

Developed by the Billinge Group, this commercial software is described in
`arXiv 1402.3163 (2014) <http://arxiv.org/abs/1402.3163>`_

xPDFsuite is an application that facilitates the workflow of atomic pair
distribution function analysis of x-ray diffraction measurements from
complex materials.  It is specially designed to help the scientist
visualize, handle and process large numbers of datasets that is common
when working with high throughput modern synchrotron sources.  It has a
full-featured interactive graphical user interface (GUI) with 3D and 3D
graphics for plotting data and it  incorporates a number of powerful
packages for integrating 2D powder diffraction images, analyzing the
curves to obtain PDFs and then tools for assessing the data and modeling
it.  It is available from `diffpy.org <http://diffpy.org>`_.


