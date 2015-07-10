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

Bubble
......

Developed for the SNBL and Dubble beamlines by Vadim DIADKIN.

Dahu
....

Dahu is a lightweight plugin based framework.
Lighter then EDNA, it is technically a JSON-RPC server over Tango.
Used on TRUSAXS beamline at ESRF (ID02), dahu uses pyFAI to process data
up to the kHz range.

Dioptas
.......

Graphical user interface for high-pressure diffraction, developed at the
APS synchrotron by C. Prescher and described in:
http://www.tandfonline.com/doi/full/10.1080/08957959.2015.1059835
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
The Library for Image Acquisition is used at many European synchrotrons
to control various types of camera.
A pyFAI plugin is available to integrate images on the fly without saving them.


NanoPeakCell
............
TODO ... Developed at IBS (Grenoble) by N. Coquelle

PySAXS
......
TODO ... Developed at CEA by O. Taché

xPDFsuite
.........

Developed by the Billinge Group, this commercial software is described in `arXiv 1402.3163 (2014) <http://arxiv.org/abs/1402.3163>`_

xPDFsuite is an end-to-end software solution for high throughput
Pair Distribution Function transformation, visualization and analysis.
It provides a convenient GUI for SrXplanar and PDFgetX3, allowing the users
to easily obtain 1D diffraction pattern from raw 2D diffraction images and then transform them to PDFs.


