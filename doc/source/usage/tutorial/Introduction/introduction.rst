
Introduction to the tutorials
=============================

The iPython/Jupyter notebook
----------------------------

The document you are seeing it may be a PDF or a web page or another
format, but what is important is how it has been made ...

According to this article in
`Nature <http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261>`__
the Notebook, invented by iPython, and now part of the Jupyter project,
is the revolution for data analysis which will allow reproducable
science.

This tutorial will introduce you to how to access to the notebook, how
to use it and perform some basic data analysis with it and the pyFAI
library.

Getting access to the notebook
------------------------------

There are many cases ...

Inside ESRF
~~~~~~~~~~~

The simplest case as the data analysis unit offers you a notebook
server: just connect your web browser to http://scisoft13:8000 and
authenticate with your ESRF credentials.

Outside ESRF with an ESRF account.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JupyterHub server is not directly available on the internet, but you
can
``login into the firewall<http://www.esrf.eu/Infrastructure/Computing/Networks/InternetAndTheFirewall/UsersManual/SSH>``\ \_
to forward the web server:

ssh -XC -p5022 -L8000:scisoft13:8000 user@firewall.esrf.fr

Once logged in ESRF, keep the terminal opened and browse on your local
computer to http://localhost:8000 to authenticate with your ESRF
credentials. Do not worry about confidentiality as the connection from
your computer to the ESRF firewall is encrypted by the SSH connection.

Other cases
~~~~~~~~~~~

In the most general case you will need to install the notebook on your
local computer in addition to pyFAI and FabIO to follow the tutorial.
WinPython provides it under windows. Please refer to the installation
procedure of pyFAI to install locally pyFAI on your computer

Getting trained in using the notebook
-------------------------------------

There are plenty of good tutorials on how to use the notebook. `This
one <https://github.com/jupyter/mozfest15-training/blob/master/00-python-intro.ipynb>`__
presents a quick overview of the Python programming language and
explains how to use the notebook. Reading it is strongly encouraged
before proceeding to the pyFAI itself.

Anyway, the most important information is to use **Control-Enter** to
evaluate a cell.

In addition to this, we will need to download some files from the
internet. The following cell contains a piece of code to download files.
You do not need to understand what it does, but you may have to adjust
the proxy settings to be able to connect to internet, especially at
ESRF.

.. code:: python

    import os, sys
    os.environ["http_proxy"] = "http://proxy.site.com:3128"
    
    def download(url):
        """download the file given in URL and return its local path"""
        if sys.version_info[0]<3:
            from urllib2 import urlopen, ProxyHandler, build_opener
        else:
            from urllib.request import urlopen, ProxyHandler, build_opener
        dictProxies = {}
        if "http_proxy" in os.environ:
            dictProxies['http'] = os.environ["http_proxy"]
            dictProxies['https'] = os.environ["http_proxy"]
        if "https_proxy" in os.environ:
            dictProxies['https'] = os.environ["https_proxy"]
        if dictProxies:
            proxy_handler = ProxyHandler(dictProxies)
            opener = build_opener(proxy_handler).open
        else:
            opener = urlopen
        target = os.path.split(url)[-1]
        with open(target,"wb") as dest, opener(url) as src:
            dest.write(src.read())
        return target

Introduction to diffraction image analysis using the notebook
-------------------------------------------------------------

All the tutorials in pyFAI are based on the notebook and if you wish to
practice the exercises, you can download the notebook files (.ipynb)
from
`Github <https://github.com/kif/pyFAI/tree/master/doc/source/usage/tutorial>`__

Load and display diffraction images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all we will download an image and display it. Displaying it the
right way is important as the orientation of the image imposes the
azimuthal angle sign.

.. code:: python

    #initializes the visualization module
    %pylab inline


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    moke = download("http://www.silx.org/pub/pyFAI/testimages/moke.tif")
    print(moke)


.. parsed-literal::

    moke.tif


The *moke.tif* image we just downloaded is not a real diffraction image
but it is a test pattern used in the tests of pyFAI.

Prior to displaying it, we will use the Fable Input/Output library to
read the content of the file:

.. code:: python

    import fabio
    img = fabio.open(moke).data
    imshow(img, origin="lower", cmap="gray")




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fd7aaa975c0>




.. image:: output_6_1.png


As you can see, the image looks like an archery target. The option
*origin="lower"* of *imshow* allows to display the image with the origin
at the **lower left** of the image.

Displaying the image without this option ends with having the azimuthal
angle (which angles are displayed in degrees on the image) to turn
clockwise, so the inverse of the trigonometric order.

.. code:: python

    imshow(img, cmap="gray")




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fd7aaa70b00>




.. image:: output_8_1.png


**Nota:** Displaying the image properly or not does not change the
content of the image or its representation in memory, it only changes
its representation, which is important only for the user. **DO NOT USE**
*numpy.flipud* or other array-manipulation which changes the memory
representation of the image. This is likely to mess-up all your
subsequent calculation.

1D azimuthal integration
~~~~~~~~~~~~~~~~~~~~~~~~

To perform an azimuthal integration of this image, we need to create an
**AzimuthalIntegrator** object we will call *ai*. Fortunately, the
geometry is explained on the image.

.. code:: python

    import pyFAI
    ai = pyFAI.AzimuthalIntegrator(dist=0.1, pixel1=1e-4, pixel2=1e-4)
    print(ai)


.. parsed-literal::

    Detector Detector	 Spline= None	 PixelSize= 1.000e-04, 1.000e-04 m
    SampleDetDist= 1.000000e-01m	PONI= 0.000000e+00, 0.000000e+00m	rot1=0.000000  rot2= 0.000000  rot3= 0.000000 rad
    DirectBeamDist= 100.000mm	Center: x=0.000, y=0.000 pix	Tilt=0.000 deg  tiltPlanRotation= 0.000 deg


Printing the *ai* object displays 3 lines:

-  The detector definition, here a simple detector with square, regular
   pixels with the right size
-  The detector position in space using the *pyFAI* coordinate system
-  The detector position in space using the *FIT2D* coordinate system

Right now, the geometry in the *ai* object is wrong. It may be easier to
define it correctly using the *FIT2D* geometry which uses pixels for the
center coordinates (but the sample-detector distance is in millimeters).

.. code:: python

    help(ai.setFit2D)


.. parsed-literal::

    Help on method setFit2D in module pyFAI.geometry:
    
    setFit2D(directDist, centerX, centerY, tilt=0.0, tiltPlanRotation=0.0, pixelX=None, pixelY=None, splineFile=None) method of pyFAI.azimuthalIntegrator.AzimuthalIntegrator instance
        Set the Fit2D-like parameter set: For geometry description see
        HPR 1996 (14) pp-240
        
        Warning: Fit2D flips automatically images depending on their file-format.
        By reverse engineering we noticed this behavour for Tiff and Mar345 images (at least).
        To obtaine correct result you will have to flip images using numpy.flipud.
        
        @param direct: direct distance from sample to detector along the incident beam (in millimeter as in fit2d)
        @param tilt: tilt in degrees
        @param tiltPlanRotation: Rotation (in degrees) of the tilt plan arround the Z-detector axis
                * 0deg -> Y does not move, +X goes to Z<0
                * 90deg -> X does not move, +Y goes to Z<0
                * 180deg -> Y does not move, +X goes to Z>0
                * 270deg -> X does not move, +Y goes to Z>0
        
        @param pixelX,pixelY: as in fit2d they ar given in micron, not in meter
        @param centerX, centerY: pixel position of the beam center
        @param splineFile: name of the file containing the spline
    


.. code:: python

    ai.setFit2D(100, 300, 300)
    print(ai)


.. parsed-literal::

    Detector Detector	 Spline= None	 PixelSize= 1.000e-04, 1.000e-04 m
    SampleDetDist= 1.000000e-01m	PONI= 3.000000e-02, 3.000000e-02m	rot1=0.000000  rot2= 0.000000  rot3= 0.000000 rad
    DirectBeamDist= 100.000mm	Center: x=300.000, y=300.000 pix	Tilt=0.000 deg  tiltPlanRotation= 0.000 deg


With the *ai* object properly setup, we can perform the azimuthal
integration using the *intergate1d* method. This methods takes only 2
mandatory parameters: the image to integrate and the number of bins. We
will provide a few other to enforce the calculations to be performed in
2theta-space and in degrees:

.. code:: python

    tth, I = ai.integrate1d(img, 300, unit="2th_deg")
    plot(tth, I, label="moke")
    xlabel("2theta (deg)")




.. parsed-literal::

    <matplotlib.text.Text at 0x7fd7aab3e908>




.. image:: output_15_1.png


As you can see, the 9 rings gave 9 sharp peaks at 2theta position
regularly ranging from 4 to 12 degrees as expected from the image
annotation.

**Nota:** the default unit is "q\_nm^1", so the scattering vector length
expressed in inverse nanometers. To be able to calculate *q*, one needs
to specify the wavelength used (here we didn't). For example:
ai.wavelength = 1e-10

To save the content of the integrated pattern into a 2 column ASCII
file, one can either save the (tth, I) arrays, or directly ask pyFAI to
do it by providing an output filename:

.. code:: python

    ai.integrate1d(img, 30, unit="2th_deg", filename="moke.dat")
    !cat moke.dat


.. parsed-literal::

    # == pyFAI calibration ==
    # SplineFile: None
    # PixelSize: 1.000e-04, 1.000e-04 m
    # PONI: 3.000e-02, 3.000e-02 m
    # Distance Sample to Detector: 0.1 m
    # Rotations: 0.000000 0.000000 0.000000 rad
    # 
    # == Fit2d calibration ==
    # Distance Sample-beamCenter: 100.000 mm
    # Center: x=300.000, y=300.000 pix
    # Tilt: 0.000 deg  TiltPlanRot: 0.000 deg
    # 
    # Polarization factor: None
    # Normalization factor: 1.0
    # --> moke.dat
    #       2th_deg             I 
      3.831631e-01    6.384597e+00
      1.149489e+00    1.240657e+01
      1.915815e+00    1.222277e+01
      2.682141e+00    1.170348e+01
      3.448468e+00    9.964798e+00
      4.214794e+00    8.913503e+00
      4.981120e+00    9.104074e+00
      5.747446e+00    9.242975e+00
      6.513772e+00    6.136262e+00
      7.280098e+00    9.039030e+00
      8.046424e+00    9.203415e+00
      8.812750e+00    9.324570e+00
      9.579076e+00    6.470130e+00
      1.034540e+01    7.790757e+00
      1.111173e+01    9.410036e+00
      1.187805e+01    9.464832e+00
      1.264438e+01    7.749060e+00
      1.341071e+01    1.151200e+01
      1.417703e+01    1.324891e+01
      1.494336e+01    1.038730e+01
      1.570969e+01    1.069764e+01
      1.647601e+01    1.056094e+01
      1.724234e+01    1.286720e+01
      1.800866e+01    1.323239e+01
      1.877499e+01    1.548398e+01
      1.954132e+01    2.364553e+01
      2.030764e+01    2.537154e+01
      2.107397e+01    2.512984e+01
      2.184029e+01    2.191267e+01
      2.260662e+01    7.605135e+00


Here the excalmation mark indicates the notebook to call the *cat*
command from UNIX to print the content of the file. This "moke.dat" file
contains in addition to the 2th/I value, a header commented with "#"
with the geometry used to perform the calculation.

**Nota: ** The *ai* object has initialized the geometry on the first
call and re-uses it on subsequent calls. This is why it is important to
re-use the geometry in performance critical applications.

2D integration or Caking
~~~~~~~~~~~~~~~~~~~~~~~~

One can perform the 2D integration which is called caking in FIT2D by
simply calling the *intrgate2d* method with 3 mandatroy parameters: the
data to integrate, the number of radial bins and the number of azimuthal
bins.

.. code:: python

    I, tth, chi = ai.integrate2d(img, 300, 360, unit="2th_deg")
    imshow(I, origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto")
    xlabel("2 theta (deg)")
    ylabel("Azimuthal angle chi (deg)")




.. parsed-literal::

    <matplotlib.text.Text at 0x7fd7aaa0aa58>




.. image:: output_19_1.png


The displayed image presents the "caked" image with the radial and
azimuthal angles properly set on the axes. Search for the -180, -90,
360/0 and 180 mark on the transformed image.

Like *integrate1d*, *integrate2d* offers the ability to save the
intgrated image into an image file (EDF format by default) with again
all metadata in the headers.

Radial integration
~~~~~~~~~~~~~~~~~~

Radial integration can directly be obtained from Caked images:

.. code:: python

    target = 8 #degrees
    #work on fewer radial bins in order to have an actual averaging:
    I, tth, chi = ai.integrate2d(img, 100, 90, unit="2th_deg")
    column = argmin(abs(tth-target))
    print("Column number %s"%column)
    plot(chi, I[:,column])
    xlabel("Azimuthal angle")


.. parsed-literal::

    Column number 34




.. parsed-literal::

    <matplotlib.text.Text at 0x7fd7a833dcc0>




.. image:: output_21_2.png


**Nota:** the pattern with higher noise along the diagonals is typical
from the pixel splitting scheme employed. Here this scheme is a
"bounding box" which makes digonal pixels look a bit larger (+40%) than
the ones on the horizontal and vertical axis, explaining the variation
of the noise.

Integration of a bunch of files using pyFAI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the processing for one file is established, one can loop over a
bunch of files. A convienient way to get the list of files matching a
pattern is with the *glob* module.

Most of the time, the azimuthal integrator is obtained by simply loading
the *poni-file* into pyFAI and use it directly.

.. code:: python

    import glob
    all_files = glob.glob("al2o*.edf.bz2")
    all_files.sort()
    print(len(all_files))


.. parsed-literal::

    51


.. code:: python

    ai = pyFAI.load("al2o3_00_max_51_frames.poni")
    print(ai)


.. parsed-literal::

    Detector Detector	 Spline= /users/kieffer/workspace-400/pyFAI/doc/source/usage/tutorial/Introduction/distorsion_2x2.spline	 PixelSize= 1.034e-04, 1.025e-04 m
    Wavelength= 7.084811e-11m
    SampleDetDist= 1.168599e-01m	PONI= 5.295653e-02, 5.473342e-02m	rot1=0.015821  rot2= 0.009404  rot3= 0.000000 rad
    DirectBeamDist= 116.880mm	Center: x=515.795, y=522.995 pix	Tilt=1.055 deg  tiltPlanRotation= 149.271 deg


.. code:: python

    %%time
    for one_file in all_files:
        destination = os.path.splitext(one_file)[0]+".dat"
        image = fabio.open(one_file).data
        ai.integrate1d(image, 1000, filename=destination)


.. parsed-literal::

    CPU times: user 35.6 s, sys: 272 ms, total: 35.8 s
    Wall time: 22 s


This was a simple integration of 50 files, saving the result into 2
column ASCII files.

Conclusion
----------

Using the notebook is rather simple as it allows to mix comments, code,
and images for visualization of scientific data.

The basic use pyFAI's AzimuthalIntgrator has also been presented and may
be adapted to you specific needs.
