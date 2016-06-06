
Selection of a calibrant
========================

In this tutorial we will see how to select a calibrant for a given
experimental setup.

Experimental setup
------------------

The experimental setup is a classical protein crystallography setup
with:

-  Large Pilatus 6M detector on a translation table
-  The small and intense beam of ~50 microns in size has a wavelength of
   1 Angstrom
-  The detector is in *normal* condition: orthogonal to the beam and
   centered in the middle of the detector.

The scientist in charge of this beamline want to ensure all encoders are
working properly and needs to validate the setup for distances between
10cm and 80cm. He will buy reference material from NIST but he does not
know which calibrant is best suited for his setup. We will assume all
reference material sold by NIST are equally suitable for ray position
(no issue with grain size, ...).

The calibration works best in pyFAI if more than one Debye-Scherrer ring
is seen on the detector.

Define the detector
-------------------

.. code:: python

    import pyFAI
    dete = pyFAI.detectors.Pilatus6M()
    print(dete)

.. parsed-literal::

    Detector Pilatus6M	 PixelSize= 1.720e-04, 1.720e-04 m


Select reference materials
--------------------------

The NIST sells different `Standard Refrence
Materials <http://www.nist.gov/mml/mmsd/upload/852_08_81b.pdf>`__, among
them Silicon (SRM640), Lanthanum hexaboride (SRM660), Alumina (SRM676)
and Ceria (SRM674) are commonly used. Many other exists: Cr203, TiO2,
Zn0, SiO2, ... Evaluating them is left as an exercise.

.. code:: python

    import pyFAI.calibrant
    print(pyFAI.calibrant.ALL_CALIBRANTS)

.. parsed-literal::

    Calibrants available: Ni, CrOx, NaCl, Si_SRM640e, Si_SRM640d, Si_SRM640a, Si_SRM640c, alpha_Al2O3, Cr2O3, AgBh, Si_SRM640, CuO, PBBA, Si_SRM640b, quartz, C14H30O, cristobaltite, Si, LaB6, CeO2, LaB6_SRM660a, LaB6_SRM660b, LaB6_SRM660c, TiO2, ZnO, Al, Au


You may wonder where the names of the calibrant came from and how they
have been established.

The name of all calibrant available in your version of pyFAI can be
listed by just printing out the content of ALL\_CALIBRANTS. New
calibrant may have been added in pyFAI in more recent releases, just
have a look at the `developent web
page <https://github.com/kif/pyFAI/tree/master/calibration>`__.

Most of those calibrant files, which contain the *d-spacing* in Angstrom
between Miller plans, have been prepared from the unit cell of the
compount, found in publication. This publication is referenced in the
header of the file. If one wishes to regenerate those files, the
*pyFAI.calibrant.Cell* class may be used for.

We will now focus on a subset of calibrant, instanciate them and put
them into a dictionnary. The Python construct used here is called
*dict-comprehension* and allows the creation and population of a
dictionnary in a single line.

.. code:: python

    cals = dict((name,pyFAI.calibrant.ALL_CALIBRANTS(name)) for name in ("Si", "LaB6", "CeO2", "alpha_Al2O3"))
    print(cals)

.. parsed-literal::

    {'Si': Si Calibrant , 'alpha_Al2O3': alpha_Al2O3 Calibrant , 'LaB6': LaB6 Calibrant , 'CeO2': CeO2 Calibrant }


To be able to use those calibrants, one needs to define the wavelength
used, here 1 Angstrom.

.. code:: python

    wl = 1e-10
    for cal in cals.values():
        cal.wavelength = wl
    print(cals)

.. parsed-literal::

    {'Si': Si Calibrant at wavelength 1e-10, 'alpha_Al2O3': alpha_Al2O3 Calibrant at wavelength 1e-10, 'LaB6': LaB6 Calibrant at wavelength 1e-10, 'CeO2': CeO2 Calibrant at wavelength 1e-10}


Short distance images
---------------------

The shortest the detector can come to the sample is about 10cm (to leave
space for the beamstop). We will generate images of diffraction at this
distance.

For the display of images we will use *matplotlib* inlined.

.. code:: python

    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()
    print("Detector center at %s, %s"%(poni1, poni2))
    ai_short = pyFAI.AzimuthalIntegrator(dist=0.1, poni1=poni1, poni2=poni2,detector=dete)
    print(ai_short)

.. parsed-literal::

    Detector center at 0.217322000001, 0.211818
    Detector Pilatus6M	 PixelSize= 1.720e-04, 1.720e-04 m
    SampleDetDist= 1.000000e-01m	PONI= 2.173220e-01, 2.118180e-01m	rot1=0.000000  rot2= 0.000000  rot3= 0.000000 rad
    DirectBeamDist= 100.000mm	Center: x=1231.500, y=1263.500 pix	Tilt=0.000 deg  tiltPlanRotation= 0.000 deg


.. code:: python

    %pylab inline

.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    fig = figure(figsize=(10,10))
    for idx, key in enumerate(cals):
        cal = cals[key]
        subplot(2,2,idx+1)
        img = cal.fake_calibration_image(ai_short)
        imshow(img)
        title(key)


.. image:: output_11_0.png


As one can see, there are plenty of rings on the image: it should be
easy to calibrate. By moving further away the detector, the number of
rings will decrease.

Long distance images
--------------------

To keep good calibration one should have at lease two rings for the
calibration. The longest distance from sample to the detector is 80cm.

.. code:: python

    ai_long = pyFAI.AzimuthalIntegrator(dist=0.8, poni1=poni1, poni2=poni2,detector=dete)
    print(ai_long)

.. parsed-literal::

    Detector Pilatus6M	 PixelSize= 1.720e-04, 1.720e-04 m
    SampleDetDist= 8.000000e-01m	PONI= 2.173220e-01, 2.118180e-01m	rot1=0.000000  rot2= 0.000000  rot3= 0.000000 rad
    DirectBeamDist= 800.000mm	Center: x=1231.500, y=1263.500 pix	Tilt=0.000 deg  tiltPlanRotation= 0.000 deg


.. code:: python

    fig = figure(figsize=(10,10))
    for idx, key in enumerate(cals):
        cal = cals[key]
        subplot(2,2,idx+1)
        img = cal.fake_calibration_image(ai_long)
        imshow(img)
        title(key)


.. image:: output_14_0.png


The most adapted calibrant is probably the *LaB6* as 2 rings are still
visible at 80 cm from the detector.

Integration of the pattern for the two extreme cases
----------------------------------------------------

We can integrate the image for the two extreme cases:

.. code:: python

    lab6 = cals["LaB6"]
    ai_short.wavelength = ai_long.wavelength = wl
    
    fig = figure(figsize=(10,10))
    ax = subplot(2,2,1)
    img_short = lab6.fake_calibration_image(ai_short)
    ax.imshow(img_short)
    ax = subplot(2,2,2)
    ax.plot(*ai_short.integrate1d(img_short,1000))
    ax.set_xlabel("q nm^-1")
    ax = subplot(2,2,3)
    img_long = lab6.fake_calibration_image(ai_long)
    ax.imshow(img_long)
    ax = subplot(2,2,4)
    ax.plot(*ai_long.integrate1d(img_long,1000))
    ax.set_xlabel("q nm^-1")



.. parsed-literal::

    <matplotlib.text.Text at 0x7f1b7da2de50>




.. image:: output_16_1.png


Conclusion
----------

The best calibrant in this case is probably LaB6.

