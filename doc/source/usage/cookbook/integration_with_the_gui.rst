Azimuthal integration using the graphical user interface
========================================================

Author: Jérôme Kieffer

Date: 20/01/2015

Keywords: Integration

Target: Scientists

Associated video:
  http://www.edna-site.org/pub/calibration/integration.flv

1. Look at your integrated patterns
-----------------------------------
PyFAI can perform 1D or 2D integration.
To view 1D patterns, I will use *grace*
Let's look at the integrated patterns
obtained during calibration.

As you can see, only the 10 rings used for
calibration are well defined.

This file is a text file containing as header
all metadata needed to determine the geometry.

2D integrated pattern (aka cake images)
are EDF images. Try *fabio_viewer* to see them.

Once again check the header of the file and the
associated metadata.

2. Integrate a bunch of images
------------------------------
We will work with the 20 images used for the calibration.

3. Start pyFAI-intgrate
-----------------------
Either select files to process using the file-dialog or provide
them on the command line.

4. Set the geometry
-------------------
Simply load the PONI file and check the populated fields.


5. Azimuthal integration options
--------------------------------
Check the dark/flat/ ... options
Use the check-box to activate the option.

Do NOT forget to specify the number of radial bins !

6. Select the device for processing
-----------------------------------
Unless the processing will be done on the
CPU using OpenMP.

Press OK to start the processing.

The generation of the Look-Up table takes a few seconds
then all files get processed quickly

7. Run it again to perform caking
---------------------------------
Same a previously ... but
provide a number of azimuthal bins !


8. Visualize the integrated patterns
------------------------------------
Once again I used *grace* and *fabio_viewer*
to display the result.

That's all.
