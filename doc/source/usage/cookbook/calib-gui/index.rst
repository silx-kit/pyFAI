:Author: Jérôme Kieffer
:Date: 18/10/2018
:Keywords: Tutorials
:Target: Scientists

Calibration of a diffraction setup using the Graphical User Interface
=====================================================================

The graphical tool for geometry calibration is called **pyFAI-calib2**,
just open a terminal and type its name plus return to startup the application
which looks like this:

.. code-block:: shell

   $ pyFAI-calib2

.. figure:: 0_startup.png
   :align: center
   :alt: startup of calib2

The windows is divided in 3 vertical tiles containing:

* On the left side, a list of tasks to be performed: setting-up, masking,
  peak-picking, ring-fitting and caking.
* A large central panel to display the image
* The right side contains a pannel with a set of tools to perform each task and
  finishes with the "Next" button to switch to the next task.

We will now describe shortly every task by following a simple example.

Experiment settings
-------------------

This task contains all entries relative to the experiment setup.
It is the place where you can enter the energy of the beam (or the wavelength),
the calibrant and the detector used.

You have to provide the calibration image which is then displayed in the cental panel.
For example the file used in this cookbook can be downloaded from this link:
`Eiger4M_Al2O3_13.45keV.edf <http://www.silx.org/pub/pyFAI/cookbook/calibration/Eiger4M_Al2O3_13.45keV.edf>`_
Click on the ... next to "Image file" to open a file-browser and select the file you just downloaded.

Finally, select the type of detector, here the image has been acquired using an Eiger4M manufactured by Dectris.
Once the detector is selected, its shape is overlaid to the image allowing a direct validation of the detector size.
Mask and dark-current files can be provided.

.. figure:: 1_experiment.png
   :align: center
   :alt: Experiment setup
   
Finally set the energy to 13.45 keV and select the calibrant:
corundum which is alpha Al2O3 as in the figure.

You are now ready to define the mask. click on Next

Defining the mask
-----------------

The mask defines the list of pixels which are considered invalid.
It is displayed in green by default.
There are 3 drawing tools:

* rectangular selection
* polygonal selection (click on the first point to finish edition)
* Pencil selection

In addition, the mask can often be setup easily using thresholds.
In our test-image the Eiger detector has flagged overflow pixel with a very high value.
Discarding pixel above 65000 removes automatically all invalid pixels.
To remove pixels with some shadow, like the beam stop, the easiest is to use the
polygon tool as in the figure bellow:
The mask can be saved

.. figure:: 2_mask.png
   :align: center
   :alt: Draw a mask on the image
   
Once you are done, click next to the peak-picking.

Peak-picking and ring assignment
--------------------------------

The geometry calibration is performed in pyFAI by a least-square fitting of peak
position on ring number.

This step consists in selecting groups of peaks and to
assign them to a ring number.
You will need to pick least two rings, I advice
you to select the first group on the inner-most ring and a couple of other rings.

.. figure:: 3_picking.png
   :align: center
   :alt: Select peaks and assign them to rings 


Double check the ring-number assignment: "pyFAI-calib2" uses 1-based numbers,
unlike the command line tool.

We will skip the "recalibrate" tool for now and go directly to the
geometry fitting by clicking on "Next"

Geometry fitting
----------------

When arriving on this task, the geometry is immediately fitted and you can see
the values of the 3 translation and 3 rotation.

.. figure:: 4_geometry.png
   :align: center
   :alt: Geometry optimization 


Values can be modified and fixed by clicking on the lock.
Click on the "Fit" button to re-fit the geometry.

Results may be displayed in various units by right-clicking on the unit.

Depending on the result, you may want to come back on the "peak-picking" task to
re-assign the ring number.
Or if it looks good, you can extract many ring for an even better fit like in this figure:

.. figure:: 3_extract.png
   :align: center
   :alt: Extract many more rings for geometry refinement 
 
Cake and Integration
--------------------

The last task displays the 1D and 2D integrated image with the ring position
overlaid to validate the quality of the calibration.

The radial unit can be customized and the images can be saved.

.. figure:: 5_cake.png
   :align: center
   :alt: Azimuthal integration 

Last but not least, do not forget to save the geometry as a PONI-file.

Conclusion
----------

This tutorial explained the 5 steps needed to perform the calibration of the
detector position prior to any diffraction experiment on a synchrotron.
