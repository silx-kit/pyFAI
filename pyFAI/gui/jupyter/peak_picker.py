#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Semi-graphical tool for peak-picking and extracting visually control points
from an image with Debye-Scherer rings in Jupyter environment"""

__authors__ = ["Philipp Hans", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/07/2021"
__status__ = "production"

import logging
logger = logging.getLogger(__name__)
import numpy
from matplotlib.pyplot import subplots
from ..peak_picker import PeakPicker as _PeakPicker, preprocess_image
try:
    import ipywidgets as widgets
    from IPython.display import display
except ModuleNotFoundError:
    logger.error("`ipywidgets` and `IPython` are needed to perform the calibration in Jupyter")
    

class PeakPicker(_PeakPicker):
    """Peak picker optimized for the Jupyter environment with
    * Matplotlib
    * Ipywidgets
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = self.data
        self.figsize = (10, 8)
        self.ringintwidget = None
        
    @property
    def ring_value(self):
        if self.ringintwidget:
            return self.ringintwidget.value

    
    def gui(self, log=False, maximize=False, pick=True):
        """
        :param log: show z in log scale
        """
        ### Several nested functions:
        def undo_on_click(*kargs, **kwargs):
            with self._sem:
                lbls = self.points.get_labels()
                if lbls:
                    self.remove_grp(lbls[-1])
       
        def reset_picking_on_click(*kargs, **kwargs):
            with self._sem:
                self.reset()

        def save_ctrl_pts_on_click(b):
            with self._sem:
                filename = text_field_output_name.value
                work = self.dict_all_points
                for key in work.keys():
                    self.points.append(work[key][0], key)
                self.points.save(filename)
       
        self.fig, self.ax = subplots(figsize=self.figsize)

        self.ringintwidget = widgets.IntText(description="Ring #", continuous_update=True)

        button_undo = widgets.Button(description='undo')
        button_undo.on_click(undo_on_click)
                
        button_reset = widgets.Button(description='reset')
        button_reset.on_click(reset_picking_on_click)
        
        text_field_output_name = widgets.Text(
            value='./controlpoint_file.npt',
            #placeholder='Type something',
            description='filename:',
            disabled=False
        )        
        button_save_ctrl_pts = widgets.Button(description='Save control-points')
        button_save_ctrl_pts.on_click(save_ctrl_pts_on_click)
        
        layout = widgets.VBox([
                                widgets.HBox([self.ringintwidget, button_undo, button_reset]), 
                                widgets.HBox([text_field_output_name, button_save_ctrl_pts])])
        _ = display(layout)
                
        disp_image, bounds = preprocess_image(self.data, log)
        show_min, show_max = bounds 
        self.im = self.ax.imshow(disp_image, 
                                  cmap='magma', 
                                  origin="lower",
                                  vmin=show_min, 
                                  vmax=show_max,
                                  )
        self.ax.set_ylabel('y in pixels')
        self.ax.set_xlabel('x in pixels')
        self.ax.set_title("Shift+click to select a ring. Mind to first set the ring number")
        self.ax.images[0].set_picker(True)
        self.cid = self.fig.canvas.mpl_connect("pick_event", self.onclick)

    def onclick(self, event):
        
        def annontate(x, x0=None, idx=None, gpt=None):
            """
            Call back method to annotate the figure while calculation are going on ...
            :param x: coordinates
            :param x0: coordinates of the starting point
            :param gpt: group of point, instance of PointGroup
            :return: annotatoin
            """
            if x0 is None:
                annot = self.ax.annotate(".", xy=(x[1], x[0]), weight="bold", size="large", color="black")
            else:
                if gpt:
                    annot = self.ax.annotate(gpt.label, xy=(x[1], x[0]), xytext=(x0[1], x0[0]),
                                             weight="bold", size="large", color="black",
                                             arrowprops=dict(facecolor='white', edgecolor='white'),)
                    gpt.annotate = annot
                else:
                    annot = self.ax.annotate("%i" % (len(self.points)), xy=(x[1], x[0]), xytext=(x0[1], x0[0]),
                                             weight="bold", size="large", color="black",
                                             arrowprops=dict(facecolor='white', edgecolor='white'),)
#                 self.draw()
            return annot
        
        def create_gpt(points, gpt=None):
            """
            plot new set of points

            :param points: list of points
            :param gpt: : group of point, instance of PointGroup
            """
            if points:
                if not gpt:
                    gpt = self.points.append(points, ring=self.ring_value)
                npl = numpy.array(points)
                gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
#                 self.draw()
            return gpt

        def common_creation(points, gpt=None):
            """
            plot new set of points

            :param points: list of points
            :param gpt: : group of point, instance of PointGroup
            """
            if points:
                if not gpt:
                    gpt = self.points.append(points, ring=self.ring_value)
                npl = numpy.array(points)
                gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
#                 self.draw()
            return gpt


        def new_grp(event):
            " * new_grp Right-click (click+n):         try an auto find for a ring"
            # ydata is a float, and matplotlib display pixels centered.
            # we use floor (int cast) instead of round to avoid use of
            # banker's rounding
            ypix, xpix = int(event.ydata + 0.5), int(event.xdata + 0.5)
            points = self.massif.find_peaks([ypix, xpix],
                                            self.defaultNbPoints,
                                            None, None)
            if points:
                gpt = common_creation(points)
                annontate(points[0], [ypix, xpix], gpt=gpt)
                self.draw()
                print(f"Created group {gpt.label} with {len(gpt)} points")
            else:
                print("No peak found !!!")

        ## Core part of the function
        with self._sem:
            event = event.mouseevent
            logger.debug("Button: %i, Key modifier: %s", event.button, event.key)
            if (event.inaxes and event.button == 1 and event.key == 'shift'):
                new_grp(event)

    def remove_grp(self, lbl):
        """
        remove a group of points

        :param lbl: label of the group of points
        """
        gpt = self.points.pop(lbl=lbl)
        if gpt and self.ax:
            if gpt.annotate in self.ax.texts:
                self.ax.texts.remove(gpt.annotate)
            for plot in gpt.plot:
                    if plot in self.ax.lines:
                        self.ax.lines.remove(plot)
            self.draw()


    def reset(self):
        """
        Reset control point and graph (if needed)
        """
        self.points.reset()
        if self.fig and self.ax:
            # empty annotate and plots
            if len(self.ax.texts) > 0:
                self.ax.texts = []
            if len(self.ax.lines) > 0:
                self.ax.lines = []
            # Redraw the image
            self.draw()
            
    def draw(self):
        """Update plot"""
        if self.fig is not None:
            self.fig.canvas.draw_idle()