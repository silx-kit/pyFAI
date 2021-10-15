#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
pyFAI-calib: The Matplotlib-CLI visualization widget

A tool for determining the geometry of a detector using a reference sample.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/10/2021"
__status__ = "development"


import logging
import gc
from collections import namedtuple
logger = logging.getLogger(__name__)
import numpy
try:
    from silx.gui import qt
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    qt = None

if qt is not None:
    from .utils import update_fig, maximize_fig
    from .matplotlib import matplotlib, pyplot, colors
    from . import utils as gui_utils

GroupOfPoints = namedtuple("GroupOfPoints", "annotation plot")


class MplCalibWidget:
    """Provides a unified interface for:
    - display the image
    - Plot group of points
    - overlay the contour-plot for 2th values
    - shade out some part of the image
    - annotate rings
    """
    def __init__(self, 
                 click_cb=None,
                 refine_cb=None,
                 option_cb=None,
                 ):
        """
        
        :param click_cb: function to be called when clicking on the plot
        :param refine_cb: function to be called when clicking of the refine button 
        :param: option_cb: function to be called when clicking on the option button 
        """
        #store the various callbacks
        self.click_cb = click_cb
        self.option_cb = option_cb
        self.refine_cb = refine_cb
        
        self.shape = None
        self.fig = None
        self.ax = None  # axe for the plot
        self.axc = None # axe for the colorbar
        self.axd = None # axe for the detector coordinate
        
        self.spinbox = None
        self.sb_action = None
        self.ref_action = None
        self.mpl_connectId = None
        self.append_mode = None
        #This is for tracking the different types of plots  
        self.background = None
        self.overlay = None
        self.points = {} #key: label, value (
                
    def init(self, pick=True, update=True):
        if self.fig is None:
            self.fig, (self.ax, self.axc) = pyplot.subplots(1, 2, gridspec_kw={"width_ratios": (10,1)})
            self.ax.set_ylabel('y in pixels')
            self.ax.set_xlabel('x in pixels')
            # self.axc.yaxis.set_label_position('left')
            # self.axc.set_ylabel("Colorbar")
            toolbar = self.fig.canvas.toolbar
            toolbar.addSeparator()
            a = toolbar.addAction('Opts', self.on_option_clicked)
            a.setToolTip('open options window')
            if pick:
                label = qt.QLabel("Ring #", toolbar)
                toolbar.addWidget(label)
                self.spinbox = qt.QSpinBox(toolbar)
                self.spinbox.setMinimum(0)
                self.sb_action = toolbar.addWidget(self.spinbox)
                a = toolbar.addAction('Refine', self.on_refine_clicked)
                a.setToolTip('switch to refinement mode')
                self.ref_action = a
                self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            if update:
                self.fig.show()
        elif update:
            self.update()
    
    def set_title(self, text):
        self.ax.set_title(text)
        
    def set_detector(self, detector=None, update=True):
        """Add a second axis to have the distances in meter on the detector
        """
        if self.fig is None:
            self.init()
        
        s1, s2 = detector.shape
        s1 -= 1
        s2 -= 1
        self.ax.set_xlim(0, s2)
        self.ax.set_ylim(0, s1)
        d1 = numpy.array([0, s1, s1, 0])
        d2 = numpy.array([0, 0, s2, s2])
        p1, p2, _ = detector.calc_cartesian_positions(d1=d1, d2=d2)
        self.axd = self.fig.add_subplot(1, 2, 1,
                                      xbound=False,
                                      ybound=False,
                                      xlabel=r'dim2 ($\approx m$)',
                                      ylabel=r'dim1 ($\approx m$)',
                                      xlim=(p2.min(), p2.max()),
                                      ylim=(p1.min(), p1.max()),
                                      aspect='equal',
                                      zorder=-1)
        self.axd.xaxis.set_label_position('top')
        self.axd.yaxis.set_label_position('right')
        self.axd.yaxis.label.set_color('blue')
        self.axd.xaxis.label.set_color('blue')
        self.axd.tick_params(colors="blue", labelbottom='off', labeltop='on',
                             labelleft='off', labelright='on')
        if update:
            self.update()
    
    def imshow(self, img, bounds=None, log=False, update=True):
        """Display a 2Dscattering image
        
        :param img: preprocessed image
        :param bounds: 2-tuple with (vmin, vmax)
        :param log: display data in log-space or linear space
        :param update: update ime
        """
        if self.fig is None:
            self.init()

        if bounds:
            show_min, show_max = bounds
        else:
            show_min = numpy.nanmin(img)
            show_max = numpy.nanmax(img)
        
        if log:
            if show_min<=0:
                show_min = img[numpy.where(img>0)].min()
            norm = colors.LogNorm(show_min, show_max)
        else:
            norm = colors.Normalize(show_min, show_max)
        
        self.background = self.ax.imshow(img, norm=norm,
                                         origin="lower", 
                                         interpolation="nearest")
        s1, s2 = self.shape = img.shape        
        mask = numpy.zeros(self.shape+(4,), "uint8")
        self.overlay = self.ax.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")

        pyplot.colorbar(self.background, cax=self.axc)
        self.axc.yaxis.set_label_position('left')
        self.axc.set_ylabel("Colorbar")
        s1 -= 1
        s2 -= 1
        self.ax.set_xlim(0, s2)
        self.ax.set_ylim(0, s1)

        if update:
            self.update()

    def shadow(self, mask=None, update=True):
        """Apply som shadowing overlay on top of background image
        
        :param mask: mask to be overlaid. set to None to remove
        :param update: finally update the plot
        """
        if self.shape is None:
            logger.warning("No background image defined !")
            return
        
        shape = self.shape+ (4,)
        
        if mask is not None:
            assert mask.shape == self.shape
            mask4 = numpy.outer(numpy.logical_not(mask), 100*numpy.ones(4)).astype(numpy.uint8).reshape(shape)
        else:
            mask4 = numpy.zeros(shape, dtype=numpy.uint8)
        self.overlay.set_data(mask4)
        if update: 
            self.update()

    def contour(self, data, values=None, cmap="autumn", linewidths=2, linestyles="dashed", update=True):
        """
        Overlay a contour-plot of data 

        :param data: 2d-array with the 2theta values in radians...
        :param values: 1d-array with numerical values of the rings
        :param update: finally update the plot 
        """
        if self.fig is None:
            logging.warning("No diffraction image available => not showing the contour")
            return
        #clean previous contour plots:
        while len(self.ax.collections) > 0:
            self.ax.collections.pop()
        if data is not None:
            try:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                if not isinstance(cmap, matplotlib.colors.Colormap):
                    cmap = matplotlib.cm.get_cmap(cmap)
                
                self.ax.contour(data, levels=values, cmap=cmap, linewidths=linewidths, linestyles=linestyles)
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the 2-theta contour plot")
            except ValueError:
                logging.error("No contour-plot to display !")
        if update:
            self.update()

    def reset(self, update=True):
        """
        Reset all control point and annotation from graph
        
        :param update: finally update the plot
        """
        if self.fig:
            # empty annotate and plots
            if len(self.ax.texts) > 0:
                self.ax.texts = []
            if len(self.ax.lines) > 0:
                self.ax.lines = []
            # Redraw the image
            if update:
                if not gui_utils.main_loop:
                    self.fig.show()
                self.update()

    def add_grp(self, label, points, update=True):
        """
        Append a group of points to the graph with its annotations
        
        :param label: string with the label
        :param points: list of coordinates [(y,x)]
        :param update: finally update the plot 
        """
        if self.fig:
            if len(points):
                pt0y, pt0x = points.points[0]
                annotate = self.ax.annotate(label, xy=(pt0x, pt0y), xytext=(pt0x + 10, pt0y + 10),
                                                    weight="bold", size="large", color="black",
                                                    arrowprops=dict(facecolor='white', edgecolor='white'))
                npl = numpy.array(points)
                plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False, label=label)
                self.points[label] = GroupOfPoints(annotate, plot)
            if update:
                self.update()
                

    def remove_grp(self, label, update=True):
        """
        remove a group of points

        :param label: label of the group of points
        :param update: finally update the plot 
        """
        if self.fig and label in self.points:
            gpt = self.points[label]
            if gpt.annotate in self.ax.texts:
                self.ax.texts.remove(gpt.annotate)
            for plot in gpt.plot:
                    if plot in self.ax.lines:
                        self.ax.lines.remove(plot)
            if update:
                self.update()


    def onclick(self, *args):
        """
        Called when a mouse is clicked
        """
        if self.click_cb:
            self.click_cb(*args)
    
    def on_plus_pts_clicked(self, *args):
        """
        callback function
        """
        self.append_mode = True

    def on_minus_pts_clicked(self, *args):
        """
        callback function
        """
        self.append_mode = False

    def on_option_clicked(self, *args):
        """
        callback function
        """
        if self.option_cb:
            self.option_cb(*args) 

    def on_refine_clicked(self, *args):
        """
        callback function
        """
        self.sb_action.setDisabled(True)
        self.ref_action.setDisabled(True)
        self.spinbox.setEnabled(False)
        self.mpl_connectId = None
        self.fig.canvas.mpl_disconnect(self.mpl_connectId)
        pyplot.ion()
        if self.refine_cb:
            self.refine_cb(*args)
    
    def update(self):
        if self.fig:
            update_fig(self.fig)
            
    def maximize(self):
        if self.fig:
            maximize_fig(self.fig)
    
    def close(self):
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
            gc.collect()
