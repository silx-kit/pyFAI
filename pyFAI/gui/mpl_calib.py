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
__date__ = "17/01/2022"
__status__ = "development"

import logging
import threading
import contextlib
import gc
from collections import namedtuple
logger = logging.getLogger(__name__)
import numpy
import matplotlib
from matplotlib import pyplot, colors
from ..utils.callback import dangling_callback
GroupOfPoints = namedtuple("GroupOfPoints", "annotate plot")


class MplCalibWidget:
    """Provides a unified interface for:
    - display the image
    - Plot group of points
    - overlay the contour-plot for 2th values
    - shade out some part of the image
    - annotate rings
    """

    def __init__(self,
                 new_grp_cb=dangling_callback,
                 append_single_cb=dangling_callback,
                 single_point_cb=dangling_callback,
                 append_more_cb=dangling_callback,
                 erase_pt_cb=dangling_callback,
                 erase_grp_cb=dangling_callback,
                 refine_cb=dangling_callback,
                 option_cb=dangling_callback,
                 ):
        """
        
        :param new_grp_cb: function to be called when new group of point needs to be added
        :param append_single_cb: function to be called when one point is added to the current group
        :param single_point_cb: function to be called to create a new group with a single point
        :param append_more_cb: function to be called to append more points to a group 
        :param erase_pt_cb: function to be called to delete one point 
        :param erase_grp_cb: function to be called to delete one group
        :param refine_cb: function to be called when clicking of the refine button
        :param option_cb:   function to be called when clicking of the option button
        """
        # store the various callbacks
        self.cb_new_grp = new_grp_cb
        self.cb_append_1_point = append_single_cb
        self.cb_single_point = single_point_cb
        self.cb_append_more_points = append_more_cb
        self.cb_erase_1_point = erase_pt_cb
        self.cb_erase_grp = erase_grp_cb
        self.cb_refine = refine_cb
        self.cb_option = option_cb

        # This is a dummy context-manager, used in Jupyter environment only
        self.mplw = contextlib.suppress()
        self.mpl_connectId = None  # Callback id from matplotlib is stored here

        self.shape = None
        self.fig = None
        self.ax = None  # axe for the plot
        self.axc = None  # axe for the colorbar
        self.axd = None  # axe for the detector coordinate

        self.spinbox = None
        self.sb_action = None
        self.ref_action = None
        self.mpl_connectId = None
        self.append_mode = None
        # This is for tracking the different types of plots
        self.background = None
        self.overlay = None
        self.foreground = None
        self.points = {}  # key: label, value (
        self._sem = threading.Semaphore()
        self.msg = []

    def set_title(self, text):
        self.ax.set_title(text)

    def set_detector(self, detector=None, update=True):
        """Add a second axis to have the distances in meter on the detector
        """
        if self.fig is None:
            self.init()
        if detector is not None:
            if detector.shape is None:
                s1, s2 = self.shape
            else:
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
            self.axd.set_xlabel(r'dim2 ($\approx m$)')
            self.axd.set_ylabel(r'dim1 ($\approx m$)')
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
            self.init(pick=True)
            init = True
        else:
            init = False

        if bounds:
            show_min, show_max = bounds
        else:
            show_min = numpy.nanmin(img)
            show_max = numpy.nanmax(img)

        if log:
            if show_min <= 0:
                show_min = img[numpy.where(img > 0)].min()
            norm = colors.LogNorm(show_min, show_max)
            txt = 'Log colour scale (skipping lowest/highest per mille)'
        else:
            norm = colors.Normalize(show_min, show_max)
            txt = 'Linear colour scale (skipping lowest/highest per mille)'
        with self.mplw:
            with pyplot.ioff():
                self.background = self.ax.imshow(img, norm=norm,
                                                 cmap="inferno",
                                                 origin="lower",
                                                 interpolation="nearest")
                s1, s2 = self.shape = img.shape
                mask = numpy.zeros(self.shape + (4,), "uint8")
                self.overlay = self.ax.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")
                self.foreground = self.ax.imshow(img, norm=norm,
                                                 origin="lower",
                                                 interpolation="nearest", alpha=0)
                pyplot.colorbar(self.background, cax=self.axc)  # , label=txt)
                self.axc.yaxis.set_label_position('left')
                self.axc.set_ylabel(txt)
                s1 -= 1
                s2 -= 1
                self.ax.set_xlim(0, s2)
                self.ax.set_ylim(0, s1)

        if update:
            if init:
                self.show()
            else:
                self.update()

    def shadow(self, mask=None, update=True):
        """Apply som shadowing overlay on top of background image
        
        :param mask: mask to be overlaid. set to None to remove
        :param update: finally update the plot
        """
        if self.shape is None:
            logger.warning("No background image defined !")
            return

        shape = self.shape + (4,)

        if mask is not None:
            assert mask.shape == self.shape
            mask4 = numpy.outer(numpy.logical_not(mask), 100 * numpy.ones(4)).astype(numpy.uint8).reshape(shape)
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
        # clean previous contour plots:
        self.ax.collections.clear()
        if data is not None:
            try:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                if not isinstance(cmap, colors.Colormap):
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
                self.ax.texts.clear()
            if len(self.ax.lines) > 0:
                self.ax.lines.clear()
            # Redraw the image
            if update:
                # TODO: fix this
                # if not gui_utils.main_loop:
                #     self.fig.show()
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
                pt0y, pt0x = points[0]
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

    def onclick(self, event):
        """
        Called when a mouse is clicked: distribute the call to different functions
        """
        with self._sem:
            logger.info(f"Button: {event.button}, Key modifier: {event.key}")
            button = event.button
            key = event.key
            ring = self.get_ring_value()

            if (event.xdata and event.ydata) is None:  # no coordinates
                logger.info("invalid coodinates")
                return
            yx = int(event.ydata + 0.5), int(event.xdata + 0.5)

            if ((button == 3) and (key == 'shift')) or \
               ((button == 1) and (key == 'v')):
                # if 'shift' pressed add nearest maximum to the current group
                self.cb_append_1_point(yx, ring)
            elif ((event.button == 3) and (event.key == 'control')) or\
                 ((event.button == 1) and (event.key == 'b')):
                # if 'control' pressed add nearest maximum to a new group
                self.cb_single_point(yx, ring)
            elif (event.button in [1, 3]) and (event.key == 'm'):
                self.cb_append_more_points(yx, ring)
            elif (event.button == 3) or ((event.button == 1) and (event.key == 'n')):
                # create new group
                self.cb_new_grp(yx, ring)
            elif (event.key == "1") and (event.button in [1, 2]):
                self.cb_erase_1_point(yx, ring)
            elif (event.button == 2) or (event.button == 1 and event.key == "d"):
                self.cb_erase_grp(yx, ring)
            else:
                logger.warning(f"Unknown combination: Button: {button}, Key modifier: {key} at x:{yx[1]} y:{yx[0]}")

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

    def onclick_option(self, *args):
        """
        callback function
        """
        self.cb_option(*args)

    def onclick_refine(self, *args):
        """
        callback function
        """
        self.sb_action.setDisabled(True)
        self.ref_action.setDisabled(True)
        self.spinbox.setEnabled(False)
        self.finish()
        self.cb_refine(*args)

    def close(self):
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
            gc.collect()

    def finish(self):
        """Stop managing interaction with display"""
        if (self.fig is not None) and (self.mpl_connectId is not None):
            self.fig.canvas.mpl_disconnect(self.mpl_connectId)
            self.mpl_connectId = None

    def show(self):
        """Show the widget"""
        if self.fig is not None:
            self.fig.show()

    # Those methods need to be spacialized:
    def init(self, pick=True, update=True):
        import inspect
        raise NotImplementedError("MplCalibWidget is an Abstract class, {inspect.currentframe().f_code.co_name} not defined!")

    def update(self):
        import inspect
        raise NotImplementedError("MplCalibWidget is an Abstract class, {inspect.currentframe().f_code.co_name} not defined!")

    def maximize(self, update=True):
        import inspect
        raise NotImplementedError("MplCalibWidget is an Abstract class, {inspect.currentframe().f_code.co_name} not defined!")

    def get_ring_value(self):
        import inspect
        raise NotImplementedError("MplCalibWidget is an Abstract class, {inspect.currentframe().f_code.co_name} not defined!")
