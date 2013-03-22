#!/usr/bin/env python
# coding: utf8
"""
PyMca plugin for pyFAI azimuthal integrator in a LImA ProcessLib.

Destination path:
Lima/tango/plugins/AzimuthalIntegration
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "21/03/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import os, json
import sys
import threading
import logging
logger = logging.getLogger("lima.tango.pyfai")
# set loglevel at least at INFO
if logger.getEffectiveLevel() > logging.INFO:
    logger.setLevel(logging.INFO)
import numpy
try:
    import pyFAI
    import pyFAI.integrate_widget
except:
    pass
try:
    from PyMca import StackPluginBase
except ImportError:
    from . import StackPluginBase

DEBUG = 0

class AzimuthalIntegrationPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show':[self._showWidget,
                                   "Setup Azimuthal Intgration Filter",
                                   None]}
        self.__methodKeys = ['Show']
        self.widget = None

    def stackUpdated(self):
        if DEBUG:
            print("StackBrowserPlugin.stackUpdated() called")
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        stack = self.getStackDataObject()
        self.widget.setStackDataObject(stack, stack_name="Stack Index")
        self.widget.setBackgroundImage(self._getBackgroundImage())
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def _getBackgroundImage(self):
        images, names = self.getStackROIImagesAndNames()
        B = None
        for key in names:
            if key.endswith("ackground"):
                B = images[names.index(key)]
        return B

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def stackROIImageListUpdated(self):
        if self.widget is None:
            return
        self.widget.setBackgroundImage(self._getBackgroundImage())

    def mySlot(self, ddict):
        if DEBUG:
            print("mySlot ", ddict['event'], ddict.keys())
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _showWidget(self):
        if self.widget is None:
            self.widget = pyFAI.integrate_widget.AIWidget()
            #window.set_input_data(args)
            #window.show()
#             = Median2DBrowser.Median2DBrowser(parent=None,
#                                                    rgbwidget=None,
#                                                    selection=True,
#                                                    colormap=True,
#                                                    imageicons=True,
#                                                    standalonesave=True,
#                                                    profileselection=True)
#            self.widget.setKernelWidth(1)
#            self.widget.setSelectionMode(True)
#            qt = Median2DBrowser.qt
#            qt.QObject.connect(self.widget,
#                   qt.SIGNAL('MaskImageWidgetSignal'),
#                   self.mySlot)

        #Show
        self.widget.show()
        self.widget.raise_()

        #update
        self.stackUpdated()


MENU_TEXT = "Azimuthal Integration Filter"
def getStackPluginInstance(stackWindow, **kw):
    ob = AzimuthalIntegrationPlugin(stackWindow)
    return ob
