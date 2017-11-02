# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/
"""
This module contains an :class:`ImageFileDialog`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "19/10/2017"

import sys
import os
import logging
import fabio
import numpy
import silx.io
from silx.gui.plot import actions
from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget
from silx.gui.hdf5.Hdf5TreeModel import Hdf5TreeModel
from . import utils
from .FileTypeComboBox import FileTypeComboBox
from pyFAI.third_party import six


_logger = logging.getLogger(__name__)


class _ImageUri(object):

    def __init__(self, path=None, filename=None, dataPath=None, slice=None):
        self.__isValid = False
        if path is not None:
            self.__fromPath(path)
        else:
            self.__filename = filename
            self.__dataPath = dataPath
            self.__slice = slice
            self.__path = None
            self.__isValid = self.__filename is not None

    def __fromPath(self, path):
        elements = path.split("::", 1)
        self.__path = path
        self.__filename = elements[0]
        self.__slice = None
        self.__dataPath = None
        if len(elements) == 1:
            pass
        else:
            selector = elements[1]
            selectors = selector.split("[", 1)
            self.__dataPath = selectors[0]
            if len(selectors) == 2:
                slicing = selectors[1].split("]", 1)
                if len(slicing) < 2 or slicing[1] != "":
                    self.__isValid = False
                    return
                slicing = slicing[0].split(",")
                try:
                    slicing = [int(s) for s in slicing]
                    self.__slice = slicing
                except ValueError:
                    self.__isValid = False
                    return

        self.__isValid = True

    def isValid(self):
        return self.__isValid

    def path(self):
        if self.__path is not None:
            return self.__path

        if self.__path is None:
            path = ""
            selector = ""
            if self.__filename is not None:
                path += self.__filename
            if self.__dataPath is not None:
                selector += self.__dataPath
            if self.__slice is not None:
                selector += "[%s]" % ",".join([str(s) for s in self.__slice])
            if selector != "":
                return path + "::" + selector
            else:
                return path

        return self.__path

    def filename(self):
        return self.__filename

    def dataPath(self):
        return self.__dataPath

    def slice(self):
        return self.__slice


class _IconProvider(object):

    FileDialogToParentDir = qt.QStyle.SP_CustomBase + 1

    FileDialogToParentFile = qt.QStyle.SP_CustomBase + 2

    def __init__(self):
        self.__iconFileDialogToParentDir = None
        self.__iconFileDialogToParentFile = None

    def _createIconToParent(self, standardPixmap):
        """

        FIXME: It have to be tested for some OS (arrow icon do not have always
        the same direction)
        """
        style = qt.QApplication.style()
        baseIcon = style.standardIcon(qt.QStyle.SP_FileDialogToParent)
        backgroundIcon = style.standardIcon(standardPixmap)
        icon = qt.QIcon(self)

        sizes = baseIcon.availableSizes()
        sizes = sorted(sizes, key=lambda s: s.height())
        sizes = filter(lambda s: s.height() < 100, sizes)
        if len(sizes) > 0:
            baseSize = sizes[-1]
        else:
            baseIcon.availableSizes()[0]
        size = qt.QSize(baseSize.width(), baseSize.height() * 3 / 2)

        modes = [qt.QIcon.Normal, qt.QIcon.Disabled]
        for mode in modes:
            pixmap = qt.QPixmap(size)
            pixmap.fill(qt.Qt.transparent)
            painter = qt.QPainter(pixmap)
            painter.drawPixmap(0, 0, backgroundIcon.pixmap(baseSize, mode=mode))
            painter.drawPixmap(0, size.height() / 3, baseIcon.pixmap(baseSize, mode=mode))
            painter.end()
            icon.addPixmap(pixmap, mode=mode)

        return icon

    def getFileDialogToParentDir(self):
        if self.__iconFileDialogToParentDir is None:
            self.__iconFileDialogToParentDir = self._createIconToParent(qt.QStyle.SP_DirIcon)
        return self.__iconFileDialogToParentDir

    def getFileDialogToParentFile(self):
        if self.__iconFileDialogToParentFile is None:
            self.__iconFileDialogToParentFile = self._createIconToParent(qt.QStyle.SP_FileIcon)
        return self.__iconFileDialogToParentFile

    def icon(self, kind):
        if kind == self.FileDialogToParentDir:
            return self.getFileDialogToParentDir()
        elif kind == self.FileDialogToParentFile:
            return self.getFileDialogToParentFile()
        else:
            style = qt.QApplication.style()
            icon = style.standardIcon(kind)
            return icon


class _Slicing(qt.QWidget):

    slicingChanged = qt.Signal()

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.__shape = None
        self.__axis = []
        layout = qt.QVBoxLayout()
        self.setLayout(layout)

    def hasVisibleSliders(self):
        return self.__visibleSliders > 0

    def setShape(self, shape):
        if self.__shape is not None:
            # clean up
            for widget in self.__axis:
                self.layout().removeWidget(widget)
                widget.deleteLater()
            self.__axis = []

        self.__shape = shape
        self.__visibleSliders = 0

        if shape is not None:
            # create expected axes
            for index in range(len(shape) - 2):
                axis = qt.QSlider(self)
                axis.setMinimum(0)
                axis.setMaximum(shape[index] - 1)
                axis.setOrientation(qt.Qt.Horizontal)
                if shape[index] == 1:
                    axis.setVisible(False)
                else:
                    self.__visibleSliders += 1

                axis.valueChanged.connect(self.__axisValueChanged)
                self.layout().addWidget(axis)
                self.__axis.append(axis)

        self.slicingChanged.emit()

    def __axisValueChanged(self):
        self.slicingChanged.emit()

    def slicing(self):
        slicing = []
        for axes in self.__axis:
            slicing.append(axes.value())
        return tuple(slicing)

    def setSlicing(self, slicing):
        for i, value in enumerate(slicing):
            if i > len(self.__axis):
                break
            self.__axis[i].setValue(value)


class _ImagePreview(qt.QWidget):

    def __init__(self, parent=None):
        super(_ImagePreview, self).__init__(parent)

        self.__image = None
        self.__plot = PlotWidget(self)
        self.__plot.setAxesDisplayed(False)
        self.__plot.setKeepDataAspectRatio(True)
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__plot)
        self.setLayout(layout)

    def resizeEvent(self, event):
        self.__updateConstraints()
        return qt.QWidget.resizeEvent(self, event)

    def sizeHint(self):
        return qt.QSize(200, 200)

    def plot(self):
        return self.__plot

    def setImage(self, image):
        if image is None:
            self.clear()
            return

        self.__plot.addImage(legend="data", data=image)
        self.__plot.resetZoom()
        self.__image = image
        self.__updateConstraints()

    def __updateConstraints(self):
        """
        Update the constraints depending on the size of the widget
        """
        if self.__image is None:
            return
        size = self.size()
        if size.width() == 0 or size.height() == 0:
            return

        heightData, widthData = self.__image.shape

        widthContraint = heightData * size.width() / size.height()
        if widthContraint > widthData:
            heightContraint = heightData
        else:
            heightContraint = heightData * size.height() / size.width()
            widthContraint = widthData

        midWidth, midHeight = widthData * 0.5, heightData * 0.5
        heightContraint, widthContraint = heightContraint * 0.5, widthContraint * 0.5

        axis = self.__plot.getXAxis()
        axis.setLimitsConstraints(midWidth - widthContraint, midWidth + widthContraint)
        axis = self.__plot.getYAxis()
        axis.setLimitsConstraints(midHeight - heightContraint, midHeight + heightContraint)

    def __imageItem(self):
        image = self.__plot.getImage("data")
        return image

    def image(self):
        return self.__image

    def colormap(self):
        image = self.__imageItem()
        if image is not None:
            return image.getColormap()
        return self.__plot.getDefaultColormap()

    def setColormap(self, colormap):
        self.__plot.setDefaultColormap(colormap)

    def clear(self):
        self.__image = None
        image = self.__imageItem()
        if image is not None:
            self.__plot.removeImage(legend="data")


class _SideBar(qt.QListView):
    """Sidebar containing shortcuts for common directories"""

    def __init__(self, parent=None):
        super(_SideBar, self).__init__(parent)
        self.__iconProvider = qt.QFileIconProvider()
        self.setUniformItemSizes(True)
        model = qt.QStandardItemModel(self)
        self.setModel(model)
        self._initModel()
        self.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)

    def iconProvider(self):
        return self.__iconProvider

    def _initModel(self):

        # Get default shortcut
        # There is no other way
        d = qt.QFileDialog()
        urls = d.sidebarUrls()
        d = None

        model = qt.QStandardItemModel(self)
        self.setUrls(urls)
        return model

    def setUrls(self, urls):
        model = self.model()
        model.clear()

        names = {}
        names[qt.QDir.rootPath()] = "Computer"
        names[qt.QDir.homePath()] = "Home"

        style = qt.QApplication.style()
        iconProvider = self.iconProvider()
        for url in urls:
            path = url.toLocalFile()
            if path == "":
                name = "Computer"
                icon = style.standardIcon(qt.QStyle.SP_ComputerIcon)
            else:
                fileInfo = qt.QFileInfo(path)
                name = names.get(path, fileInfo.fileName())
                icon = iconProvider.icon(fileInfo)

            if icon.isNull():
                icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)

            item = qt.QStandardItem()
            item.setText(name)
            item.setIcon(icon)
            item.setData(url, role=qt.Qt.UserRole)
            model.appendRow(item)

    def urls(self):
        result = []
        model = self.model()
        for i in range(model.rowCount()):
            index = model.index(i, 0)
            url = model.data(index, qt.Qt.UserRole)
            result.append(url)
        return result

    def sizeHint(self):
        index = self.model().index(0, 0)
        return self.sizeHintForIndex(index) + qt.QSize(2 * self.frameWidth(), 2 * self.frameWidth())


class _Browser(qt.QStackedWidget):

    activated = qt.Signal(qt.QModelIndex)
    selected = qt.Signal(qt.QModelIndex)
    rootIndexChanged = qt.Signal(qt.QModelIndex)

    def __init__(self, parent, listView, detailView):
        qt.QStackedWidget.__init__(self, parent)
        self.__listView = listView
        self.__detailView = detailView
        self.insertWidget(0, self.__listView)
        self.insertWidget(1, self.__detailView)

        self.__listView.activated.connect(self.__emitActivated)
        self.__detailView.activated.connect(self.__emitActivated)

    def __emitActivated(self, index):
        self.activated.emit(index)

    def __emitSelected(self, selected, deselected):
        index = self.selectedIndex()
        if index is not None:
            self.selected.emit(index)

    def selectedIndex(self):
        if self.currentIndex() == 0:
            selectionModel = self.__listView.selectionModel()
        else:
            selectionModel = self.__detailView.selectionModel()

        if selectionModel is None:
            return None

        indexes = selectionModel.selectedIndexes()
        if len(indexes) == 1:
            index = indexes[0]
            return index
        return None

    def selectIndex(self, index):
        if self.currentIndex() == 0:
            selectionModel = self.__listView.selectionModel()
        else:
            selectionModel = self.__detailView.selectionModel()
        if selectionModel is None:
            return
        selectionModel.setCurrentIndex(index, qt.QItemSelectionModel.ClearAndSelect)

    def viewMode(self):
        """Returns the current view mode.

        :rtype: qt.QFileDialog.ViewMode
        """
        if self.currentIndex() == 0:
            return qt.QFileDialog.List
        elif self.currentIndex() == 1:
            return qt.QFileDialog.Detail
        else:
            assert(False)

    def setViewMode(self, mode):
        """Set the current view mode.

        :param qt.QFileDialog.ViewMode mode: The new view mode
        """
        if mode == qt.QFileDialog.Detail:
            self.showDetails()
        elif mode == qt.QFileDialog.List:
            self.showList()
        else:
            assert(False)

    def showList(self):
        self.__listView.show()
        self.__detailView.hide()
        self.setCurrentIndex(0)

    def showDetails(self):
        self.__listView.hide()
        self.__detailView.show()
        self.setCurrentIndex(1)
        self.__detailView.updateGeometry()

    def setRootIndex(self, index):
        """Sets the root item to the item at the given index.
        """
        rootIndex = self.__listView.rootIndex()
        if rootIndex == index:
            return
        newModel = index.model()
        if rootIndex is None or rootIndex.model() is not newModel:
            # update the model
            if self.__listView.selectionModel() is not None:
                self.__listView.selectionModel().selectionChanged.disconnect()
            if self.__detailView.selectionModel() is not None:
                self.__detailView.selectionModel().selectionChanged.disconnect()
            self.__listView.setModel(newModel)
            self.__detailView.setModel(newModel)
            self.__listView.selectionModel().selectionChanged.connect(self.__emitSelected)
            self.__detailView.selectionModel().selectionChanged.connect(self.__emitSelected)

        self.__listView.setRootIndex(index)
        self.__detailView.setRootIndex(index)
        self.rootIndexChanged.emit(index)

    def rootIndex(self):
        """Returns the model index of the model's root item. The root item is
        the parent item to the view's toplevel items. The root can be invalid.
        """
        return self.__listView.rootIndex()

    __serialVersion = 1
    """Store the current version of the serialized data"""

    @classmethod
    def qualifiedName(cls):
        cls.__module__ + "." + cls.__class__.__name__

    def restoreState(self, state):
        """Restores the dialogs's layout, history and current directory to the
        state specified.

        :param qt.QByeArray state: Stream containing the new state
        :rtype: bool
        """
        stream = qt.QDataStream(state, qt.QIODevice.ReadOnly)

        qualifiedName = stream.readString()
        if qualifiedName != self.qualifiedName():
            return False

        version = stream.readInt32()
        if version != self.__serialVersion:
            return False

        headerData = stream.readQVariant()
        self.__detailView.header().restoreState(headerData)

        viewMode = stream.readInt32()
        self.setViewMode(viewMode)
        return True

    def saveState(self):
        """Saves the state of the dialog's layout.

        :rtype: qt.QByteArray
        """
        data = qt.QByteArray()
        stream = qt.QDataStream(data, qt.QIODevice.WriteOnly)

        qualifiedName = self.__module__ + "." + self.__class__.__name__
        stream.writeString(qualifiedName)
        stream.writeInt32(self.__serialVersion)
        stream.writeQVariant(self.__detailView.header().saveState())
        stream.writeInt32(self.viewMode())

        return data


class _FabioData(object):

    def __init__(self, fabioFile):
        self.__fabioFile = fabioFile

    @property
    def dtype(self):
        # Let say it is a valid type
        return numpy.dtype("float")

    @property
    def shape(self):
        if self.__fabioFile.nframes == 0:
            return None
        return [self.__fabioFile.nframes, slice(None), slice(None)]

    def __getitem__(self, selector):
        if isinstance(selector, tuple) and len(selector) == 1:
            selector = selector[0]

        if isinstance(selector, six.integer_types):
            if 0 <= selector < self.__fabioFile.nframes:
                if self.__fabioFile.nframes == 1:
                    return self.__fabioFile.data
                else:
                    frame = self.__fabioFile.getframe(selector)
                    return frame.data
            else:
                raise ValueError("Invalid selector %s" % selector)
        else:
            raise TypeError("Unsupported selector type %s" % type(selector))


class _PathEdit(qt.QLineEdit):
    pass


class _CatchResizeEvent(qt.QObject):

    resized = qt.Signal(qt.QResizeEvent)

    def __init__(self, parent, target):
        super(_CatchResizeEvent, self).__init__(parent)
        self.__target = target
        self.__target_oldResizeEvent = self.__target.resizeEvent
        self.__target.resizeEvent = self.__resizeEvent

    def __resizeEvent(self, event):
        result = self.__target_oldResizeEvent(event)
        self.resized.emit(event)
        return result


class ImageFileDialog(qt.QDialog):
    """The ImageFileDialog class provides a dialog that allow users to select
    an image from a file.

    The ImageFileDialog class enables a user to traverse the file system in
    order to select one file. Then to traverse the file to select a frame or
    a slice of a dataset.

    The selected data is an image in 2 dimension.

    Using an ImageFileDialog can be done like that.

    .. code-block:: python

        dialog = ImageFileDialog()
        result = dialog.exec_()
        if result:
            print("Selection:")
            print(dialog.selectedFile())
            print(dialog.selectedImage())
            print(dialog.selectedPath())
        else:
            print("Nothing selected")
    """

    _defaultIconProvider = None
    """Lazy loaded default icon provider"""

    def __init__(self, parent=None):
        super(ImageFileDialog, self).__init__(parent)

        self.__selectedFile = None
        self.__selectedImage = None
        self.__currentHistory = []
        """Store history of URLs, last index one is the latest one"""
        self.__currentHistoryLocation = -1
        """Store the location in the history. Bigger is older"""

        self.__h5 = None
        self.__fabio = None
        self.__fileModel = qt.QFileSystemModel(self)
        # The common file dialog filter only on Mac OS X
        self.__fileModel.setNameFilterDisables(sys.platform == "darwin")
        self.__fileModel.setReadOnly(True)
        self.__fileModel.directoryLoaded.connect(self.__directoryLoaded)
        path = os.getcwd()
        self.__fileModel_setRootPath(path)

        self.__dataModel = Hdf5TreeModel(self)

        self.__createWidgets()
        self.__initLayout()
        self.__showAsListView()
        self.__clearData()
        self.__updatePath()

        # Update the file model filter
        self.__fileTypeCombo.setCurrentIndex(0)
        self.__filterSelected(0)

    # User interface

    def __createWidgets(self):
        self.__sidebar = self._createSideBar()
        self.__sidebar.selectionModel().selectionChanged.connect(self.__shortcutSelected)
        self.__sidebar.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        listView = qt.QListView(self)
        listView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        listView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        listView.setResizeMode(qt.QListView.Adjust)
        listView.setWrapping(True)
        listView.setEditTriggers(qt.QAbstractItemView.EditKeyPressed)
        listView.setContextMenuPolicy(qt.Qt.CustomContextMenu)

        treeView = qt.QTreeView(self)
        treeView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        treeView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        treeView.setRootIsDecorated(False)
        treeView.setItemsExpandable(False)
        treeView.setSortingEnabled(True)
        treeView.header().setSortIndicator(0, qt.Qt.AscendingOrder)
        treeView.header().setStretchLastSection(False)
        treeView.setTextElideMode(qt.Qt.ElideMiddle)
        treeView.setEditTriggers(qt.QAbstractItemView.EditKeyPressed)
        treeView.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        treeView.setDragDropMode(qt.QAbstractItemView.InternalMove)

        self.__browser = _Browser(self, listView, treeView)
        self.__browser.activated.connect(self.__browsedItemActivated)
        self.__browser.selected.connect(self.__browsedItemSelected)
        self.__browser.rootIndexChanged.connect(self.__rootIndexChanged)

        self.__imagePreview = _ImagePreview(self)
        self.__imagePreview.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        self.__fileTypeCombo = FileTypeComboBox(self)
        self.__fileTypeCombo.setDuplicatesEnabled(False)
        self.__fileTypeCombo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLength)
        self.__fileTypeCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.__fileTypeCombo.activated[int].connect(self.__filterSelected)

        self.__pathEdit = _PathEdit(self)
        self.__pathEdit.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.__pathEdit.textChanged.connect(self.__pathChanged)

        self.__buttons = qt.QDialogButtonBox(self)
        self.__buttons.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
        types = qt.QDialogButtonBox.Open | qt.QDialogButtonBox.Cancel
        self.__buttons.setStandardButtons(types)
        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)

        self.__browseToolBar = self._createBrowseToolBar()
        self.__backwardAction.setEnabled(False)
        self.__forwardAction.setEnabled(False)

        self.__previewToolBar = self._createPreviewToolbar(self.__imagePreview.plot())

        self.__slicing = _Slicing(self)
        self.__slicing.slicingChanged.connect(self.__slicingChanged)

        self.__dataInfo = qt.QLabel(self)

    def _createPreviewToolbar(self, plot):
        toolbar = qt.QToolBar(self)
        toolbar.setStyleSheet("QToolBar { border: 0px }")
        toolbar.addAction(actions.mode.ZoomModeAction(plot, self))
        toolbar.addAction(actions.mode.PanModeAction(plot, self))
        toolbar.addSeparator()
        toolbar.addAction(actions.control.ResetZoomAction(plot, self))
        toolbar.addSeparator()
        toolbar.addAction(actions.control.ColormapAction(plot, self))
        return toolbar

    def _createSideBar(self):
        return _SideBar(self)

    def iconProvider(self):
        iconProvider = self.__class__._defaultIconProvider
        if iconProvider is None:
            iconProvider = _IconProvider()
            self.__class__._defaultIconProvider = iconProvider
        return iconProvider

    def _createBrowseToolBar(self):
        toolbar = qt.QToolBar(self)
        iconProvider = self.iconProvider()

        backward = qt.QAction(toolbar)
        backward.setText("Back")
        backward.setIcon(iconProvider.icon(qt.QStyle.SP_ArrowBack))
        backward.triggered.connect(self.__navigateBackward)
        self.__backwardAction = backward

        forward = qt.QAction(toolbar)
        forward.setText("Forward")
        forward.setIcon(iconProvider.icon(qt.QStyle.SP_ArrowForward))
        forward.triggered.connect(self.__navigateForward)
        self.__forwardAction = forward

        parentDirectory = qt.QAction(toolbar)
        parentDirectory.setText("Parent directory")
        parentDirectory.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogToParent))
        parentDirectory.triggered.connect(self.__navigateToParent)

        fileDirectory = qt.QAction(toolbar)
        fileDirectory.setText("Root of the file")
        fileDirectory.setIcon(iconProvider.icon(iconProvider.FileDialogToParentFile))
        fileDirectory.triggered.connect(self.__navigateToParentFile)

        parentFileDirectory = qt.QAction(toolbar)
        parentFileDirectory.setText("Parent directory of the file")
        parentFileDirectory.setIcon(iconProvider.icon(iconProvider.FileDialogToParentDir))
        parentFileDirectory.triggered.connect(self.__navigateToParentDir)

        listView = qt.QAction(toolbar)
        listView.setText("List view")
        listView.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogListView))
        listView.triggered.connect(self.__showAsListView)
        listView.setCheckable(True)

        detailView = qt.QAction(toolbar)
        detailView.setText("List view")
        detailView.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogDetailedView))
        detailView.triggered.connect(self.__showAsDetailedView)
        detailView.setCheckable(True)

        self.__listViewAction = listView
        self.__detailViewAction = detailView

        toolbar.addAction(backward)
        toolbar.addAction(forward)
        toolbar.addSeparator()
        toolbar.addAction(parentDirectory)
        toolbar.addAction(fileDirectory)
        toolbar.addAction(parentFileDirectory)
        toolbar.addSeparator()
        toolbar.addAction(listView)
        toolbar.addAction(detailView)

        toolbar.setStyleSheet("QToolBar { border: 0px }")

        return toolbar

    def __initLayout(self):
        sideBarLayout = qt.QVBoxLayout()
        sideBarLayout.setContentsMargins(0, 0, 0, 0)
        dummyToolBar = qt.QWidget(self)
        dummyToolBar.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        dummyCombo = qt.QWidget(self)
        dummyCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        sideBarLayout.addWidget(dummyToolBar)
        sideBarLayout.addWidget(self.__sidebar)
        sideBarLayout.addWidget(dummyCombo)
        sideBarWidget = qt.QWidget(self)
        sideBarWidget.setLayout(sideBarLayout)

        dummyCombo.setFixedHeight(self.__fileTypeCombo.height())
        self.__resizeCombo = _CatchResizeEvent(self, self.__fileTypeCombo)
        self.__resizeCombo.resized.connect(lambda e: dummyCombo.setFixedHeight(e.size().height()))

        dummyToolBar.setFixedHeight(self.__browseToolBar.height())
        self.__resizeToolbar = _CatchResizeEvent(self, self.__browseToolBar)
        self.__resizeToolbar.resized.connect(lambda e: dummyToolBar.setFixedHeight(e.size().height()))

        datasetSelection = qt.QWidget(self)
        layoutLeft = qt.QVBoxLayout()
        layoutLeft.setContentsMargins(0, 0, 0, 0)
        layoutLeft.addWidget(self.__browseToolBar)
        layoutLeft.addWidget(self.__browser)
        layoutLeft.addWidget(self.__fileTypeCombo)
        datasetSelection.setLayout(layoutLeft)
        datasetSelection.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Expanding)

        imageFrame = qt.QFrame(self)
        imageFrame.setFrameShape(qt.QFrame.StyledPanel)
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.__imagePreview)
        layout.addWidget(self.__dataInfo)
        imageFrame.setLayout(layout)

        imageSelection = qt.QWidget(self)
        imageLayout = qt.QVBoxLayout()
        imageLayout.setContentsMargins(0, 0, 0, 0)
        imageLayout.addWidget(self.__previewToolBar)
        imageLayout.addWidget(imageFrame)
        imageLayout.addWidget(self.__slicing)
        imageSelection.setLayout(imageLayout)

        self.__splitter = qt.QSplitter(self)
        self.__splitter.setContentsMargins(0, 0, 0, 0)
        self.__splitter.addWidget(sideBarWidget)
        self.__splitter.addWidget(datasetSelection)
        self.__splitter.addWidget(imageSelection)
        self.__splitter.setStretchFactor(1, 10)

        bottomLayout = qt.QHBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.__pathEdit)
        bottomLayout.addWidget(self.__buttons)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__splitter)
        layout.addLayout(bottomLayout)

        self.setLayout(layout)
        self.updateGeometry()

    # Logic

    def __navigateBackward(self):
        """Navigate throug  the history one step backward."""
        if len(self.__currentHistory) > 0 and self.__currentHistoryLocation > 0:
            self.__currentHistoryLocation -= 1
            url = self.__currentHistory[self.__currentHistoryLocation]
            self.selectPath(url)

    def __navigateForward(self):
        """Navigate throug  the history one step forward."""
        if len(self.__currentHistory) > 0 and self.__currentHistoryLocation < len(self.__currentHistory) - 1:
            self.__currentHistoryLocation += 1
            url = self.__currentHistory[self.__currentHistoryLocation]
            self.selectPath(url)

    def __navigateToParent(self):
        index = self.__browser.rootIndex()
        if index.model() is self.__fileModel:
            # browse throw the file system
            index = index.parent()
            if index.isValid():
                self.__browser.setRootIndex(index)
        elif index.model() is self.__dataModel:
            index = index.parent()
            if index.isValid():
                # browse throw the hdf5
                self.__browser.setRootIndex(index)
            else:
                # go back to the file system
                self.__navigateToParentDir()
        else:
            assert(False)

    def __navigateToParentFile(self):
        index = self.__browser.rootIndex()
        if index.model() is self.__dataModel:
            index = utils.indexFromH5Object(self.__dataModel, self.__h5)
            self.__browser.setRootIndex(index)

    def __navigateToParentDir(self):
        index = self.__browser.rootIndex()
        if index.model() is self.__dataModel:
            path = os.path.dirname(self.__h5.file.filename)
            index = self.__fileModel.index(path)
            self.__browser.setRootIndex(index)
            self.__closeFile()

    def viewMode(self):
        """Returns the current view mode.

        :rtype: qt.QFileDialog.ViewMode
        """
        return self.__browser.viewMode()

    def setViewMode(self, mode):
        """Set the current view mode.

        :param qt.QFileDialog.ViewMode mode: The new view mode
        """
        if mode == qt.QFileDialog.Detail:
            self.__browser.showDetails()
            self.__listViewAction.setChecked(False)
            self.__detailViewAction.setChecked(True)
        elif mode == qt.QFileDialog.List:
            self.__browser.showList()
            self.__listViewAction.setChecked(True)
            self.__detailViewAction.setChecked(False)
        else:
            assert(False)

    def __showAsListView(self):
        self.setViewMode(qt.QFileDialog.List)

    def __showAsDetailedView(self):
        self.setViewMode(qt.QFileDialog.Detail)

    def __shortcutSelected(self):
        indexes = self.__sidebar.selectionModel().selectedIndexes()
        if len(indexes) == 1:
            index = indexes[0]
            url = self.__sidebar.model().data(index, role=qt.Qt.UserRole)
            path = url.toLocalFile()
            if path == "":
                path = qt.QDir.rootPath()
            self.__fileModel_setRootPath(path)

    def __browsedItemActivated(self, index):
        if index.model() is self.__fileModel:
            if self.__fileModel.isDir(index):
                self.__browser.setRootIndex(index)
            path = self.__fileModel.filePath(index)
            if os.path.isfile(path):
                self.__fileActivated(index)
        elif index.model() is self.__dataModel:
            obj = index.data(role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
            if silx.io.is_group(obj):
                self.__browser.setRootIndex(index)

    def __browsedItemSelected(self, index):
        self.__dataSelected(index)
        self.__updatePath()

    def __fileModel_setRootPath(self, path):
        """Set the root path of the fileModel with a filter on the
        directoryLoaded event.

        Without this filter an extra event is received (at least with PyQt4)
        when we use for the first time the sidebar.

        :param str path: Path to load
        """
        self.__directoryLoadedFilter = path
        self.__fileModel.setRootPath(path)

    def __directoryLoaded(self, path):
        if self.__directoryLoadedFilter != path:
            # Filter event which should not arrive in PyQt4
            # The first click on the sidebar sent 2 events
            return
        index = self.__fileModel.index(path)
        self.__browser.setRootIndex(index)

    def __closeFile(self):
        if self.__h5 is not None:
            self.__dataModel.removeH5pyObject(self.__h5)
            self.__h5.close()
            self.__h5 = None
        if self.__fabio is not None:
            if hasattr(self.__fabio, "close"):
                self.__fabio.close()
            self.__fabio = None

    def __openFabioFile(self, filename):
        self.__closeFile()
        try:
            self.__fabio = fabio.open(filename)
            self.__selectedFile = filename
        except Exception as e:
            _logger.error("Error while loading file %s: %s", filename, e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            return None
        else:
            return self.__fabio

    def __openSilxFile(self, filename):
        self.__closeFile()
        try:
            self.__h5 = silx.io.open(filename)
            self.__selectedFile = filename
        except IOError as e:
            _logger.error("Error while loading file %s: %s", filename, e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            return None
        else:
            self.__dataModel.insertH5pyObject(self.__h5)
            return self.__h5

    def __isSilxFile(self, filename):
        _, ext = os.path.splitext(filename)
        return ext in set([".h5", ".nx", ".npz", ".dat", ".spec"])

    def __openFile(self, filename):
        if self.__isSilxFile(filename):
            h5 = self.__openSilxFile(filename)
            if h5 is not None:
                return True
        else:
            fabio = self.__openFabioFile(filename)
            if fabio is not None:
                return True
        return False

    def __fileActivated(self, index):
        self.__selectedFile = None
        path = self.__fileModel.filePath(index)
        if os.path.isfile(path):
            loaded = self.__openFile(path)
            if loaded:
                if self.__h5 is not None:
                    index = utils.indexFromH5Object(self.__dataModel, self.__h5)
                    self.__browser.setRootIndex(index)
                elif self.__fabio is not None:
                    data = _FabioData(self.__fabio)
                    self.__setData(data)
            else:
                self.__clearData()

    def __dataSelected(self, index):
        selectedData = None
        if index is not None:
            if index.model() is self.__dataModel:
                obj = index.data(self.__dataModel.H5PY_OBJECT_ROLE)
                if silx.io.is_dataset(obj):
                    if obj.shape is not None and len(obj.shape) >= 2:
                        selectedData = obj
            elif index.model() is self.__fileModel:
                path = self.__fileModel.filePath(index)
                if os.path.isfile(path):
                    if not self.__isSilxFile(path):
                        # Then it's flat frame container
                        self.__openFabioFile(path)
                        if self.__fabio is not None:
                            selectedData = _FabioData(self.__fabio)
            else:
                assert(False)

        self.__setData(selectedData)

    def __filterSelected(self, index):
        filters = self.__fileTypeCombo.itemExtensions(index)
        self.__fileModel.setNameFilters(filters)

    def __setData(self, data):
        self.__data = data
        self.__selectedImage = None

        if data is None or data.shape is None:
            self.__clearData()
            self.__updatePath()
            return

        if data.dtype.kind not in set(["f", "u", "i", "b"]):
            self.__clearData()
            self.__updatePath()
            return

        dim = len(data.shape)
        if dim == 2:
            self.__selectedImage = data
            self.__imagePreview.setImage(data)
            self.__updateDataInfo()
            self.__updatePath()
            self.__slicing.hide()
            button = self.__buttons.button(qt.QDialogButtonBox.Open)
            button.setEnabled(True)
        elif dim > 2:
            self.__slicing.setShape(data.shape)
            self.__slicing.setVisible(self.__slicing.hasVisibleSliders())
            self.__slicing.slicingChanged.emit()
            button = self.__buttons.button(qt.QDialogButtonBox.Open)
            button.setEnabled(True)
        else:
            self.__clearData()
            self.__updatePath()

    def __clearData(self):
        """Clear the image part of the GUI"""
        self.__imagePreview.setImage(None)
        self.__slicing.hide()
        self.__selectedImage = None
        self.__data = None
        self.__updateDataInfo()
        button = self.__buttons.button(qt.QDialogButtonBox.Open)
        button.setEnabled(False)

    def __slicingChanged(self):
        slicing = self.__slicing.slicing()
        image = self.__data[slicing]
        self.__setImage(image)

    def __setImage(self, image):
        self.__imagePreview.setImage(image)
        self.__selectedImage = image
        self.__updateDataInfo()
        self.__updatePath()

    def __formatShape(self, shape):
        result = []
        for s in shape:
            if isinstance(s, slice):
                v = u"\u2026"
            else:
                v = str(s)
            result.append(v)
        return u" \u00D7 ".join(result)

    def __updateDataInfo(self):
        if self.__selectedImage is None:
            self.__dataInfo.setText("No data selected")
        else:
            destination = self.__formatShape(self.__selectedImage.shape)
            source = self.__formatShape(self.__data.shape)
            self.__dataInfo.setText(u"%s \u2192 %s" % (source, destination))

    def __createUriFromIndex(self, index, useSlicingWidget=True):
        if index.model() is self.__fileModel:
            filename = self.__fileModel.filePath(index)
            dataPath = None
        elif index.model() is self.__dataModel:
            obj = index.data(role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
            filename = obj.file.filename
            dataPath = obj.name
        else:
            filename = None
            dataPath = None

        if useSlicingWidget and self.__slicing.isVisible():
            slicing = self.__slicing.slicing()
        else:
            slicing = None

        uri = _ImageUri(filename=filename, dataPath=dataPath, slice=slicing)
        return uri

    def __updatePath(self):
        index = self.__browser.selectedIndex()
        if index is None:
            index = self.__browser.rootIndex()
        uri = self.__createUriFromIndex(index)
        if uri.path() != self.__pathEdit.text():
            old = self.__pathEdit.blockSignals(True)
            self.__pathEdit.setText(uri.path())
            self.__pathEdit.blockSignals(old)

    def __rootIndexChanged(self, index):
        uri = self.__createUriFromIndex(index, useSlicingWidget=False)

        currentUri = None
        if 0 <= self.__currentHistoryLocation < len(self.__currentHistory):
            currentUri = self.__currentHistory[self.__currentHistoryLocation]

        if currentUri is None or currentUri != uri.path():
            # clean up the forward history
            self.__currentHistory = self.__currentHistory[0:self.__currentHistoryLocation + 1]
            self.__currentHistory.append(uri.path())
            self.__currentHistoryLocation += 1

        self.__updateActionHistory()

    def __updateActionHistory(self):
        self.__forwardAction.setEnabled(len(self.__currentHistory) - 1 > self.__currentHistoryLocation)
        self.__backwardAction.setEnabled(self.__currentHistoryLocation > 0)

    def __pathChanged(self):
        uri = _ImageUri(path=self.__pathEdit.text())
        if uri.isValid():
            if os.path.exists(uri.filename()):
                self.__fileModel_setRootPath(uri.filename())
                index = self.__fileModel.index(uri.filename())
                rootIndex = None
                if os.path.isfile(uri.filename()):
                    loaded = self.__openFile(uri.filename())
                    if loaded:
                        if self.__h5 is not None:
                            rootIndex = utils.indexFromH5Object(self.__dataModel, self.__h5)
                            self.__browser.setRootIndex(index)
                        elif self.__fabio is not None:
                            rootIndex = index
                    else:
                        self.__clearData()

                if rootIndex is not None:
                    if rootIndex.model() == self.__dataModel:
                        if uri.dataPath() is not None:
                            dataPath = uri.dataPath()
                            if dataPath in self.__h5:
                                obj = self.__h5[dataPath]
                            else:
                                path = utils.findClosestSubPath(self.__h5, dataPath)
                                if path is None:
                                    path = "/"
                                obj = self.__h5[path]

                            if silx.io.is_file(obj):
                                self.__browser.setRootIndex(rootIndex)
                                self.__clearData()
                            elif silx.io.is_group(obj):
                                index = utils.indexFromH5Object(rootIndex.model(), obj)
                                self.__browser.setRootIndex(index)
                                self.__clearData()
                            else:
                                index = utils.indexFromH5Object(rootIndex.model(), obj)
                                self.__browser.setRootIndex(index.parent())
                                self.__browser.selectIndex(index)
                        else:
                            self.__browser.setRootIndex(rootIndex)
                            self.__clearData()
                    elif rootIndex.model() == self.__fileModel:
                        # that's a fabio file
                        self.__browser.setRootIndex(rootIndex.parent())
                        self.__browser.selectIndex(rootIndex)
                        # data = _FabioData(self.__fabio)
                        # self.__setData(data)
                else:
                    self.__browser.setRootIndex(index)
                    self.__clearData()

                self.__slicing.setVisible(uri.slice() is not None)
                if uri.slice() is not None:
                    self.__slicing.setSlicing(uri.slice())

    # Selected file

    def setDirectory(self, path):
        """Sets the image dialog's current directory."""
        self.__fileModel.reset()
        self.__fileModel_setRootPath(path)

    def selectedFile(self):
        """Returns the file path containing the selected data.

        :rtype: str
        """
        return self.__selectedFile

    def selectFile(self, path):
        """Sets the image dialog's current file."""
        self.__pathEdit.setText(path)
        self.__pathChanged()

    # Selected image

    def selectPath(self, path):
        """Sets the image dialog's current image path.

        :param str path: Path identifying an image or a path
        """
        self.__pathEdit.setText(path)
        self.__pathChanged()

    def selectedPath(self):
        """Returns the URI from the file path to the image.

        If the dialog is not validated, the path can be an intermediat
        selected path, or an invalid path.

        :rtype: str
        """
        return self.__pathEdit.text()

    def selectedDirectory(self):
        """Returns the path from the current browsed directory.

        :rtype: str
        """
        index = self.__browser.rootIndex()
        if index.model() is self.__fileModel:
            path = self.__fileModel.filePath(index)
            if os.path.isfile(path):
                path = os.path.dirname(path)
            return path
        elif index.model() is self.__dataModel:
            path = os.path.dirname(self.__h5.file.filename)
            return path

    def selectedImage(self):
        """Returns the numpy array selected.

        :rtype: numpy.ndarray
        """
        return self.__selectedImage

    # Filters

    def selectedNameFilter(self):
        """Returns the filter that the user selected in the file dialog."""
        return self.__fileTypeCombo.currentText()

    # History

    def history(self):
        """Returns the browsing history of the filedialog as a list of paths.

        :rtype: List<str>
        """
        if len(self.__currentHistory) <= 1:
            return []
        history = self.__currentHistory[0:self.__currentHistoryLocation]
        return list(history)

    def setHistory(self, history):
        self.__currentHistory = []
        self.__currentHistory.extend(history)
        self.__currentHistoryLocation = len(self.__currentHistory) - 1
        self.__updateActionHistory()

    # Colormap

    def colormap(self):
        return self.__imagePreview.colormap()

    def setColormap(self, colormap):
        self.__imagePreview.setColormap(colormap)

    # State

    __serialVersion = 1
    """Store the current version of the serialized data"""

    @classmethod
    def qualifiedName(cls):
        cls.__module__ + "." + cls.__class__.__name__

    def restoreState(self, state):
        """Restores the dialogs's layout, history and current directory to the
        state specified.

        :param qt.QByeArray state: Stream containing the new state
        :rtype: bool
        """
        stream = qt.QDataStream(state, qt.QIODevice.ReadOnly)

        qualifiedName = stream.readString()
        if qualifiedName != self.qualifiedName():
            return False

        version = stream.readInt32()
        if version != self.__serialVersion:
            return False

        splitterData = stream.readQVariant()
        sidebarUrls = stream.readQVariantList()
        history = stream.readQVariantList()
        selectedDirectory = stream.readString()
        browserData = stream.readQVariant()
        viewMode = stream.readInt32()
        colormap = utils.readColormap(stream)

        self.__splitter.restoreState(splitterData)
        self.__sidebar.setUrls(list(sidebarUrls))
        self.setHistory(list(history))
        self.__splitter.restoreState(browserData)
        self.setDirectory(selectedDirectory)
        self.setViewMode(viewMode)
        self.setColormap(colormap)

        return True

    def saveState(self):
        """Saves the state of the dialog's layout, history and current
        directory.

        :rtype: qt.QByteArray
        """
        data = qt.QByteArray()
        stream = qt.QDataStream(data, qt.QIODevice.WriteOnly)

        stream.writeString(self.qualifiedName())
        stream.writeInt32(self.__serialVersion)
        stream.writeQVariant(self.__splitter.saveState())
        stream.writeQVariantList(self.__sidebar.urls())
        stream.writeQVariantList(self.history())
        stream.writeString(self.selectedDirectory())
        stream.writeQVariant(self.__browser.saveState())
        stream.writeInt32(self.viewMode())
        utils.writeColormap(stream, self.colormap())

        return data
