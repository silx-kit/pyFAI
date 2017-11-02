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
This module contains an :class:`SafeFileSystemModel`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "31/10/2017"

import sys
import os.path
import logging
import weakref
from silx.gui import qt
from pyFAI.third_party import six
from .SafeFileIconProvider import SafeFileIconProvider

_logger = logging.getLogger(__name__)


class _Item(object):

    def __init__(self, fileInfo):
        self.__fileInfo = fileInfo
        self.__parent = None
        self.__children = None
        self.__absolutePath = None

    def isDriver(self):
        return self.parent().parent() is None

    def isRoot(self):
        return self.parent() is None

    def absoluteFilePath(self):
        """
        Returns an absolute path including the file name.

        This function uses in most cases the default
        `qt.QFileInfo.absoluteFilePath`. But it is known to freeze the file
        system with network drivers.

        This function uses `qt.QFileInfo.filePath` in case of root drivers, to
        avoid this kind of issues. In case of driver, the result is the same,
        while the file path is already absolute.

        :rtype: str
        """
        if self.__absolutePath is None:
            if self.isRoot():
                path = ""
            elif self.isDriver():
                path = self.__fileInfo.filePath()
            else:
                path = os.path.join(self.parent().absoluteFilePath(), self.__fileInfo.fileName())
            self.__absolutePath = path
        return self.__absolutePath

    def child(self):
        self.populate()
        return self.__children

    def childAt(self, position):
        self.populate()
        return self.__children[position]

    def childCount(self):
        self.populate()
        return len(self.__children)

    def indexOf(self, item):
        self.populate()
        return self.__children.index(item)

    def parent(self):
        parent = self.__parent
        if parent is None:
            return None
        return parent()

    def filePath(self):
        return self.__fileInfo.filePath()

    def fileName(self):
        if self.isDriver():
            name = self.absoluteFilePath()
            if name[-1] == "/":
                name = name[:-1]
            return name
        return os.path.basename(self.absoluteFilePath())

    def fileInfo(self):
        return self.__fileInfo

    def _setParent(self, parent):
        self.__parent = weakref.ref(parent)

    def findChlidrenByPath(self, path):
        if path == "":
            return self
        path = path.replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        names = path.split("/")
        count = len(names)
        cursor = self
        for name in names:
            for item in cursor.child():
                if item.fileName() == name:
                    cursor = item
                    count -= 1
                    break
            else:
                return None
            if count == 0:
                break
        else:
            return None
        return cursor

    def populate(self):
        if self.__children is not None:
            return
        self.__children = []
        if self.isRoot():
            items = qt.QDir.drives()
        else:
            directory = qt.QDir(self.absoluteFilePath())
            items = directory.entryInfoList()
        for fileInfo in items:
            i = _Item(fileInfo)
            self.__children.append(i)
            i._setParent(self)


class _RawFileSystemModel(qt.QAbstractItemModel):
    """
    This class implement a file system model and try to avoid freeze. On Qt4,
    :class:`qt.QFileSystemModel` is known to freeze the file system when
    network drives are available.
    
    To avoid this behaviour, this class does not use
    `qt.QFileInfo.absoluteFilePath` nor `qt.QFileInfo.canonicalPath` to reach
    information on drives.

    This model do not take care of sorting and filtering. This features are
    managed by another model, by composition.

    And because it is the end of life of Qt4, we do not implement asynchronous
    loading of files as it is done by :class:`qt.QFileSystemModel`, nor some
    useful features.
    """

    __directoryLoadedSync = qt.Signal(str)
    """This signal is connected asynchronously to a slot. It allows to
    emit directoryLoaded as an asynchronous signal."""
    
    directoryLoaded = qt.Signal(str)
    """This signal is emitted when the gatherer thread has finished to load the
    path."""

    rootPathChanged = qt.Signal(str)
    """This signal is emitted whenever the root path has been changed to a
    newPath."""

    NAME_COLUMN = 0
    SIZE_COLUMN = 1
    TYPE_COLUMN = 2
    LAST_MODIFIED_COLUMN = 3

    def __init__(self, parent=None):
        qt.QAbstractItemModel.__init__(self, parent)
        self.__computer = _Item(qt.QFileInfo())
        self.__header = "Name", "Size", "Type", "Last modification"
        self.__currentPath = ""
        self.__iconProvider = SafeFileIconProvider()
        self.__directoryLoadedSync.connect(self.__emitDirectoryLoaded, qt.Qt.QueuedConnection)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if orientation == qt.Qt.Horizontal:
            if role == qt.Qt.DisplayRole:
                return self.__header[section]
            if role == qt.Qt.TextAlignmentRole:
                return qt.Qt.AlignRight if section == 1 else qt.Qt.AlignLeft
        return None

    def flags(self, index):
        if not index.isValid():
            return 0
        return qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable

    def columnCount(self, parent=qt.QModelIndex()):
        return len(self.__header)

    def rowCount(self, parent=qt.QModelIndex()):
        item = self.__getItem(parent)
        return item.childCount()

    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid():
            return None

        column = index.column()
        if role in [qt.Qt.DisplayRole, qt.Qt.EditRole]:
            if column == self.NAME_COLUMN:
                return self.__displayName(index)
            elif column == self.SIZE_COLUMN:
                return self.size(index)
            elif column == self.TYPE_COLUMN:
                return self.type(index)
            elif column == self.LAST_MODIFIED_COLUMN:
                return self.lastModified(index)
            else:
                _logger.warning("data: invalid display value column %d", index.column())
        elif role == qt.QFileSystemModel.FilePathRole:
            return self.filePath(index)
        elif role == qt.QFileSystemModel.FileNameRole:
            return self.fileName(index)
        elif role == qt.Qt.DecorationRole:
            if column == self.NAME_COLUMN:
                icon = self.fileIcon(index)
                if icon is None or icon.isNull():
                    if self.isDir(index):
                        self.__iconProvider.icon(qt.QFileIconProvider.Folder)
                    else:
                        self.__iconProvider.icon(qt.QFileIconProvider.File)
                return icon
        elif role == qt.Qt.TextAlignmentRole:
            if column == self.SIZE_COLUMN:
                return qt.Qt.AlignRight
        elif role == qt.QFileSystemModel.FilePermissions:
            return self.permissions(index)

        return None

    def index(self, *args, **kwargs):
        path_api = False
        path_api |= len(args) >= 1 and isinstance(args[0], six.string_types)
        path_api |= "path" in kwargs

        if path_api:
            return self.__indexFromPath(*args, **kwargs)
        else:
            return self.__index(*args, **kwargs)

    def __index(self, row, column, parent=qt.QModelIndex()):
        if parent.isValid() and parent.column() != 0:
            return None

        parentItem = self.__getItem(parent)
        item = parentItem.childAt(row)
        return self.createIndex(row, column, item)

    def __indexFromPath(self, path, column=0):
        """
        Uses the index(str) C++ API

        :rtype: qt.QModelIndex
        """
        if path == "":
            return qt.QModelIndex()

        item = self.__computer.findChlidrenByPath(path)
        if item is None:
            return qt.QModelIndex()

        return self.createIndex(item.parent().indexOf(item), column, item)

    def parent(self, index):
        if not index.isValid():
            return qt.QModelIndex()

        item = self.__getItem(index)
        if index is None:
            return qt.QModelIndex()

        parent = item.parent()
        if parent is None or parent is self.__computer:
            return qt.QModelIndex()

        return self.createIndex(parent.parent().indexOf(parent), 0, parent)

    def __emitDirectoryLoaded(self, path):
        self.directoryLoaded.emit(path)

    def __emitRootPathChanged(self, path):
        self.rootPathChanged.emit(path)

    def __getItem(self, index):
        if not index.isValid():
            return self.__computer
        item = index.internalPointer()
        return item

    def fileIcon(self, index):
        item = self.__getItem(index)
        if self.__iconProvider is not None:
            fileInfo = item.fileInfo()
            result = self.__iconProvider.icon(fileInfo)
        else:
            style = qt.QApplication.instance().style()
            if item.isRoot():
                result = style.standardIcon(qt.QStyle.SP_ComputerIcon)
            elif item.isDriver():
                result = style.standardIcon(qt.QStyle.SP_DriveHDIcon)
            elif item.isDir():
                result = style.standardIcon(qt.QStyle.SP_DirIcon)
            else:
                result = style.standardIcon(qt.QStyle.SP_FileIcon)
        return result

    def fileInfo(self, index):
        item = self.__getItem(index)
        result = item.fileInfo()
        return result

    def __fileIcon(self, index):
        item = self.__getItem(index)
        result = item.fileName()
        return result

    def __displayName(self, index):
        item = self.__getItem(index)
        result = item.fileName()
        return result

    def fileName(self, index):
        item = self.__getItem(index)
        result = item.fileName()
        return result

    def filePath(self, index):
        item = self.__getItem(index)
        result = item.fileInfo().filePath()
        return result

    def isDir(self, index):
        item = self.__getItem(index)
        result = item.fileInfo().isDir()
        return result

    def lastModified(self, index):
        item = self.__getItem(index)
        result = item.fileInfo().lastModified()
        return result

    def permissions(self, index):
        item = self.__getItem(index)
        result = item.fileInfo().permissions()
        return result

    def size(self, index):
        item = self.__getItem(index)
        result = item.fileInfo().size()
        return result

    def type(self, index):
        item = self.__getItem(index)
        if self.__iconProvider is not None:
            fileInfo = item.fileInfo()
            result = self.__iconProvider.type(fileInfo)
        else:
            if item.isRoot():
                result = "Computer"
            elif item.isDriver():
                result = "Driver"
            elif item.isDir():
                result = "Directory"
            else:
                fileInfo = item.fileInfo()
                result = fileInfo.suffix()
        return result

    # File manipulation

    # bool        remove(const QModelIndex & index) const
    # bool        rmdir(const QModelIndex & index) const
    # QModelIndex mkdir(const QModelIndex & parent, const QString & name)

    # Configuration

    def rootDirectory(self):
        return qt.QDir(self.rootPath())

    def rootPath(self):
        return self.__currentPath

    def setRootPath(self, path):
        if self.__currentPath == path:
            return
        self.__currentPath = path
        item = self.__computer.findChlidrenByPath(path)
        self.__emitRootPathChanged(path)
        if item is None or item.parent() is None:
            return qt.QModelIndex()
        index = self.createIndex(item.parent().indexOf(item), 0, item)
        self.__directoryLoadedSync.emit(path)
        return index

    def iconProvider(self):
        # FIXME: invalidate the model
        return self.__iconProvider

    def setIconProvider(self, provider):
        # FIXME: invalidate the model
        self.__iconProvider = provider

    # bool               resolveSymlinks() const
    # void               setResolveSymlinks(bool enable)

    def setNameFilterDisables(self, enable):
        return None

    def nameFilterDisables(self):
        return None

    def myComputer(self, role=qt.Qt.DisplayRole):
        return None

    def setNameFilters(self, filters):
        return

    def nameFilters(self):
        return None

    def filter(self):
        return self.__filters

    def setFilter(self, filters):
        return

    def setReadOnly(self, enable):
        assert(enable is True)

    def isReadOnly(self):
        return False


class SafeFileSystemModel(qt.QSortFilterProxyModel):
    """
    This class implement a file system model and try to avoid freeze. On Qt4,
    :class:`qt.QFileSystemModel` is known to freeze the file system when
    network drives are available.

    To avoid this behaviour, this class does not use
    `qt.QFileInfo.absoluteFilePath` nor `qt.QFileInfo.canonicalPath` to reach
    information on drives.

    And because it is the end of life of Qt4, we do not implement asynchronous
    loading of files as it is done by :class:`qt.QFileSystemModel`, nor some
    useful features.
    """

    def __init__(self, parent=None):
        qt.QSortFilterProxyModel.__init__(self, parent=parent)
        self.__nameFilterDisables = sys.platform == "darwin"
        self.__nameFilters = []
        self.__filters = qt.QDir.AllEntries | qt.QDir.NoDotAndDotDot | qt.QDir.AllDirs
        sourceModel = _RawFileSystemModel(self)
        self.setSourceModel(sourceModel)

    @property
    def directoryLoaded(self):
        return self.sourceModel().directoryLoaded

    @property
    def rootPathChanged(self):
        return self.sourceModel().rootPathChanged

    def index(self, *args, **kwargs):
        path_api = False
        path_api |= len(args) >= 1 and isinstance(args[0], six.string_types)
        path_api |= "path" in kwargs

        if path_api:
            return self.__indexFromPath(*args, **kwargs)
        else:
            return self.__index(*args, **kwargs)

    def __index(self, row, column, parent=qt.QModelIndex()):
        return qt.QSortFilterProxyModel.index(self, row, column, parent)

    def __indexFromPath(self, path, column=0):
        """
        Uses the index(str) C++ API

        :rtype: qt.QModelIndex
        """
        if path == "":
            return qt.QModelIndex()

        index = self.sourceModel().index(path, column)
        index = self.mapFromSource(index)
        return index

    def lessThan(self, leftSourceIndex, rightSourceIndex):
        sourceModel = self.sourceModel()
        sortColumn = self.sortColumn()
        if sortColumn == _RawFileSystemModel.NAME_COLUMN:
            left = sourceModel.fileInfo(leftSourceIndex)
            right = sourceModel.fileInfo(rightSourceIndex)
            if sys.platform != "darwin":
                # Sort directories before files
                leftIsDir = left.isDir()
                rightIsDir = right.isDir()
                if leftIsDir ^ rightIsDir:
                    return leftIsDir
            return left.fileName().lower() < right.fileName().lower()
        elif sortColumn == _RawFileSystemModel.SIZE_COLUMN:
            left = sourceModel.fileInfo(leftSourceIndex)
            right = sourceModel.fileInfo(rightSourceIndex)
            return left.size() < right.size()
        elif sortColumn == _RawFileSystemModel.TYPE_COLUMN:
            left = sourceModel.type(leftSourceIndex)
            right = sourceModel.type(rightSourceIndex)
            return left < right
        elif sortColumn == _RawFileSystemModel.LAST_MODIFIED_COLUMN:
            left = sourceModel.fileInfo(leftSourceIndex)
            right = sourceModel.fileInfo(rightSourceIndex)
            return left.lastModified() < right.lastModified()
        else:
            _logger.warning("Unsupported sorted column %d", sortColumn)

        return False

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if not sourceParent.isValid():
            return True

        sourceModel = self.sourceModel()
        index = sourceModel.index(sourceRow, 0, sourceParent)
        if not index.isValid():
            return True
        fileInfo = sourceModel.fileInfo(index)
        if fileInfo is None:
            return False

        filters = self.__filters
        filterPermissions = ((filters & qt.QDir.PermissionMask)
                            and (filters & qt.QDir.PermissionMask) != qt.QDir.PermissionMask)
        hideDirs = not (filters & (qt.QDir.Dirs | qt.QDir.AllDirs))
        hideFiles = not (filters & qt.QDir.Files)
        hideReadable = not (not filterPermissions or (filters & qt.QDir.Readable))
        hideWritable = not (not filterPermissions or (filters & qt.QDir.Writable))
        hideExecutable = not (not filterPermissions or (filters & qt.QDir.Executable))
        hideHidden = not (filters & qt.QDir.Hidden)
        hideSystem = not (filters & qt.QDir.System)
        hideSymlinks = (filters & qt.QDir.NoSymLinks)
        hideDot = (filters & qt.QDir.NoDot) or (filters & qt.QDir.NoDotAndDotDot)
        hideDotDot = (filters & qt.QDir.NoDotDot) or (filters & qt.QDir.NoDotAndDotDot)

        fileName = fileInfo.fileName()
        isDot = fileName == "."
        isDotDot = fileName == ".."
        isSystem = not fileInfo.isDir() and not fileInfo.isFile()

        if ((hideHidden and not (isDot or isDotDot) and fileInfo.isHidden())
            or (hideSystem and isSystem)
            or (hideDirs and fileInfo.isDir())
            or (hideFiles and fileInfo.isFile())
            or (hideSymlinks and fileInfo.isSymLink())
            or (hideReadable and fileInfo.isReadable())
            or (hideWritable and fileInfo.isWritable())
            or (hideExecutable and fileInfo.isExecutable())
            or (hideDot and isDot)
            or (hideDotDot and isDotDot)):
            return False

        if self.__nameFilterDisables:
            return True

        if fileInfo.isDir() and (filters & qt.QDir.AllDirs):
            return True

        return self.__nameFiltersAccepted(fileInfo)

    def __nameFiltersAccepted(self, fileInfo):
        if len(self.__nameFilters) == 0:
            return True

        fileName = fileInfo.fileName()
        for reg in self.__nameFilters:
            if reg.exactMatch(fileName):
                return True
        return False

    def setNameFilterDisables(self, enable):
        self.__nameFilterDisables = enable
        self.invalidateFilter()

    def nameFilterDisables(self):
        return self.__nameFilterDisables

    def myComputer(self, role=qt.Qt.DisplayRole):
        return self.sourceModel().myComputer(role)

    def setNameFilters(self, filters):
        self.__nameFilters = []
        isCaseSensitive = self.__filters & qt.QDir.CaseSensitive
        caseSensitive = qt.Qt.CaseSensitive if isCaseSensitive  else qt.Qt.CaseInsensitive
        for filter in filters:
            reg = qt.QRegExp(filter, caseSensitive, qt.QRegExp.Wildcard)
            self.__nameFilters.append(reg)
        self.invalidateFilter()

    def nameFilters(self):
        return [f.pattern() for f in self.__nameFilters]

    def filter(self):
        return self.__filters

    def setFilter(self, filters):
        self.__filters = filters
        # In case of change of case sensitivity
        self.setNameFilters(self.nameFilters())
        self.invalidateFilter()

    def setReadOnly(self, enable):
        assert(enable is True)

    def isReadOnly(self):
        return False

    def rootPath(self):
        return self.sourceModel().rootPath()

    def setRootPath(self, path):
        index = self.sourceModel().setRootPath(path)
        index = self.mapFromSource(index)
        return index

    def flags(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        filters = sourceModel.flags(index)

        if self.__nameFilterDisables:
            fileInfo = sourceModel.fileInfo(index)
            if not self.__nameFiltersAccepted(fileInfo):
                filters &= ~qt.Qt.ItemIsEnabled

        return filters

    def fileIcon(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.fileIcon(index)

    def fileInfo(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.fileInfo(index)

    def fileName(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.fileName(index)

    def filePath(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.filePath(index)

    def isDir(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.isDir(index)

    def lastModified(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.lastModified(index)

    def permissions(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.permissions(index)

    def size(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.size(index)

    def type(self, index):
        sourceModel = self.sourceModel()
        index = self.mapToSource(index)
        return sourceModel.type(index)
