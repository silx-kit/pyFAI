# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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


from __future__ import absolute_import, print_function, division

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/11/2015"
__status__ = "development"
__docformat__ = 'restructuredtext'
__doc__ = """

Module with GUI for diffraction mapping experiments 


"""
import os
import time
import json
import threading
import numpy
from .gui_utils import QtGui, QtCore, uic, pyplot, update_fig
from .utils import float_, int_, str_, get_ui_file
from .units import to_unit
from .integrate_widget import AIWidget
from .diffmap import DiffMap
from .tree import ListDataSet, DataSet
import logging
logger = logging.getLogger("diffmap_widget")


class IntegrateWidget(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self)
        self.widget = AIWidget()
        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(self.widget)
        self.widget.okButton.clicked.disconnect()
        self.widget.cancelButton.clicked.disconnect()
        self.widget.okButton.clicked.connect(self.accept)
        self.widget.cancelButton.clicked.connect(self.reject)

    def get_config(self):
        res = self.widget.dump()
        res["method"] = self.widget.get_method()
        return res


class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, win, root_item):
        super(TreeModel, self).__init__(win)
        self._root_item = root_item
        self._win = win
        self._current_branch = None

    def set_root(self, new_root):
#         self.beginResetModel()
        self._root_item = new_root
#         self.endResetModel()

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        pitem = parent.internalPointer()
        if not parent.isValid():
            pitem = self._root_item
        return len(pitem.children)

    def columnCount(self, parent):
        return 2

    def flags(self, midx):
#        if midx.column()==1:
        return QtCore.Qt.ItemIsEnabled

    def index(self, row, column, parent):
        pitem = parent.internalPointer()
        if not parent.isValid():
            pitem = self._root_item
        try:
            item = pitem.children[row]
        except IndexError:
            return QtCore.QModelIndex()
        return self.createIndex(row, column, item)

    def data(self, midx, role):
        """
        What to display depending on model_index and role
        """
        leaf = midx.internalPointer()
        if midx.column() == 0 and role == QtCore.Qt.DisplayRole:
            return leaf.label

#         if midx.column() == 1 and role == QtCore.Qt.DisplayRole:
#             if leaf.order == 4:
#                 if leaf.extra is None:
#                     data = Photo(leaf.name).readExif()
#                     leaf.extra = data["title"]
#                 return leaf.extra

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return ["Path", "shape"][section]

    def parent(self, midx):
        pitem = midx.internalPointer().parent
        if pitem is self._root_item:
#             return self.createIndex(0, 0, self._root_item)
            return QtCore.QModelIndex()
        row_idx = pitem.parent.children.index(pitem)
        return self.createIndex(row_idx, 0, pitem)



class ListModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, actual_data=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._ref = actual_data
#     def dropMimeData(self, *args, **kwargs):
#         print("ListModel.dropMimeData %s %s" % (args, kwargs))
    def rowCount(self, parent=None):
        return len(self._ref)
    def columnCount(self, parent=None):
        return 3
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if index.row() >= len(self._ref):
                return
            data = self._ref[index.row()]
            if index.column() == 0:
                return data.path
            elif index.column() == 1:
                return data.h5
            if index.column() == 2:
                return data.nframes

    def setData(self, *args, **kwargs):
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                if section == 0:
                    return "File path"
                elif section == 1:
                    return "h5"
                elif section == 2:
                    return "#"
            elif role == QtCore.Qt.WhatsThisRole:
                if section == 0:
                    return "Path of the file in the computer"
                elif section == 1:
                    return "Internal path in the HDF5 tree"
                elif section == 2:
                    return "Number of frames in the dataset"
            elif role == QtCore.Qt.SizeHintRole:
                if section == 0:
                    return QtCore.QSize(200, 20)
                elif section == 1:
                    return QtCore.QSize(20, 20)
                elif section == 2:
                    return QtCore.QSize(20, 20)



class DiffMapWidget(QtGui.QWidget):
    progressbarChanged = QtCore.pyqtSignal(int, int)
#     progressbarAborted = QtCore.pyqtSignal()
    uif = "diffmap.ui"
    json_file = ".diffmap.json"
    URL = "http://pyfai.readthedocs.org/en/latest/man/scripts.html"
    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.integration_config = {}
        self.list_dataset = ListDataSet()  # Contains all datasets to be treated.

        try:
            uic.loadUi(get_ui_file(self.uif), self)
        except AttributeError as error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")
        self.aborted = False
        self.progressBar.setValue(0)
        self.list_model = TreeModel(self, self.list_dataset.as_tree())
        self.listFiles.setModel(self.list_model)
        self.listFiles.hideColumn(1)
        self.listFiles.hideColumn(2)
        self.create_connections()
        self.set_validator()
        self.update_number_of_frames()
        self.update_number_of_points()
        self.processing_thread = None
        self.processing_sem = threading.Semaphore()
        self.update_sem = threading.Semaphore()

        # Online visualization
        self.fig = None
        self.axplt = None
        self.aximg = None
        self.img = None
        self.plot = None
        self.radial_data = None
        self.data = None
        self.last_idx = -1


    def set_validator(self):
        validator = QtGui.QIntValidator(0, 999999, self)
        self.fastMotorPts.setValidator(validator)
        self.slowMotorPts.setValidator(validator)
        self.offset.setValidator(validator)

    def create_connections(self):
        """Signal-slot connection
        """
        self.configureDiffraction.clicked.connect(self.configure_diffraction)
        self.outputFileSelector.clicked.connect(self.configure_output)
        self.runButton.clicked.connect(self.start_processing)
        self.addFiles.clicked.connect(self.input_filer)
        self.sortButton.clicked.connect(self.sort_input)
        self.saveButton.clicked.connect(self.save_config)
        self.abortButton.clicked.connect(self.do_abort)
        self.fastMotorPts.editingFinished.connect(self.update_number_of_points)
        self.slowMotorPts.editingFinished.connect(self.update_number_of_points)
        self.offset.editingFinished.connect(self.update_number_of_points)
        self.progressbarChanged.connect(self.update_processing)

#         self.progressbarAborted.connect(self.just_aborted)

    def do_abort(self):
        self.aborted = True

    def input_filer(self, *args, **kwargs):
        """
        Called when addFiles clicked: opens a file-brower and populates the 
        listFiles object
        """
        fnames = QtGui.QFileDialog.getOpenFileNames(self,
                         "Select one or more diffraction image files",
                         QtCore.QDir.currentPath(),
                         filter=self.tr("EDF image files (*.edf);;TIFF image files (*.tif);;CBF files (*.cbf);;MarCCD image files (*.mccd);;Any file (*)"))
                         # filter=self.tr("NeXuS files (*.nxs);;HDF5 files (*.h5);;HDF5 files (*.hdf5);;EDF image files (*.edf);;TIFF image files (*.tif);;CBF files (*.cbf);;MarCCD image files (*.mccd);;Any file (*)"))
        for i in fnames:
            self.list_dataset.append(DataSet(str_(i), None, None, None))

#         self.list_model.set_root(self.list_dataset.as_tree())
        tree = self.list_dataset.as_tree()
        print(tree)
        t0 = time.time()
        self.list_model.set_root(tree)
#         self.list_model = TreeModel(self, tree)
        print("set TreeModel: %.3fs" % (time.time() - t0))
        t0 = time.time()
        self.listFiles.setModel(self.list_model)
        print("create setModel: %.3fs" % (time.time() - t0))
        t0 = time.time()
        self.update_number_of_frames()
        print("create update_number_of_frames: %.3fs" % (time.time() - t0))



    def configure_diffraction(self, *arg, **kwarg):
        """
        """
        logger.info("in configure_diffraction")
        iw = IntegrateWidget(self)
        if self.integration_config:
            iw.widget.set_config(self.integration_config)
        res = iw.exec_()
        if res == QtGui.QDialog.Accepted:
            iw.widget.input_data = [i.path for i in self.list_dataset]
            self.integration_config = iw.get_config()
        print(self.integration_config)

    def configure_output(self, *args, **kwargs):
        """
        called when clicking on "outputFileSelector"
        """
        fname = QtGui.QFileDialog.getSaveFileName(self, "Output file",
                                                  QtCore.QDir.currentPath(),
                                                  filter=self.tr("NeXuS file (*.nxs);;HDF5 file (*.h5);;HDF5 file (*.hdf5)"))
        self.outputFile.setText(fname)


    def start_processing(self, *arg, **kwarg):
        logger.info("in start_processing")
        if not self.integration_config:
            result = QtGui.QMessageBox.warning(self, "Azimuthal Integration",
                                                   "You need to configure first the Azimuthal integration")
            if result:
                self.configure_diffraction()
            else:
                return
        if not str(self.outputFile.text()):
            result = QtGui.QMessageBox.warning(self, "Destination",
                                                   "You need to configure first the destination file")
            if result:
                self.configure_output()
            else:
                return
        config = self.get_config()
        self.progressBar.setRange(0, len(self.list_dataset))
        self.aborted = False
        self.display_processing(config)
        self.last_idx = -1
        self.processing_thread = threading.Thread(name="process", target=self.process, args=(config,))
        self.processing_thread.start()

    def update_number_of_frames(self):
        cnt = len(self.list_dataset)
        self.numberOfFrames.setText(str(cnt))

    def update_number_of_points(self):
        try:
            slow = int(self.slowMotorPts.text())
        except:
            slow = 1
        try:
            fast = int(self.fastMotorPts.text())
        except:
            fast = 1
        try:
            offset = int(self.offset.text())
        except:
            offset = 0
        self.numberOfPoints.setText(str(slow * fast + offset))

    def sort_input(self):
        self.list_dataset.sort(key=lambda i: i.path)
#         self.list_model = TreeModel(self, self.list_dataset.as_tree())
#         self.listFiles.setModel(self.list_model)
        self.list_model.set_root(self.list_dataset.as_tree())

    def get_config(self):
        """Return a dict with the plugin configuration which is JSON-serializable 
        """
        res = {
               "ai": self.integration_config,
               "experiment_title": str_(self.experimentTitle.text()).strip(),
               "fast_motor_name": str_(self.fastMotorName.text()).strip(),
               "slow_motor_name": str_(self.slowMotorName.text()).strip(),
               "fast_motor_points": int_(self.fastMotorPts.text()),
               "slow_motor_points": int_(self.slowMotorPts.text()),
               "offset": int_(self.offset.text()),
               "output_file": str_(self.outputFile.text()).strip(),
               "input_data": [i.as_tuple() for i in self.list_dataset]
               }
        return res

    def set_config(self, dico):
        """Set up the widget from dictionary
        
        @param  dico: dictionary 
        """
        self.integration_config = dico.get("ai", {})
        # TODO
        setup_data = {"experiment_title": self.experimentTitle.setText,
                      "fast_motor_name": self.fastMotorName.setText,
                      "slow_motor_name":self.slowMotorName.setText,
                      "fast_motor_points":lambda a:self.fastMotorPts.setText(str_(a)),
                      "slow_motor_points":lambda a:self.slowMotorPts.setText(str_(a)),
                      "offset":lambda a:self.offset.setText(str_(a)),
                      "output_file":self.outputFile.setText
                   }
        for key, value in setup_data.items():
            if key in dico:
                value(dico[key])
        self.list_dataset = ListDataSet(DataSet(*(str_(j) for j in i)) for i in dico.get("input_data", []))
        self.list_model.set_root(self.list_dataset.as_tree())
#         self.list_model = TreeModel(self, self.list_dataset.as_tree())
#         self.listFiles.setModel(self.list_model)
        self.update_number_of_frames()
        self.update_number_of_points()



    def dump(self, fname=None):
        """Save the configuration in a JSON file
        
        @param fname: file where the config is saved as JSON 
        """
        if fname is None:
            fname = self.json_file
        config = self.get_config()
        with open(fname, "w") as fd:
            fd.write(json.dumps(config, indent=2))
        return config

    def restore(self, fname=None):
        """Restore the widget from saved config
        
        @param fname: file where the config is saved as JSON
        """
        if fname is None:
            fname = self.json_file
        if not os.path.exists(fname):
            logger.warning("No such configuration file: %s" % fname)
            return
        with open(fname, "r") as fd:
            dico = json.loads(fd.read())
        self.set_config(dico)

    def save_config(self):
        logger.debug("save_config")
        json_file = str_(QtGui.QFileDialog.getSaveFileName(caption="Save configuration as json",
                                                           directory=self.json_file,
                                                           filter="Config (*.json)"))
        if json_file:
            self.dump(json_file)

    def process(self, config=None):
        """
        Called in a separate thread 
        """
        logger.info("process")
        t0 = time.time()
        with self.processing_sem:

            if config is None:
                config = self.dump()
            config_ai = config.get("ai", {})
            diffmap = DiffMap(npt_fast=config.get("fast_motor_points", 1),
                              npt_slow=config.get("slow_motor_points", 1),
                              npt_rad=config_ai.get("nbpt_rad", 1000),
                              npt_azim=config_ai.get("nbpt_azim", 1) if config_ai.get("do_2D") else None)
            diffmap.ai = AIWidget.make_ai(config_ai)
            diffmap.method = config_ai.get("method", "csr")
            diffmap.hdf5 = config.get("output_file", "unamed.h5")
            self.radial_data = diffmap.init_ai()
            self.data = diffmap.dataset
            for i, fn in enumerate(self.list_dataset):
                diffmap.process_one_file(fn.path)
                self.progressbarChanged.emit(i, diffmap._idx)
                if self.aborted:
                    logger.warning("Aborted by user")
                    self.progressbarChanged.emit(0, 0)
                    if diffmap.nxs:
                        diffmap.nxs.close()
                    return
            if diffmap.nxs:
                diffmap.nxs.close()
        logger.warning("Processing finished in %.3fs" % (time.time() - t0))
        self.progressbarChanged.emit(len(self.list_dataset), 0)

    def display_processing(self, config):
        """Setup the display for visualizing the processing
        
        @param config: configuration of the processing ongoing
        """
        self.fig = pyplot.figure()
        self.aximg = self.fig.add_subplot(2, 1, 1,
                                          xlabel=config.get("fast_motor_name", "Fast motor"),
                                          ylabel=config.get("slow_motor_name", "Slow motor"),
                                          xlim=(0, config.get("fast_motor_points", 1)),
                                          ylim=(0, config.get("slow_motor_points", 1)))
        self.aximg.set_title(config.get("experiment_title", "Diffraction imaging"))

        self.axplt = self.fig.add_subplot(2, 1, 2,
                                          xlabel=to_unit(config.get("ai").get("unit")).label,
                                          ylabel="Scattered intensity")
        self.axplt.set_title("Average diffraction pattern")


        self.fig.show()

    def update_processing(self, idx_file, idx_img):
        """ Update the process bar and the images 
        
        """
        self.progressBar.setValue(idx_file)
        if self.update_sem._Semaphore__value < 1:
            return
        with self.update_sem:
            try:
                data = self.data.value
            except ValueError:
                logger.warning("Dataset not valid")
                return
            npt = self.radial_data.size

            img = data.mean(axis=-1)
            if self.last_idx < 0:
                I = data[0, 0, :]
                self.plot = self.axplt.plot(self.radial_data, I)[0]
                self.img = self.aximg.imshow(img, interpolation="nearest")
            else:
                I = data.reshape(-1, npt)[:idx_img].mean(axis=0)
                self.plot.set_ydata(I)
                self.img.set_data(img)
            self.last_idx = idx_img
            self.fig.canvas.draw()
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.1)
