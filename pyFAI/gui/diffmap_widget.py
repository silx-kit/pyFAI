# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module with GUI for diffraction mapping experiments"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/02/2019"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import time
import json
import threading
import logging

from silx.gui import qt
from silx.gui import icons

from .matplotlib import pyplot
from ..utils import int_, str_, get_ui_file
from ..units import to_unit
from .widgets.WorkerConfigurator import WorkerConfigurator
from .. import worker
from ..io.integration_config import ConfigurationReader
from ..diffmap import DiffMap
from .utils.tree import ListDataSet, DataSet

logger = logging.getLogger(__name__)


class IntegrateDialog(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self)
        self.widget = WorkerConfigurator(self)
        self.widget.set1dIntegrationOnly(True)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.widget)
        buttons = qt.QDialogButtonBox(self)
        buttons.setStandardButtons(qt.QDialogButtonBox.Cancel | qt.QDialogButtonBox.Ok)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


class TreeModel(qt.QAbstractItemModel):
    def __init__(self, win, root_item):
        super(TreeModel, self).__init__(win)
        self._root_item = root_item
        self._win = win
        self._current_branch = None

    def update(self, new_root):
        self.beginResetModel()
        new_labels = [i.label for i in new_root.children]
        old_lables = [i.label for i in self._root_item.children]
        if new_labels == old_lables:
            print("update labels")
            self._root_item.update(new_root)
        else:
            print("replace labels")
            self._root_item.children = []
            for child in new_root.children:
                self._root_item.add_child(child)
        self.endResetModel()

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        pitem = parent.internalPointer()
        if (pitem is None) or (not parent.isValid()):
            pitem = self._root_item
        return len(pitem.children)

    def columnCount(self, parent):
        return 1

    def flags(self, midx):
        # if midx.column()==1:
        return qt.Qt.ItemIsEnabled

    def index(self, row, column, parent):
        pitem = parent.internalPointer()
        if not parent.isValid():
            pitem = self._root_item
        try:
            item = pitem.children[row]
        except IndexError:
            return qt.QModelIndex()
        return self.createIndex(row, column, item)

    def data(self, midx, role):
        """
        What to display depending on model_index and role
        """
        leaf = midx.internalPointer()
        if midx.column() == 0 and role == qt.Qt.DisplayRole:
            return leaf.label

    def headerData(self, section, orientation, role):
        if role == qt.Qt.DisplayRole and orientation == qt.Qt.Horizontal:
            # return ["Path", "shape"][section]
            return ["Path"][section]

    def parent(self, midx):
        item = midx.internalPointer()
        if (item is None) or (item is self._root_item):
            print(midx, midx.row(), midx.column())
            return  # QtCore.QModelIndex()
        pitem = item.parent
        if pitem is self._root_item:
            return qt.QModelIndex()
        row_idx = pitem.parent.children.index(pitem)
        return self.createIndex(row_idx, 0, pitem)


class DiffMapWidget(qt.QWidget):
    progressbarChanged = qt.Signal(int, int)
#     progressbarAborted = Signal()
    uif = "diffmap.ui"
    json_file = ".diffmap.json"

    def __init__(self):
        qt.QWidget.__init__(self)

        self.integration_config = {}
        self.list_dataset = ListDataSet()  # Contains all datasets to be treated.

        try:
            qt.loadUi(get_ui_file(self.uif), self)
        except AttributeError as _error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")

        pyfaiIcon = icons.getQIcon("pyfai:gui/images/icon")
        self.setWindowIcon(pyfaiIcon)

        self.aborted = False
        self.progressBar.setValue(0)
        self.list_model = TreeModel(self, self.list_dataset.as_tree())
        self.listFiles.setModel(self.list_model)
        self.listFiles.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.listFiles.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.create_connections()
        self.set_validator()
        self.update_number_of_frames()
        self.update_number_of_points()
        self.processing_thread = None
        self.processing_sem = threading.Semaphore()
        self.update_sem = threading.Semaphore()

        # disable some widgets:
        self.multiframe.setVisible(False)
        self.label_10.setVisible(False)
        self.frameShape.setVisible(False)
        # Online visualization
        self.fig = None
        self.axplt = None
        self.aximg = None
        self.img = None
        self.plot = None
        self.radial_data = None
        self.data_h5 = None  # one in hdf5 dataset while processing.
        self.data_np = None  # The numpy one is used only at the end.
        self.last_idx = -1
        self.slice = slice(0, -1, 1)  # Default slicing
        self._menu_file()

    def set_validator(self):
        validator = qt.QIntValidator(0, 999999, self)
        self.fastMotorPts.setValidator(validator)
        self.slowMotorPts.setValidator(validator)
        self.offset.setValidator(validator)

        float_valid = qt.QDoubleValidator(self)
        self.rMin.setValidator(float_valid)
        self.rMax.setValidator(float_valid)

    def create_connections(self):
        """Signal-slot connection
        """
        self.configureDiffraction.clicked.connect(self.configure_diffraction)
        self.outputFileSelector.clicked.connect(self.configure_output)
        self.runButton.clicked.connect(self.start_processing)
        self.saveButton.clicked.connect(self.save_config)
        self.abortButton.clicked.connect(self.do_abort)
        self.fastMotorPts.editingFinished.connect(self.update_number_of_points)
        self.slowMotorPts.editingFinished.connect(self.update_number_of_points)
        self.offset.editingFinished.connect(self.update_number_of_points)
        self.progressbarChanged.connect(self.update_processing)
        self.rMin.editingFinished.connect(self.update_slice)
        self.rMax.editingFinished.connect(self.update_slice)
        # self.listFiles.expanded.connect(lambda:self.listFiles.resizeColumnToContents(0))

    def _menu_file(self):
        # Drop-down file menu
        self.files_menu = qt.QMenu("Files")

        action_more = qt.QAction("add files", self.files)
        self.files_menu.addAction(action_more)
        action_more.triggered.connect(self.input_filer)

        action_sort = qt.QAction("sort files", self.files)
        self.files_menu.addAction(action_sort)
        action_sort.triggered.connect(self.sort_input)

        action_clear = qt.QAction("clear selected files", self.files)
        self.files_menu.addAction(action_clear)
        action_clear.triggered.connect(self.clear_selection)

        self.files.setMenu(self.files_menu)

    def do_abort(self):
        self.aborted = True

    def input_filer(self, *args, **kwargs):
        """
        Called when addFiles clicked: opens a file-browser and populates the
        listFiles object
        """
        filters = [
            # "NeXuS files (*.nxs)"
            # "HDF5 files (*.h5)"
            # "HDF5 files (*.hdf5)"
            "EDF image files (*.edf)",
            "TIFF image files (*.tif)",
            "CBF files (*.cbf)",
            "MarCCD image files (*.mccd)",
            "Any file (*)"]
        fnames = qt.QFileDialog.getOpenFileNames(self,
                                                 "Select one or more diffraction image files",
                                                 qt.QDir.currentPath(),
                                                 filter=self.tr(";;".join(filters)))
        if isinstance(fnames, tuple):
            # Compatibility with PyQt5
            fnames = fnames[0]

        for i in fnames:
            self.list_dataset.append(DataSet(str_(i), None, None, None))

        self.list_model.update(self.list_dataset.as_tree())
        self.update_number_of_frames()
        self.listFiles.resizeColumnToContents(0)

    def clear_selection(self, *args, **kwargs):
        """called to remove selected files from the list
        """
        print(self.listFiles.selectedIndexes())
        logger.warning("remove all files for now !! not yet implemented")
        self.list_dataset.empty()
        self.list_model.update(self.list_dataset.as_tree())

    def configure_diffraction(self, *arg, **kwarg):
        """
        """
        logger.info("in configure_diffraction")
        iw = IntegrateDialog(self)
        if self.integration_config:
            iw.widget.setConfig(self.integration_config)
        res = iw.exec_()
        if res == qt.QDialog.Accepted:
            self.integration_config = iw.widget.getConfig()
        print(json.dumps(self.integration_config, indent=2))

    def configure_output(self, *args, **kwargs):
        """
        called when clicking on "outputFileSelector"
        """
        fname = qt.QFileDialog.getSaveFileName(self, "Output file",
                                               qt.QDir.currentPath(),
                                               filter=self.tr("NeXuS file (*.nxs);;HDF5 file (*.h5);;HDF5 file (*.hdf5)"))
        if isinstance(fname, tuple):
            # Compatibility with PyQt5
            fname = fname[0]

        self.outputFile.setText(fname)

    def start_processing(self, *arg, **kwarg):
        logger.info("in start_processing")
        if not self.integration_config:
            result = qt.QMessageBox.warning(self,
                                            "Azimuthal Integration",
                                            "You need to configure first the Azimuthal integration")
            if result:
                self.configure_diffraction()
            else:
                return
        if not str(self.outputFile.text()):
            result = qt.QMessageBox.warning(self,
                                            "Destination",
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
        self.numberOfFrames.setText("list: %s, tree: %s" % (cnt, self.list_model._root_item.size))

    def update_number_of_points(self):
        try:
            slow = int(self.slowMotorPts.text())
        except ValueError:
            slow = 1
        try:
            fast = int(self.fastMotorPts.text())
        except ValueError:
            fast = 1
        try:
            offset = int(self.offset.text())
        except ValueError:
            offset = 0
        self.numberOfPoints.setText(str(slow * fast + offset))

    def sort_input(self):
        self.list_dataset.sort(key=lambda i: i.path)
        self.list_model.update(self.list_dataset.as_tree())

    def get_config(self):
        """Return a dict with the plugin configuration which is JSON-serializable
        """
        res = {"ai": self.integration_config,
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

        :param  dico: dictionary
        """
        self.integration_config = dico.get("ai", {})
        # TODO
        setup_data = {"experiment_title": self.experimentTitle.setText,
                      "fast_motor_name": self.fastMotorName.setText,
                      "slow_motor_name": self.slowMotorName.setText,
                      "fast_motor_points": lambda a: self.fastMotorPts.setText(str_(a)),
                      "slow_motor_points": lambda a: self.slowMotorPts.setText(str_(a)),
                      "offset": lambda a: self.offset.setText(str_(a)),
                      "output_file": self.outputFile.setText
                      }
        for key, value in setup_data.items():
            if key in dico:
                value(dico[key])
        self.list_dataset = ListDataSet(DataSet(*(str_(j) for j in i)) for i in dico.get("input_data", []))
        self.list_model.update(self.list_dataset.as_tree())
        self.update_number_of_frames()
        self.update_number_of_points()
        self.listFiles.resizeColumnToContents(0)

    def dump(self, fname=None):
        """Save the configuration in a JSON file

        :param fname: file where the config is saved as JSON
        """
        if fname is None:
            fname = self.json_file
        config = self.get_config()
        with open(fname, "w") as fd:
            fd.write(json.dumps(config, indent=2))
        return config

    def restore(self, fname=None):
        """Restore the widget from saved config

        :param fname: file where the config is saved as JSON
        """
        if fname is None:
            fname = self.json_file
        if not os.path.exists(fname):
            logger.warning("No such configuration file: %s", fname)
            return
        with open(fname, "r") as fd:
            dico = json.loads(fd.read())
        self.set_config(dico)

    def save_config(self):
        logger.debug("save_config")
        json_file = qt.QFileDialog.getSaveFileName(caption="Save configuration as json",
                                                   directory=self.json_file,
                                                   filter="Config (*.json)")
        if isinstance(json_file, tuple):
            # Compatibility with PyQt5
            json_file = json_file[0]

        if json_file:
            self.dump(json_file)

    def process(self, config=None):
        """
        Called in a separate thread
        """
        logger.info("process")
        t0 = time.time()
        with self.processing_sem:
            config = self.dump()
            config_ai = config.get("ai", {})
            config_ai = config_ai.copy()

            diffmap = DiffMap(npt_fast=config.get("fast_motor_points", 1),
                              npt_slow=config.get("slow_motor_points", 1),
                              npt_rad=config_ai.get("nbpt_rad", 1000),
                              npt_azim=config_ai.get("nbpt_azim", 1) if config_ai.get("do_2D") else None)
            diffmap.inputfiles = [i.path for i in self.list_dataset]  # in case generic detector without shape
            diffmap.ai = worker.make_ai(config_ai)
            # TODO: This diffmap configuration file should be cleaned up
            reader = ConfigurationReader(config_ai)
            diffmap.method = reader.pop_method("csr")
            diffmap.unit = to_unit(config_ai.get("unit", "2th_deg"))
            diffmap.hdf5 = config.get("output_file", "unamed.h5")
            self.radial_data = diffmap.init_ai()
            self.data_h5 = diffmap.dataset
            for i, fn in enumerate(self.list_dataset):
                diffmap.process_one_file(fn.path)
                self.progressbarChanged.emit(i, diffmap._idx)
                if self.aborted:
                    logger.warning("Aborted by user")
                    self.progressbarChanged.emit(0, 0)
                    if diffmap.nxs:
                        self.data_np = diffmap.dataset.value
                        diffmap.nxs.close()
                    return
            if diffmap.nxs:
                self.data_np = diffmap.dataset.value
                diffmap.nxs.close()
        logger.warning("Processing finished in %.3fs", time.time() - t0)
        self.progressbarChanged.emit(len(self.list_dataset), 0)

    def display_processing(self, config):
        """Setup the display for visualizing the processing

        :param config: configuration of the processing ongoing
        """
        self.fig = pyplot.figure(figsize=(12, 5))
        self.aximg = self.fig.add_subplot(1, 2, 1,
                                          xlabel=config.get("fast_motor_name", "Fast motor"),
                                          ylabel=config.get("slow_motor_name", "Slow motor"),
                                          xlim=(-0.5, config.get("fast_motor_points", 1) - 0.5),
                                          ylim=(-0.5, config.get("slow_motor_points", 1) - 0.5))
        self.aximg.set_title(config.get("experiment_title", "Diffraction imaging"))

        self.axplt = self.fig.add_subplot(1, 2, 2,
                                          xlabel=to_unit(config.get("ai").get("unit")).label,
                                          ylabel="Scattered intensity")
        self.axplt.set_title("Average diffraction pattern")
        self.fig.show()

    def update_processing(self, idx_file, idx_img):
        """ Update the process bar and the images

        :param idx_file: file number
        :param idx_img: frame number
        """
        if idx_file >= 0:
            self.progressBar.setValue(idx_file)

        # Check if there is a free semaphore without blocking
        if self.update_sem.acquire(blocking=False):
            self.update_sem.release()
        else:
            # It's full
            return

        with self.update_sem:
            try:
                data = self.data_h5.value
            except ValueError:
                data = self.data_np
            if self.radial_data is None:
                return

            npt = self.radial_data.size

            if self.last_idx < 0:
                self.update_slice()
                intensity = data[0, 0, :]
                img = data[:, :, self.slice].mean(axis=2)
                self.plot = self.axplt.plot(self.radial_data, intensity)[0]
                self.img = self.aximg.imshow(img, interpolation="nearest")
            else:
                img = data[:, :, self.slice].mean(axis=2)
                intensity = data.reshape(-1, npt)[:idx_img].mean(axis=0)
                self.plot.set_ydata(intensity)
                self.img.set_data(img)
            self.last_idx = idx_img
            self.fig.canvas.draw()
            qt.QCoreApplication.processEvents()
            time.sleep(0.1)

    def update_slice(self, *args):
        """
        Update the slice
        """
        if self.radial_data is None:
            return
        try:
            qmin = float(self.rMin.text())
        except ValueError:
            qmin = 0
        try:
            qmax = float(self.rMax.text())
        except ValueError:
            qmax = 1e300

        start = (self.radial_data < qmin).sum()
        stop = (self.radial_data <= qmax).sum()
        self.slice = slice(start, stop)
        self.update_processing(-1, self.last_idx)
