#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module with GUI for diffraction mapping experiments"""
from __future__ import annotations
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/05/2025"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import time
import json

import logging
import numpy
import fabio
from fabio.fabioutils import exists as fabio_exists
from silx.gui import qt
from silx.gui import icons

from .matplotlib import pyplot, colors
import threading
from ..utils import int_, str_, float_, get_ui_file
from ..units import to_unit
from .widgets.WorkerConfigurator import WorkerConfigurator
from ..diffmap import DiffMap
from .utils.tree import ListDataSet, DataSet
from .dialog import MessageBox
from ..io.integration_config import WorkerConfig
from ..io.diffmap_config import DiffmapConfig, MotorRange
from .pilx import MainWindow as pilx_main
logger = logging.getLogger(__name__)
logger.setLevel(logging.getLogger("pyFAI").level)
lognorm = colors.LogNorm()


class IntegrateDialog(qt.QDialog):

    def __init__(self, parent=None):
        qt.QDialog.__init__(self)
        self.widget = WorkerConfigurator(self)
        # self.widget.set1dIntegrationOnly(True)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.widget)
        buttons = qt.QDialogButtonBox(self)
        buttons.setStandardButtons(qt.QDialogButtonBox.Cancel | qt.QDialogButtonBox.Ok)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


class TreeModel(qt.QAbstractItemModel):

    def __init__(self, win, root_item):
        super().__init__(win)
        self._root_item = root_item
        self._win = win
        self._current_branch = None

    def update(self, new_root):
        self.beginResetModel()
        new_labels = [i.label for i in new_root.children]
        old_lables = [i.label for i in self._root_item.children]
        if new_labels == old_lables:
            self._root_item.update(new_root)
        else:
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
            return  # QtCore.QModelIndex()
        pitem = item.parent
        if pitem is self._root_item:
            return qt.QModelIndex()
        row_idx = pitem.parent.children.index(pitem)
        return self.createIndex(row_idx, 0, pitem)


class DiffMapWidget(qt.QWidget):
    progressbarChanged = qt.Signal(int, int)
    processingFinished = qt.Signal()
    pilxDisplay = qt.Signal(str)
    uif = "diffmap.ui"
    json_file = ".diffmap.json"

    def __init__(self):
        qt.QWidget.__init__(self)

        self.integration_config = None
        self.list_dataset = ListDataSet()  # Contains all datasets to be treated.

        try:
            qt.loadUi(get_ui_file(self.uif), self)
        except AttributeError as _error:
            logger.error("It looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            logger.error(f"{type(_error)}: {_error}")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")

        pyfaiIcon = icons.getQIcon("pyfai:gui/images/icon")
        self.setWindowIcon(pyfaiIcon)
        self.abort = threading.Event()
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
        # self.multiframe.setVisible(False)
        # self.label_10.setVisible(False)
        # self.frameShape.setVisible(False)
        self.frameShape.setText("Click `Scan`")
        # Online visualization
        self.fig = None
        self.axplt = None
        self.aximg = None
        self.img = None
        self.plot = None
        self.pilx_widget = None
        self.radial_data = None
        self.azimuthal_data = None
        self.data_h5 = None  # one in hdf5 dataset while processing.
        self.data_np = None  # The numpy one is used only at the end.
        self.last_idx = -1
        self.slice = slice(0, -1, 1)  # Default slicing
        self._menu_file()
        self.update_period = 1  # Update the live plot every second or so
        self.next_update = time.perf_counter()

    def set_validator(self):
        validator0 = qt.QIntValidator(0, 999999, self)
        validator1 = qt.QIntValidator(1, 999999, self)
        self.fastMotorPts.setValidator(validator1)
        self.slowMotorPts.setValidator(validator1)
        self.offset.setValidator(validator0)

        float_valid = qt.QDoubleValidator(self)
        self.rMin.setValidator(float_valid)
        self.rMax.setValidator(float_valid)
        self.fastMotorMinimum.setValidator(float_valid)
        self.fastMotorMaximum.setValidator(float_valid)
        self.slowMotorMinimum.setValidator(float_valid)
        self.slowMotorMaximum.setValidator(float_valid)

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
        self.processingFinished.connect(self.close_fig)
        self.pilxDisplay.connect(self.start_visu)
        # self.listFiles.expanded.connect(lambda:self.listFiles.resizeColumnToContents(0))
        self.scanButton.clicked.connect(self.scan_input_files)

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
        logger.info("DiffMapWidget.do_abort")
        self.abort.set()
        _button = qt.QMessageBox.warning(self,
                    "Processing aborted !",
                    "User aborted the processing.",
                    buttons=qt.QMessageBox.Ok)


    def input_filer(self, *args, **kwargs):
        """
        Called when addFiles clicked: opens a file-browser and populates the
        listFiles object
        """
        filters = [
            "HDF5 files (*.h5)",
            "HDF5 files (*.hdf5)",
            "NeXuS files (*.nxs)",
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
        logger.warning("remove all files for now !! not yet implemented")
        self.list_dataset.empty()
        self.list_model.update(self.list_dataset.as_tree())

    def configure_diffraction(self, *arg, **kwarg):
        """
        """
        logger.info("in configure_diffraction")
        iw = IntegrateDialog(self)
        if self.integration_config is not None:
            iw.widget.setWorkerConfig(self.integration_config)
        while True:
            res = iw.exec_()
            if res == qt.QDialog.Accepted:
                self.integration_config = iw.widget.getWorkerConfig()
                if self.integration_config.nbpt_rad:
                    break
                else:
                    qt.QMessageBox.about(self, "Unconsistent configuration", "Some essential parameters are missing ... Did you set the radial number of points ?")
            else:
                break

    def configure_output(self, *args, **kwargs):
        """
        called when clicking on "outputFileSelector"
        """
        fname = qt.QFileDialog.getSaveFileName(self, "Output file",
                                               qt.QDir.currentPath(),
                                               filter=self.tr("HDF5 file (*.h5);;HDF5 file (*.hdf5);;NeXuS file (*.nxs)"))
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
        if self.fastMotorPts.text() == "" or self.slowMotorPts.text() == "" or int(self.fastMotorPts.text()) * int(self.slowMotorPts.text()) == 0:
            result = qt.QMessageBox.warning(self,
                                            "Grid size",
                                            "The number of steps for the grid (fast/slow motor) cannot be empty or null")
            if result:
                return

        config = self.get_config()
        self.progressBar.setRange(0, self.number_of_points)
        self.abort.clear()
        self.display_processing(config)
        self.last_idx = -1
        self.processing_thread = threading.Thread(name="process", target=self.process, args=(config,))
        self.processing_thread.start()

    def update_number_of_frames(self):
        cnt = len(self.list_dataset)
        self.numberOfFrames.setText(f"list: {cnt}, tree: {self.list_model._root_item.size}")

    @property
    def number_of_points(self):
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
        return slow * fast + offset

    def update_number_of_points(self):
        self.numberOfPoints.setText(str(self.number_of_points))

    def sort_input(self):
        self.list_dataset.sort(key=lambda i: i.path)
        self.list_model.update(self.list_dataset.as_tree())

    def scan_input_files(self):
        """ open all files, count the number of frames and check their size ?

        :return: number of frames
        """
        total_frames = 0
        shape = None
        for i in self.list_dataset:
            fn = i.path
            if fabio_exists(fn):
                try:
                    with fabio.open(fn) as fimg:
                        new_shape = fimg.shape
                        nframes = fimg.nframes
                except Exception as error:
                    MessageBox.exception(self, f"Unable to read file {fn}", error, None)
                    return
                if shape is None:
                    shape = new_shape
                elif shape != new_shape:
                    MessageBox.exception(self, "Frame shape missmatch: got {fimg.shape}, expected {shape}", None, None)
                    return
                total_frames += nframes
        self.numberOfFrames.setText(str(total_frames))
        if shape:
            self.frameShape.setText(f"{shape[1]} x {shape[0]}")
            if self.integration_config is not None:
                self.integration_config.shape = shape
        return total_frames

    def get_dict_config(self):
        """Return a dict with the plugin configuration which is JSON-serializable
        """
        return self.get_diffmap_config().as_dict()
    get_config = get_dict_config

    def get_diffmap_config(self):
        """Return a DiffmapConfig instance with all the settings from the widget
        """
        config = DiffmapConfig(ai=self.integration_config,
                               input_data=self.list_dataset)
        config.experiment_title = str_(self.experimentTitle.text()).strip()
        config.offset = int_(self.offset.text())
        config.zigzag_scan = bool(self.zigzagBox.isChecked())
        config.output_file = str_(self.outputFile.text()).strip()

        if config.fast_motor is None:
            config.fast_motor = MotorRange()
        config.fast_motor.start = float_(self.fastMotorMinimum.text())
        config.fast_motor.stop = float_(self.fastMotorMaximum.text())
        config.fast_motor.name = str_(self.fastMotorName.text())
        config.fast_motor.points = int_(self.fastMotorPts.text())
        if config.slow_motor is None:
            config.slow_motor = MotorRange()
        config.slow_motor.start = float_(self.slowMotorMinimum.text())
        config.slow_motor.stop = float_(self.slowMotorMaximum.text())
        config.slow_motor.name = str_(self.slowMotorName.text())
        config.slow_motor.points = int_(self.slowMotorPts.text())
        return config

    def set_config(self, dico):
        """Set up the widget from dictionary

        :param  dico: dictionary or DiffMapConfig instance
        """

        if isinstance(dico, DiffmapConfig):
            config = dico
        else:
            config = DiffmapConfig.from_dict(dico)

        self.integration_config = WorkerConfig() if config.ai is None else config.ai
        self.experimentTitle.setText(config.experiment_title or "")

        if config.fast_motor is not None:
            self.fastMotorName.setText(config.fast_motor.name or "")
            self.fastMotorPts.setText(str_(config.fast_motor.points))
            self.fastMotorMinimum.setText(str(config.fast_motor.start))
            self.fastMotorMaximum.setText(str(config.fast_motor.stop))

        if config.slow_motor is not None:
            self.slowMotorPts.setText(str_(config.slow_motor.points))
            self.slowMotorMinimum.setText(str(config.slow_motor.start))
            self.slowMotorMaximum.setText(str(config.slow_motor.stop))
            self.slowMotorName.setText(config.slow_motor.name or "")

        self.outputFile.setText(config.output_file)
        self.offset.setText(str_(config.offset))
        self.zigzagBox.setChecked(bool(config.zigzag_scan))

        self.list_dataset = config.input_data
        self.list_model.update(self.list_dataset.as_tree())
        self.update_number_of_frames()
        self.update_number_of_points()
        self.listFiles.resizeColumnToContents(0)

    def dump(self, fname=None):
        """Save the configuration in a JSON file

        :param fname: file where the config is saved as JSON
        :return: configuration as DiffmapConfig dataclass
        """
        if fname is None:
            fname = self.json_file
        config = self.get_diffmap_config()
        with open(fname, "w") as fd:
            fd.write(json.dumps(config.as_dict(), indent=2))
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
        with open(fname) as fd:
            dico = json.loads(fd.read())
        self.set_config(dico)

    def save_config(self):
        logger.info("DiffmapWidget.save_config()")
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
        logger.info("DiffmapWidget.process() thread starting")
        t0 = time.perf_counter()
        self.next_update = t0 + self.update_period
        last_processed_file = None
        with self.processing_sem:
            config = self.dump()
            config_ai = self.integration_config
            diffmap = DiffMap()
            diffmap.set_config(config)
            diffmap.configure_worker(config_ai)
            self.radial_data, self.azimuthal_data = diffmap.init_ai()
            self.data_h5 = diffmap.dataset
            for i, fn in enumerate(self.list_dataset):
                diffmap.process_one_file(fn.path,
                    callback=lambda fn, idx:self.progressbarChanged.emit(i, diffmap._idx),
                    abort = self.abort)

                if self.abort.is_set():
                    logger.warning("Aborted by user")
                    self.progressbarChanged.emit(0, 0)
                    self.processingFinished.emit()
                    break
            if diffmap.nxs:
                self.data_np = diffmap.dataset[()]
                last_processed_file = diffmap.nxs.filename
                diffmap.nxs.close()
        if not self.abort.is_set():
            logger.warning("Processing finished in %.3fs", time.perf_counter() - t0)
            self.progressbarChanged.emit(len(self.list_dataset), diffmap._idx)
            self.finish_processing(last_processed_file)
        logger.info("DiffmapWidget.process thread ending")


    def display_processing(self, config):
        """Setup the display for visualizing the processing

        :param config: configuration of the processing ongoing
        """
        logger.debug("DiffmapWidget.display_processing")
        if isinstance(config, dict):
            config = DiffmapConfig.from_dict(config)

        self.fig = pyplot.figure(figsize=(12, 5))
        self.aximg = self.fig.add_subplot(1, 2, 1,
                                          xlabel="Fast motor" if config.fast_motor is None else config.fast_motor.name,
                                          ylabel="Slow motor" if config.slow_motor is None else config.slow_motor.name,
                                          xlim=(-0.5, (config.fast_motor.points or 1) - 0.5),
                                          ylim=(-0.5, (config.slow_motor.points or 1) - 0.5))
        self.aximg.set_title(config.experiment_title or "Diffraction imaging")
        self.axplt = self.fig.add_subplot(1, 2, 2,
                                          xlabel=to_unit(config.ai.unit).label,
                                          # ylabel="Scattered intensity"
                                          )
        self.axplt.set_title("Average diffraction pattern")
        self.fig.show()

    def update_processing(self, idx_file, idx_img):
        """ Update the process bar and the images

        :param idx_file: file number
        :param idx_img: frame number
        """
        cmap = "inferno"

        if idx_img >= 0:
            self.progressBar.setValue(idx_img)

        if time.perf_counter() < self.next_update:
            # Do not update too frequently (kills performances
            return

        if self.update_sem.acquire(blocking=False):
            # Check if there is a free semaphore without blocking
            self.update_sem.release()
        else:
            # It's full
            return

        with self.update_sem:
            self.next_update = time.perf_counter() + self.update_period
            try:
                data = self.data_h5[()]
            except (ValueError, RuntimeError):
                data = self.data_np
            if self.radial_data is None or self.fig is None:
                return

            intensity = numpy.nanmean(data, axis=(0, 1))
            if self.last_idx < 0:
                self.update_slice()

                if data.ndim == 4:
                    img = data[..., self.slice].mean(axis=(2, 3))

                    self.plot = self.axplt.imshow(intensity,
                                                  interpolation="nearest",
                                                  norm=lognorm,
                                                  cmap=cmap,
                                                  origin="lower",
                                                  extent=[self.radial_data.min(), self.radial_data.max(),
                                                          self.azimuthal_data.min(), self.azimuthal_data.max()],
                                                  aspect="auto",)
                    self.axplt.set_ylabel("Azimuthal angle (°)")
                else:
                    img = data[..., self.slice].mean(axis=-1)
                    self.axplt.set_ylabel("Scattered intensity")
                    self.plot = self.axplt.plot(self.radial_data, intensity)[0]
                self.img = self.aximg.imshow(img,
                                             interpolation="nearest",
                                             cmap=cmap,
                                             origin="lower",
                                             )
            else:
                if data.ndim == 4:
                    img = numpy.nanmean(data[..., self.slice], axis=(2, 3))
                    img[img <= lognorm.vmin] = numpy.nan
                    self.plot.set_data(intensity)
                else:
                    img = data[:,:, self.slice].mean(axis=2)
                    self.plot.set_ydata(intensity)
                self.img.set_data(img)
            self.last_idx = idx_img
            try:
                self.fig.canvas.draw()
            except Exception as err:
                logger.error(f"{type(err)}: {err} intercepted in matplotlib drawing")

            qt.QCoreApplication.processEvents()
            time.sleep(0.1)

    def finish_processing(self, start_pilx=None):
        """ close the process bar widget and the images

        :param start_pilx: (str) open the pilx visualization tool with the given file
        """
        logger.debug("DiffmapWidget.finish_processing")
        self.processingFinished.emit()

        if start_pilx and isinstance(start_pilx, str) and os.path.exists(start_pilx):
            self.pilxDisplay.emit(start_pilx)

    def close_fig(self):
        logger.info("close figures from main")
        with self.update_sem:
            if self.fig:
                pyplot.close(self.fig)
                qt.QCoreApplication.processEvents()
                self.fig = None
                self.plot = None
                self.img = None
                self.axplt = None
                self.aximg = None


    def start_visu(self, filename):
        """Open a pilx window

        :param filename: name of the HDF5 file to open
        """
        logger.info("DiffmapWidget.start_visu")
        if self.pilx_widget is not None:
            self.pilx_widget.close()
            qt.QCoreApplication.processEvents()
        if filename is not None:
            self.pilx_widget = pilx_main.MainWindow()
            self.pilx_widget.initData(filename)
            self.pilx_widget.show()

    def update_slice(self, *args):
        """
        Update the slice
        """
        logger.info("DiffMapWidget.update_slice %s", args)
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
