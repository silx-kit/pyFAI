#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2013-2025 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""GUI tool for configuring azimuthal integration on series of files."""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/10/2025"
__satus__ = "production"

import sys
import time
import numpy
import os.path
import collections
import contextlib
from argparse import ArgumentParser
import logging
import fabio
from .. import utils, io, version as pyFAI_version, date as pyFAI_date
from ..io import DefaultAiWriter, HDF5Writer
from ..io.integration_config import WorkerConfig
from ..utils.shell import ProgressBar
from ..utils import logging_utils, header_utils
from ..worker import Worker
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)


def integrate_gui(options, args):
    from silx.gui import qt
    from ..gui.IntegrationDialog import IntegrationDialog
    from ..gui.IntegrationDialog import IntegrationProcess

    app = qt.QApplication([])

    from ..gui.ApplicationContext import ApplicationContext
    settings = qt.QSettings(qt.QSettings.IniFormat,
                            qt.QSettings.UserScope,
                            "pyfai",
                            "pyfai-integrate",
                            None)
    context = ApplicationContext(settings)

    def moveCenterTo(window, center):
        half = window.size() * 0.5
        half = qt.QPoint(half.width(), half.height())
        corner = center - half
        window.move(corner)

    def validateConfig():
        config = window.get_config()
        reason = Worker.validate_config(config, raise_exception=None)
        if reason is None:
            processData(config)
        else:
            "Invalid configuration"
            qt.QMessageBox.warning(window, "Inconsistent configuration", reason)

    def processData(config=None):
        center = window.geometry().center()
        input_data = window.input_data
        while input_data is None or len(input_data) == 0:
            dialog = qt.QFileDialog(directory=os.getcwd())
            dialog.setWindowTitle("Select images to integrate")

            from ..gui.utils import FilterBuilder
            builder = FilterBuilder.FilterBuilder()
            builder.addImageFormat("EDF image files", "edf")
            builder.addImageFormat("TIFF image files", "tif tiff")
            builder.addImageFormat("NumPy binary files", "npy")
            builder.addImageFormat("CBF files", "cbf")
            builder.addImageFormat("MarCCD image files", "mccd")
            builder.addImageFormat("HDF5 image stack", "h5 hdf5 nxs")
            dialog.setNameFilters(builder.getFilters())

            dialog.setFileMode(qt.QFileDialog.ExistingFiles)
            moveCenterTo(dialog, center)
            result = dialog.exec_()
            if not result:
                return
            input_data = [str(i) for i in dialog.selectedFiles()]
            center = dialog.geometry().center()
            dialog.close()

        window.setVisible(False)
        window.deleteLater()

        if config is None:
            config = window.get_worker_config()
        elif isinstance(config, dict):
            config = WorkerConfig.from_dict(config)

        dialog = IntegrationProcess(None)
        dialog.adjustSize()
        moveCenterTo(dialog, center)

        class QtProcess(qt.QThread):

            def run(self):
                observer = dialog.createObserver(qtSafe=True)
                config.monitor_name = options.monitor_key
                process(input_data, window.output_path, config, observer, options.write_mode, format_=options.format.lower())

        qtProcess = QtProcess()
        qtProcess.start()

        result = dialog.exec_()
        if result:
            qt.QMessageBox.information(dialog,
                                       "Integration",
                                       "Batch processing completed.")
        else:
            qt.QMessageBox.information(dialog,
                                       "Integration",
                                       "Batch processing interrupted.")
        dialog.deleteLater()

    window = IntegrationDialog(args, options.output, json_file=options.json, context=context)
    window.batchProcessRequested.connect(validateConfig)
    window.show()

    result = app.exec_()
    context.saveSettings()
    return result


def get_monitor_value(image, monitor_key):
    """Return the monitor value from an image using an header key.

    :param fabio.fabioimage.FabioImage image: Image containing the header
    :param str monitor_key: Key containing the monitor
    :return: returns the monitor else returns 1.0
    :rtype: float
    """
    if monitor_key is None or monitor_key == "":
        return 1.0
    try:
        monitor = header_utils.get_monitor_value(image, monitor_key)
        return monitor
    except header_utils.MonitorNotFound:
        logger.warning("Monitor %s not found. No normalization applied.", monitor_key)
        return 1.0
    except Exception as e:
        logger.warning("Fail to load monitor. No normalization applied. %s", str(e))
        return 1.0


class IntegrationObserver(object):
    """Interface providing access to the to the processing of the `process`
    function."""

    def __init__(self):
        self.__is_interruption_requested = False

    def is_interruption_requested(self):
        return self.__is_interruption_requested

    def request_interruption(self):
        self.__is_interruption_requested = True

    def worker_initialized(self, worker):
        """
        Called when the worker is initialized

        :param int data_count: Number of data to integrate
        """
        pass

    def processing_started(self, data_count):
        """
        Called before starting the full processing.

        :param int data_count: Number of data to integrate
        """
        pass

    def processing_data(self, data_info, approximate_count=None):
        """
        Start processing the data `data_info`

        :param DataInfo data_info: Contains data and metadata from the data
            to integrate
        :param int approximate_count: If set, the amount of total data to
            process have changed
        """
        pass

    def data_result(self, data_info, result):
        """
        Called after each data processing, with the result

        :param DataInfo data_info: Contains data and metadata from the data
            to integrate
        :param object result: Result of the integration.
        """
        pass

    def processing_interrupted(self, reason=None):
        """Called before `processing_finished` if the processing was
        interrupted.

        :param [str,Exception,None] error: A reason of the interruption.
        """
        pass

    def processing_succeeded(self):
        """Called before `processing_finished` if the processing succedded."""
        pass

    def processing_finished(self):
        """Called when the full processing is finisehd (interupted or not)."""
        pass


class ShellIntegrationObserver(IntegrationObserver):
    """
    Implement `IntegrationObserver` as a shell display.
    """

    def __init__(self):
        super(ShellIntegrationObserver, self).__init__()
        self._progress_bar = None
        self.__previous_sigint_callback = None

    def __signal_handler(self, sig, frame):
        logger.warning("Abort requested (please wait until end of the program execution)")
        self.request_interruption()

    def __connect_interrupt(self):
        import signal
        previous = signal.signal(signal.SIGINT, self.__signal_handler)
        self.__previous_sigint_callback = previous

    def __disconnect_interrupt(self):
        import signal
        previous = self.__previous_sigint_callback
        signal.signal(signal.SIGINT, previous)

    def processing_started(self, data_count):
        self._progress_bar = ProgressBar("Integration", data_count, 20)
        self.__connect_interrupt()

    def processing_data(self, data_info, approximate_count=None):
        if data_info.source_filename:
            if data_info.data_id == 0 and data_info.frame_id in [0, None]:
                # While we can't execute independantly the preprocessing
                message = "Preprocessing"
            elif len(data_info.source_filename) > 100:
                message = os.path.basename(data_info.source_filename)
            else:
                message = data_info.source_filename
        else:
            message = ""
        self._progress_bar.update(data_info.data_id + 1,
                                  message=message,
                                  max_value=approximate_count)

    def processing_finished(self):
        self.__disconnect_interrupt()
        self._progress_bar.clear()
        self._progress_bar = None

    def hide_info(self):
        if self._progress_bar is not None:
            self._progress_bar.clear()

    def show_info(self):
        if self._progress_bar is not None:
            self._progress_bar.display()


DataInfo = collections.namedtuple("DataInfo", "source source_id frame_id fabio_image data_id data header source_filename")


class DataSource(object):
    """Source of data to integrate."""

    def __init__(self, statistics):
        self._items = []
        self._statistics = statistics
        self._frames_per_items = []
        self._shape = None

    def append(self, item):
        self._items.append(item)

    def approximate_count(self):
        """Returns the number of frames contained in the data source.

        To speed up the processing time, this value could be approximate.
        Especially with file series, and EDF multiframes.

        :type: int"""
        if len(self._items) == 0:
            return 0
        known_frames = sum(self._frames_per_items)
        missing_items = len(self._items) - len(self._frames_per_items)
        if missing_items <= 0:
            # NOTE: Precondition, a feeded _frames_per_items will not be edited in between
            return known_frames

        if known_frames != 0:
            averate_frame_per_item = known_frames / len(self._frames_per_items)
        else:
            averate_frame_per_item = 1

        result = known_frames + missing_items * averate_frame_per_item
        return result

    def count(self):
        return len(self._items)

    def is_single_multiframe(self):
        if len(self._items) == 0:
            return False
        if len(self._items) > 1:
            return False

        item = self._items[0]
        if isinstance(item, (str,)):
            with fabio.open(item) as fabio_image:
                multiframe = fabio_image.nframes > 1
        elif isinstance(item, fabio.fabioimage.FabioImage):
            fabio_image = item
            multiframe = fabio_image.nframes > 1
        elif isinstance(item, numpy.ndarray):
            multiframe = len(item.shape) > 2 and len(item)
        return multiframe

    def is_multiframe(self):
        if len(self._items) == 0:
            return False
        if len(self._items) > 1:
            return True
        # FIXME: Should be improved if needed
        for count, _ in enumerate(self._iter_item_frames()):
            pass
        return count > 0

    def _iter_item_frames(self, iitem, start_id, item):
        if isinstance(item, (str,)):
            with self._statistics.time_reading():
                fabio_image = fabio.open(item)
            filename = fabio_image.filename
            was_openned = True
        elif isinstance(item, fabio.fabioimage.FabioImage):
            fabio_image = item
            filename = fabio_image.filename
            was_openned = False
        elif isinstance(item, numpy.ndarray):
            filename = None
            fabio_image = None

        if fabio_image is not None:
            # TODO: Reach nframes here could slow down the reading
            if fabio_image.nframes > 1:
                self._frames_per_items.append(fabio_image.nframes)
                for iframe in range(fabio_image.nframes):
                    with self._statistics.time_reading():
                        fimg = fabio_image.getframe(iframe)
                        data = fimg.data[...]
                    yield DataInfo(source=item,
                                   source_id=iitem,
                                   frame_id=iframe,
                                   data_id=start_id + iframe,
                                   data=data,
                                   fabio_image=fimg,
                                   header=fimg.header,
                                   source_filename=filename)
            else:
                self._frames_per_items.append(1)
                with self._statistics.time_reading():
                    data = fabio_image.data[...]
                yield DataInfo(source=item,
                               source_id=iitem,
                               frame_id=None,
                               data_id=start_id,
                               data=data,
                               fabio_image=fabio_image,
                               header=fabio_image.header,
                               source_filename=filename)
            if was_openned:
                fabio_image.close()
        else:
            if item.ndim == 3:
                self._frames_per_items.append(len(item))
                for iframe, data in enumerate(item):
                    with self._statistics.time_reading():
                        data = data[...]
                    yield DataInfo(source=item,
                                   source_id=iitem,
                                   frame_id=iframe,
                                   data_id=start_id + iframe,
                                   data=data,
                                   fabio_image=None,
                                   header=None,
                                   source_filename=filename)
            else:
                self._frames_per_items.append(1)
                data = item
                with self._statistics.time_reading():
                    data = data[...]
                yield DataInfo(source=item,
                               source_id=iitem,
                               data_id=start_id,
                               frame_id=None,
                               data=data,
                               fabio_image=None,
                               header=None,
                               source_filename=filename)

    def basename(self):
        """Returns a basename identifying this data source"""
        if len(self._items) == 0:
            raise RuntimeError("No ")
        if len(self._items) == 1:
            return self._items[0]
        return os.path.commonprefix(self._items)

    def frames(self):
        """
        Iterate all the frames from this data source.

        :rtype: Iterator[DataInfo]
        """
        next_id = 0
        for iitem, item in enumerate(self._items):
            for data_info in self._iter_item_frames(iitem, next_id, item):
                yield data_info
            if data_info.frame_id is not None:
                next_id += data_info.frame_id
            next_id += 1

    @property
    def shape(self):
        if self._shape is None and self._items:
            item = self._items[0]
            fabio_image = None
            if isinstance(item, (str,)):
                with self._statistics.time_reading():
                    fabio_image = fabio.open(item)
                was_openned = True
            elif isinstance(item, fabio.fabioimage.FabioImage):
                fabio_image = item
                was_openned = False
            if fabio_image is None:
                self._shape = item.shape[-2:]
            else:
                self._shape = fabio_image.shape
                if was_openned:
                    fabio_image.close()
        return self._shape


class MultiFileWriter(io.Writer):
    """Broadcast writing to differnet files for each frames"""

    def __init__(self, output_path, mode=HDF5Writer.MODE_ERROR):
        super(MultiFileWriter, self).__init__()
        if mode in [HDF5Writer.MODE_OVERWRITE, HDF5Writer.MODE_APPEND]:
            raise ValueError("Mode %s unsupported" % mode)
        self._writer = None
        self._output_path = output_path
        self._mode = mode

    def init(self, fai_cfg=None, lima_cfg=None):
        self._fai_cfg = fai_cfg
        self._lima_cfg = lima_cfg
        self._is_2d = self._fai_cfg.get("do_2D", False) is True

    def prepare_write(self, data_info, engine):
        if data_info.source_filename:
            output_name = os.path.splitext(data_info.source_filename)[0]
        else:
            output_name = f"array_{data_info.data_id}"

        if self._is_2d:
            extension = ".azim"
        else:
            extension = ".dat"

        if data_info.frame_id is not None:
            output_name = f"{output_name}_{data_info.frame_id:04d}"

        output_name = f"{output_name}{extension}"
        if self._output_path:
            if os.path.isdir(self._output_path):
                basename = os.path.basename(output_name)
                outpath = os.path.join(self._output_path, basename)
            else:
                outpath = os.path.abspath(self._output_path)
        else:
            outpath = output_name

        if os.path.exists(outpath):
            if self._mode == HDF5Writer.MODE_DELETE:
                os.unlink(outpath)
        self._writer = DefaultAiWriter(outpath, engine)
        self._writer.init(fai_cfg=self._fai_cfg, lima_cfg=self._lima_cfg)

    def write(self, data):
        self._writer.write(data)
        self._writer.close()
        self._writer = None

    def close(self):
        pass


class Statistics(object):
    """Instrument the application to collect statistics."""

    def __init__(self):
        self._timer = None
        self._first_processing = 0
        self._processing = 0
        self._reading = 0
        self._frames = 0
        self._execution = 0

    def execution_started(self):
        self._start_time = time.perf_counter()

    def execution_finished(self):
        t = time.perf_counter()
        self._execution = t - self._start_time

    @contextlib.contextmanager
    def time_processing(self):
        t1 = time.perf_counter()
        yield
        t2 = time.perf_counter()
        processing = t2 - t1
        if self._processing == 0:
            self._first_processing = processing
        self._processing += processing
        self._frames += 1

    @contextlib.contextmanager
    def time_reading(self):
        t1 = time.perf_counter()
        yield
        t2 = time.perf_counter()
        reading = t2 - t1
        self._reading += reading

    def processing_per_frame(self):
        """Average time spend to process a frame"""
        if self._frames < 2:
            return self._first_processing
        return (self._processing - self._first_processing) / (self._frames - 1)

    def preprocessing(self):
        """Try to extract the preprocessing time"""
        if self._frames < 2:
            return float("NaN")
        return self._first_processing - self.processing_per_frame()

    def reading_per_frame(self):
        """Average time spend to read a frame"""
        if self._frames == 0:
            return float("NaN")
        return self._reading / self._frames

    def total_reading(self):
        return self._reading

    def total_processing(self):
        return self._processing

    def total_execution(self):
        return self._execution


def process(input_data, output, config, observer, write_mode=HDF5Writer.MODE_ERROR, format_=None):
    """
    Integrate a set of data.

    :param List[str] input_data: List of input filenames
    :param str output: Filename of directory output
    :param dict|WorkerConfig config: Dictionary|WorkerConfig to configure `pyFAI.worker.Worker`
    :param IntegrationObserver observer: Observer of the processing
    :param str write_mode: Specify options to deal with IO errors
    :param format_: output format
    """
    statistics = Statistics()
    statistics.execution_started()

    if observer is None:
        # Create a null observer to avoid to deal with None
        observer = IntegrationObserver()

    worker = Worker()
    if isinstance(config, WorkerConfig):
        worker_config = config
    else:
        worker_config = WorkerConfig.from_dict(config)

    monitor_name = worker_config.monitor_name
    worker.set_config(worker_config)
    worker.output = "raw"
    worker.safe = False  # all processing are expected to be the same.

    observer.worker_initialized(worker)

    # Skip invalide data
    source = DataSource(statistics=statistics)
    for item in input_data:
        if isinstance(item, (str,)):
            if os.path.isfile(item):
                source.append(item)
            else:
                if "::" in item:
                    try:
                        # Only check that we can open the file
                        # It's low cost with HDF5
                        with fabio.open(item):
                            pass
                        source.append(item)
                    except Exception:
                        logger.warning("File %s do not exists. File ignored.", item)
                else:
                    logger.warning("File %s do not exists. File ignored.", item)
        elif isinstance(item, fabio.fabioimage.FabioImage):
            source.append(item)
        elif isinstance(item, numpy.ndarray):
            source.append(item)
        else:
            logger.warning("Type %s unsopported. Data ignored.", item)
    print(f"Initialize worker wih shape {source.shape}")
    worker.reconfig(source.shape, sync=True)
    print(worker.ai)
    observer.processing_started(source.approximate_count())

    writer = None
    if output:
        if "::" in output:
            output, entry_path = output.split("::", 1)
        else:
            entry_path = None
        if os.path.isdir(output):
            writer = MultiFileWriter(output, mode=write_mode)
        elif output.endswith(".h5") or output.endswith(".hdf5") or format_ in ("h5", "hdf5"):
            writer = HDF5Writer(output, hpath=entry_path, append_frames=True, mode=write_mode)
        else:
            output_path = os.path.abspath(output)
            writer = MultiFileWriter(output_path, mode=write_mode)
    else:
        if source.is_single_multiframe():
            basename = os.path.splitext(source.basename())[0]
            output_filename = f"{basename}_{'2d' if worker.do_2D() else '1d'}.h5"
            writer = HDF5Writer(output_filename, append_frames=True, mode=write_mode)
        else:
            output_path = os.path.abspath(".")
            writer = MultiFileWriter(None, mode=write_mode)

    try:
        writer.init(fai_cfg=config)
    except IOError as e:
        logger.error("Error while creating the writer: " + str(e.args[0]))
        logger.error("Processing cancelled")
        logger.info("To write HDF5, convenient options can be provided to decide what to do.")
        logger.info("Options: --delete (always delete the file) --append (create a new entry) --overwrite (overwrite this entry)")
        writer.close()
        return 1

    # Integrate all the provided frames one by one
    for data_info in source.frames():
        logger.debug("Processing %s", item)

        observer.processing_data(data_info,
                                 approximate_count=source.approximate_count())

        if data_info.fabio_image is not None:
            normalization_factor = get_monitor_value(data_info.fabio_image, monitor_name)
        else:
            normalization_factor = 1.0

        if hasattr(writer, "prepare_write"):
            writer.prepare_write(data_info, engine=worker.ai)

        with statistics.time_processing():
            result = worker.process(data=data_info.data,
                                    normalization_factor=normalization_factor,
                                    writer=writer)
        # Store reference to input data if possible
        if isinstance(writer, HDF5Writer) and (data_info.fabio_image is not None):
            fimg = data_info.fabio_image
            if "dataset" in dir(fimg):
                if isinstance(fimg.dataset, list):
                    for ds in  fimg.dataset:
                        writer.set_hdf5_input_dataset(ds)
                else:
                    writer.set_hdf5_input_dataset(fimg.dataset)

        if observer.is_interruption_requested():
            break
        observer.data_result(data_info, result)
        if observer.is_interruption_requested():
            break

    writer.close()

    if observer.is_interruption_requested():
        logger.error("Processing was aborted")
        observer.processing_interrupted()
        result = 2
    else:
        observer.processing_succeeded()
        result = 0
    observer.processing_finished()

    statistics.execution_finished()

    logger.info("[First frame] Preprocessing time: %.0fms", statistics.preprocessing() * 1000)
    logger.info("[Per frames] Reading time: %.0fms; Processing time: %.0fms", statistics.reading_per_frame() * 1000, statistics.processing_per_frame() * 1000)
    logger.info("[Total] Reading time: %.3fs; Processing time: %.3fs", statistics.total_reading(), statistics.total_processing())
    logger.info("Execution done in %.3fs !", statistics.total_execution())
    return result


def integrate_shell(options, args):
    wc = WorkerConfig.from_file(options.json)
    observer = ShellIntegrationObserver()
    default_logger = logging.getLogger()
    with logging_utils.prepost_emit_callback(default_logger,
                                             observer.hide_info,
                                             observer.show_info):
        wc.monitor_name = options.monitor_key
        filenames = args
        output = options.output
        result = process(filenames, output, wc, observer, options.write_mode)

    return result


def _main(args):
    """Execute the application

    :param str args: Command line argument without the program name
    :rtype: int
    """
    usage = "pyFAI-integrate [options] file1.edf file2.edf ..."
    version = f"pyFAI-integrate version {pyFAI_version} from {pyFAI_date}"
    description = """
    PyFAI-integrate is a graphical interface (based on Python/Qt4) to perform azimuthal integration
on a set of files. It exposes most of the important options available within pyFAI and allows you
to select a GPU (or an openCL platform) to perform the calculation on."""
    epilog = """PyFAI-integrate saves all parameters in a .azimint.json (hidden) file. This JSON file
is an ascii file which can be edited and used to configure online data analysis using
the LImA plugin of pyFAI.

Nota: there is bug in debian6 making the GUI crash (to be fixed inside pyqt)
http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348"""
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose", default=False,
                        help="switch to verbose/debug mode")
    parser.add_argument('--debug',
                        dest="debug",
                        action="store_true",
                        default=False,
                        help='Set logging system in debug mode')
    parser.add_argument("-o", "--output",
                        dest="output", default=None,
                        help="Directory or file where to store the output data")
    parser.add_argument("-f", "--format",
                        dest="format", default="None",
                        help="output data format (can be used to enforce HDF5 in combination with --output)")
    parser.add_argument("-s", "--slow-motor",
                        dest="slow", default=None,
                        help="Dimension of the scan on the slow direction (makes sense only with HDF5)")
    parser.add_argument("-r", "--fast-motor",
                        dest="rapid", default=None,
                        help="Dimension of the scan on the fast direction (makes sense only with HDF5)")
    parser.add_argument("--no-gui",
                        dest="gui", default=True, action="store_false",
                        help="Process the dataset without showing the user interface.")
    parser.add_argument("-j", "--json",
                        dest="json", default=".azimint.json",
                        help="Configuration file containing the processing to be done")
    parser.add_argument("args", metavar='FILE', type=str, nargs='*',
                        help="Files to be integrated")
    parser.add_argument("--monitor-name", dest="monitor_key", default=None,
                        help="Name of the monitor in the header of each input \
                        files. If defined the contribution of each input file \
                        is divided by the monitor. If the header does not \
                        contain or contains a wrong value, the contribution \
                        of the input file is ignored.\
                        On EDF files, values from 'counter_pos' can be accessed \
                        by using the expected mnemonic. \
                        For example 'counter/bmon'.")
    parser.add_argument("--delete",
                        dest="delete_mode",
                        action="store_true",
                        help="Delete the destination file if already exists")
    parser.add_argument("--append",
                        dest="append_mode",
                        action="store_true",
                        help="Append the processing to the destination file using an available group (HDF5 output)")
    parser.add_argument("--overwrite",
                        dest="overwrite_mode",
                        action="store_true",
                        help="Overwrite the entry of the destination file if it already exists (HDF5 output)")
    options = parser.parse_args(args)

    # Analysis arguments and options
    args = utils.expand_args(options.args)
    args = sorted(args)

    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    write_mode = HDF5Writer.MODE_ERROR
    if options.delete_mode:
        write_mode = HDF5Writer.MODE_DELETE
    elif options.append_mode:
        write_mode = HDF5Writer.MODE_APPEND
    elif options.overwrite_mode:
        write_mode = HDF5Writer.MODE_OVERWRITE
    options.write_mode = write_mode

    if options.gui:
        result = integrate_gui(options, args)
    else:
        result = integrate_shell(options, args)
    return result


def main(args=None):
    result = _main(args)
    sys.exit(result)


if __name__ == "__main__":
    main()
