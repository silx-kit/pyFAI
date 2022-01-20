# ensure %matplotlib widget
import time
import numpy
from ..mpl_calib import MplCalibWidget
from matplotlib import pyplot
from matplotlib.colors import LogNorm
# from ...geometryRefinement import GeometryRefinement
from ..cli_calibration import AbstractCalibration
from ..peak_picker import PeakPicker
try:
    from IPython.core.display import display
    import ipywidgets as widgets
except:
    from ...utils.callback import dangling_callback as display


class JupyCalibWidget(MplCalibWidget):

    def __init__(self, *args, **kwargs):
        MplCalibWidget.__init__(self, *args, **kwargs)
        self.mpl_connectId = None

    def init(self, pick=True):
        if self.fig is None:
            self.mplw = widgets.Output()
            with self.mplw:
                with pyplot.ioff():
                    self.fig, (self.ax, self.axc) = pyplot.subplots(1, 2, gridspec_kw={"width_ratios": (10, 1)})
                    self.fig.canvas.toolbar_position = 'bottom'
                    empty = numpy.zeros((100, 100))
                    self.ax.imshow(empty)
                    self.ax.set_ylabel('y in pixels')
                    self.ax.set_xlabel('x in pixels')

                if pick:
                    self.spinbox = widgets.IntText(min=0, max=100, step=1, description='Ring#',)
                    self.finishw = widgets.Button(description="Refine", tooltip='switch to refinement mode')
                    tb2 = widgets.HBox([self.finishw, self.spinbox])
                    self.widget = widgets.VBox([self.mplw, tb2])
                    self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
                    self.finishw.on_click(self.onclick_refine)
            if not pick:
                self.widget = self.mplw

    def update(self):
        pass

    def show(self):
        if self.widget is not None:
            # self.fig.show()
            display(self.widget)
            with self.mplw:
                self.fig.canvas.draw()
                pyplot.pause(0.01)

    def maximize(self, update=True):
        if self.fig:
            if update:
                self.update()

    def get_ring_value(self):
        "Return the value of the ring widget"
        return self.spinbox.value

    def onclick_refine(self, *args):
        """
        callback function
        """
        self.finish()
        self.cb_refine(*args)


class JupyCalibration(AbstractCalibration):

    def __init__(self, img, mask=None,
                 detector=None, wavelength=None, calibrant=None):
        """Simplified interface for calibrating in Jupyter-lab
        :param img: 2D image with Debye-Scherrer rings
        :param mask: 2D image with marked invalid pixels
        :param detector: instance of detector used
        :param wavelengh: wavelength in A as a float
        :param calibrant: instance of calibrant 
        """
        AbstractCalibration.__init__(self, detector=detector,
                                     wavelength=wavelength,
                                     calibrant=calibrant)
        self.img = img
        self.mask = mask
        self.gaussianWidth = None
        self.preprocess()
        self.interactive = False
        self.fixed = ["wavelength", "rot3"]
        self.peakPicker.cb_refine = self.set_data

    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        self.peakPicker = PeakPicker(self.img, reconst=self.reconstruct, mask=self.mask,
                                     pointfile=self.pointfile, calibrant=self.calibrant,
                                     wavelength=self.ai.wavelength, detector=self.detector)

        if self.gaussianWidth is not None:
            self.peakPicker.massif.valley_size = self.gaussianWidth
        else:
            self.peakPicker.massif.init_valley_size()
        if self.gui:
            self.peakPicker.gui(log=True, maximize=False, pick=True,
                                widget_klass=JupyCalibWidget)


Calibration = JupyCalibration
