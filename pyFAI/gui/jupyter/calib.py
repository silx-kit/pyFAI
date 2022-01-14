# ensure %matplotlib widget
import numpy
from ..mpl_calib import MplCalibWidget
from matplotlib import pyplot
from matplotlib.colors import LogNorm

try:
    from IPython.core.display import display
    import ipywidgets as widgets
except:
    from ...utils.callback import dangling_callback as display


class JupyCalibWidget(MplCalibWidget):

    def __init__(self, *args, **kwargs):
        MplCalibWidget.__init__(self, *args, **kwargs)
        self.events = []
        self.mpl_connectId = None

    def init(self, pick=True):
        if self.fig is None:
            self.mplw = widgets.Output()
            with self.mplw:
                with pyplot.ioff():
                    self.fig, (self.ax, self.axc) = pyplot.subplots(1, 2, gridspec_kw={"width_ratios": (10, 1)})
                    self.fig.canvas.toolbar_position = 'bottom'
                    self.ax.set_ylabel('y in pixels')
                    self.ax.set_xlabel('x in pixels')
                    empty = numpy.zeros((100, 100))
                    self.ax.imshow(empty)
            if pick:
                self.spinbox = widgets.IntText(min=0, max=100, step=1, description='Ring#',)
                self.finishw = widgets.Button(description="Refine", tooltip='switch to refinement mode')
                tb2 = widgets.HBox([self.finishw, self.spinbox])
                self.widget = widgets.VBox([self.mplw, tb2])
                self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            else:
                self.widget = self.mplw

    def update(self):
        pass

    def show(self):
        if self.widget is not None:
            display(self.widget)
            with self.mplw:
                self.fig.canvas.draw()
                pyplot.pause(0.01)

    def onclick(self, *args):
        """
        Called when a mouse is clicked
        """
        self.events.append(args)
        print(args)
        if self.click_cb:
            self.click_cb(*args)

    def maximize(self, update=True):
        if self.fig:
            if update:
                self.update()

    def get_ring_value(self):
        "Return the value of the ring widget"
        return self.spinbox.value()
