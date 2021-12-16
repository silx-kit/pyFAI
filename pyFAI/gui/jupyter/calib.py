#ensure %matplotlib widget

from ..mpl_calib import MplCalibWidget
from matplotlib import pyplot
from matplotlib.colors import LogNorm
import ipywidgets as widgets


class JupyCalibWidget(MplCalibWidget):
    def init(self, pick=True, update=True, image=None, bounds=None):
        if self.fig is None:
            self.mplw = widgets.Output()
            with self.mplw:
                with pyplot.ioff():
                    self.fig, (self.ax, self.axc) = pyplot.subplots(1, 2, gridspec_kw={"width_ratios": (10,1)})
                    self.fig.canvas.toolbar_position = 'bottom'
                    self.ax.set_ylabel('y in pixels')
                    self.ax.set_xlabel('x in pixels')
                    if image is not None:
                        self.imshow(image, bounds=bounds, log=True, update=False)
                    display(self.fig.canvas)
                    
            if pick:
                self.spinbox = widgets.IntText(min=0, max=100, step=1, description='Ring#',)
                self.finishw = widgets.Button(description="Refine", tooltip='switch to refinement mode')
                tb2 = widgets.HBox([self.finishw, self.spinbox])
                self.widget = widgets.VBox([self.mplw, tb2])
                self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            else:
                self.widget = self.mplw            
            
            display(self.widget)

        elif update:
            self.update()
            
    def update(self):
        if self.fig:
            with pyplot.ioff():
                self.fig.canvas.draw()
                pyplot.pause(.001)

    def onclick(self, *args):
        """
        Called when a mouse is clicked
        """
        print(args)
        if self.click_cb:
            self.click_cb(*args)
