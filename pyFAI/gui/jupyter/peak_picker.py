import itertools
import ipywidgets as widgets
from IPython.display import display
from ..peak_picker import PeakPicker as _PeakPicker

class PeakPicker(_PeakPicker):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ics = {}
        self.image = self.data
        self.figsize = (15, 8)
        self.dict_annotated = {}
        self.dict_all_points = {}

    def reset_picked(self):
        self.dict_annotated = {}
        self.dict_all_points = {}
    def merge_segments(self):
        work = self.dict_all_points
        for key in work.keys():
            sets = work[key]
            work[key] = [list(itertools.chain.from_iterable(sets))]

    def add_to_selected_ring(self, ring, current_points):
        """
        ring is integer from
        first ring has index 0
        """
        work = self.dict_all_points
        key = ring
        if work.get(key) is None:
            work[key] = [current_points]
        else:
            work[key].append(current_points)

    def get_ring(self, ring):
        """
        maybe delete
        """
        self.current_ring = ring

    def gui_jup(self, log=False, maximize=False, pick=True):
        """
        :param log: show z in log scale
        """

        print('ATTENTION!!!!!!')
        print('you must click add to selected ring after eack pick')
        print('pick with shift and left mouse button')
        print('then you click merge')
        print('then click save')

        self.image = image
        self.figsize = (15, 8)
        self.fig = plt.figure(figsize=self.figsize)
        self.ax1 = self.fig.add_subplot(111)

        ringer = widgets.IntText(description="Continuous", continuous_update=True)

        button_merge = widgets.Button(description='merge rings')
        def merge_segments_on_click(b):
            self.merge_segments()
        button_merge.on_click(merge_segments_on_click)

        button_add = widgets.Button(description='add_to_selected_ring')
        def add_to_ring_on_click(b):
            self.add_to_selected_ring(ringer.value, self.current_points)
        button_add.on_click(add_to_ring_on_click)

        button_reset = widgets.Button(description='reset')
        def reset_picking_on_click(b):
            self.reset_picked()
            #self plot annotated
        button_reset.on_click(reset_picking_on_click)

        text_field_output_name = widgets.Text(
            value='./controlpoint_file.npt',
            #placeholder='Type something',
            description='controlpoint file:',
            disabled=False
        )

        button_save_ctrl_pts = widgets.Button(description='save control point file')
        def save_ctrl_pts_on_click(b):
            filename = text_field_output_name.value
            work = self.dict_all_points
            for key in work.keys():
                self.points.append(work[key][0], key)
            self.points.save(filename)
        button_save_ctrl_pts.on_click(save_ctrl_pts_on_click)

        _ = display(widgets.HBox([ringer, button_add, button_merge, button_reset, text_field_output_name, button_save_ctrl_pts]))

        self.im = self.ax1.imshow(self.image)#, cmap='magma')
        self.ax1.images[0].set_picker(True)
        self.cid = self.fig.canvas.mpl_connect("pick_event", self.pick)

    def pick(self, event):
        e = event.mouseevent
        #if ((event.button == 3) and (event.key == 'shift')) or \
        #((event.button == 1) and (event.key == 'v')):
        #k = event.keyevent
        #self.ax1.set_cmap('viridis')
        if (e.inaxes and e.button == 1 and e.key == 'shift'):
            x, y = int(e.xdata), int(e.ydata)
            key = (x, y)
            if key in self.dict_annotated:
                annot = self.dict_annotated[key]
                annot.set_visible(False)
                annot.remove()
                del self.dict_annotated[key]
            else:
                pt0x, pt0y = key
                """data"""
                annot = self.ax1.annotate("", xy=(1, 1), xytext=(-1, 1),
                                     textcoords="offset points",
                                     arrowprops=dict(arrowstyle="->", color="w",
                                                     connectionstyle="arc3"),
                                     va="bottom", ha="left", fontsize=10,
                                     bbox=dict(boxstyle="round", fc="r"))
                annot.set_visible(False)
                self.dict_annotated[key] = annot
            self.annotate(key)
            #" * new_grp Right-click (click+n):         try an auto find for a ring"
            # ydata is a float, and matplotlib display pixels centered.
            # we use floor (int cast) instead of round to avoid use of
            # banker's rounding
            ypix, xpix = int(e.ydata + 0.5), int(e.xdata + 0.5)
            #self._herep = (ypix, xpix, 'test')
            try:
                #    def find_peaks(self, x, nmax=200, annotate=None, massif_contour=None, stdout=sys.stdout):
                points = self.massif.find_peaks([ypix, xpix],
                                            self.defaultNbPoints,
                                            None, None)
                self.current_points = points
            except Exception as e:
                self._herep = []
            self.annotate_found(self._herep)

    def annotate_found(self, points):
        for X in points:
            key = [int(x) for x in X]

            y, x = key
            #int(np.round(e.xdata)), int(np.round(e.ydata))

            #t0x, pt0y = key
            """data"""

            annot = self.ax1.annotate("", xy=(x, y), xytext=(-1, 1),
                                 textcoords="offset points",
                                 arrowprops=dict(arrowstyle="->", color="w",
                                                 connectionstyle="arc3"),
                                 va="bottom", ha="left", fontsize=10,
                                 bbox=dict(boxstyle="round", fc="r"))
            b3 = widgets.Button(description='button 3')
            annot.set_text('•')
            annot.xy = (x, y)
            annot.set_visible(True)
            self.dict_annotated[key] = annot

            self.fig.canvas.draw_idle()

    def annotate(self, X):
        key = X
        if key in self.dict_annotated:
            annot = self.dict_annotated.get(key)
            annot.set_visible(True)
            annot.set_text('•')
            annot.xy = key
        print(self.dict_annotated)

        self.fig.canvas.draw_idle()
