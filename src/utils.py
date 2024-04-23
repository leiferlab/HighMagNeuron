import numpy as np
from HighResTest import Multi_Slice_Viewer
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.widgets import PolygonSelector

def smooth_2d_pts(pts_ori, degree=2, win_length=20, num_point_out=100):

    # use cum_dis as x,
    dis_list = [0]
    pts_diff = np.diff(pts_ori, axis=0)
    pts_diff_dis = np.sqrt(np.sum(pts_diff ** 2, axis=1))
    dis_list += list(np.cumsum(pts_diff_dis))

    x = np.array(dis_list)
    line_length = dis_list[-1]
    line_length_step = line_length / (num_point_out - 1)

    x_new = np.arange(num_point_out) / (num_point_out - 1) * line_length
    f1 = interp1d(x, pts_ori[:, 0], fill_value='extrapolate')
    f2 = interp1d(x, pts_ori[:, 1], fill_value='extrapolate')
    pts = np.zeros((num_point_out, 2))
    pts[:, 0] = f1(x_new)
    pts[:, 1] = f2(x_new)

    pts_out = np.copy(pts)

    win = max(3, int(win_length / line_length_step))
    win = min(win, num_point_out // 3)
    win = win + 1 if not (win // 2 == 0) else win
    # pts[:, 0] = savgol_filter(pts[:, 0], window_length=win, polyorder=degree)
    # pts[:, 1] = savgol_filter(pts[:, 1], window_length=win, polyorder=degree)

    num_ext = 2 * win
    num_ext = min(num_ext, int(pts.shape[0] // 3))
    y1 = extend_ends(pts[:, 0], num_ext)

    y2 = extend_ends(pts[:, 1], num_ext)
    y1_s = gaussian_filter1d(y1, sigma=win, mode='nearest')
    pts_out[:, 0] = extract_center(y1_s, num_ext)
    y2_s = gaussian_filter1d(y2, sigma=win, mode='nearest')
    pts_out[:, 1] = extract_center(y2_s, num_ext)
    # plt.scatter(pts_out[:, 0], pts_out[:, 1], c='green')
    # plt.scatter(pts[:, 0], pts[:, 1], c='red')
    # plt.show()

    return pts_out

def extract_center(y, num_ext):
    num_pt = len(y)
    return y[num_ext:len(y)-num_ext]

def extend_ends(y, num_ext):
    # y is array
    if num_ext > 0:
        dir_head = (y[num_ext] - y[0]) / (num_ext + 1e-3)
        first_list = [i * dir_head + y[0] for i in range(-num_ext, 0)]
        dir_tail = (y[-1] - y[-num_ext-1]) / (num_ext + 1e-3)
        last_list = [y[-1] + i * dir_tail for i in range(1, num_ext + 1)]

        y_out = first_list + list(y) + last_list
    else:
        y_out = y
    return np.array(y_out)

def get_curve_representation(pts_ori, degree=2, num_point_out=100, extend=False):
    # fit the line to to a preset form.


    if pts_ori.shape[0] < num_point_out:
        # use cum_dis as x,
        dis_list = [0]
        pts_diff = np.diff(pts_ori, axis=0)
        pts_diff_dis = np.sqrt(np.sum(pts_diff ** 2, axis=1))
        dis_list += list(np.cumsum(pts_diff_dis))

        x = np.array(dis_list)
        x_new = np.arange(num_point_out) / (num_point_out - 1) * x.max()
        f1 = interp1d(x, pts_ori[:, 0])
        f2 = interp1d(x, pts_ori[:, 1])
        pts = np.zeros((num_point_out, 2))
        pts[:, 0] = f1(x_new)
        pts[:, 1] = f2(x_new)
    else:
        pts = pts_ori


    num_point = num_point_out

    pts_new = np.zeros((num_point, 2))


    x_new = np.arange((num_point)) / (num_point - 1)

    # set some buffer on the two ends.
    num_pts = pts.shape[0]
    if extend:
        num_ext = num_pts // 10
    else:
        num_ext = 0

    x = np.arange(-num_ext, num_pts + num_ext) / (num_pts - 1)
    y1 = pts[:, 0]
    y1 = extend_ends(y1, num_ext)

    y2 = pts[:, 1]
    y2 = extend_ends(y2, num_ext)


    z = np.polyfit(x, y1, degree)
    fn = np.poly1d(z)
    pts_new[:, 0] = fn(x_new)


    z = np.polyfit(x, y2, degree)
    fn = np.poly1d(z)
    pts_new[:, 1] = fn(x_new)
    #pt[:, 1] = np.array(self.Y)
    # if num_point_out != num_point:
    #     pt = self.distanceInterp(pt, num_point_out)

    # plt.scatter(pts_new[:, 0], pts_new[:, 1], c='green')
    # plt.scatter(pts[:, 0], pts[:, 1], c='red')
    # plt.scatter(y1, y2, c='black', marker='+')
    # plt.scatter(pts[:, 0], pts[:, 1], c='yellow', marker='+')
    # plt.show()
    return pts_new

class line_quadratic(Multi_Slice_Viewer):
    # get the coordinates of a line and fit it into quadratic form.
    def __init__(self, volume, points=None):
        super(line_quadratic, self).__init__(volume, points, part=True)

        self.X = []
        self.Y = []

        # Print instructions for the user
        print("\nINSTRUCTIONS TO DRAW THE LINE\n\n" + \
              "Click to add a point. Right-click to delete the last one. \n" + \
              "***Add one point outside the image in the posterior direction." + \
              "***\n" + \
              "Ignore the first point in the corner of the image. \n" + \
              "Once you're done clicking, simply close the figure window.")

        # Create the figure and plot the image

        self.line, = self.ax.plot(0, 0, '*k')

        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
        plt.show()

    # Define callback function to be called when there is a mouse click
    def smooth_cline(self, degree=2, num_point_out=100):
        pts = np.array((self.X, self.Y)).T
        pts = get_curve_representation(pts, degree=degree, num_point_out=num_point_out)
        self.X = list(pts[:, 0])
        self.Y = list(pts[:, 1])

        return pts

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()
        self.line.set_data(self.X, self.Y)
        self.line.figure.canvas.draw()

    def __call__(self, event):
        ix, iy = event.xdata, event.ydata
        # If left click, add point, otherwise delete the last one.
        if (event.button == 1):
            self.X.append(ix)
            self.Y.append(iy)
        else:
            self.X.pop(-1)
            self.Y.pop(-1)

        self.line.set_data(self.X, self.Y)
        self.line.figure.canvas.draw()

    def getLine(self):
        # Convert to np.array and return
        return np.array([self.X, self.Y]).T  # [::-1]


class polygon(Multi_Slice_Viewer):
    def __init__(self, volume, points=None):
        super(polygon, self).__init__(volume, points, part=True)
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.verts = []

        # Print instructions for the user
        print("\nINSTRUCTIONS TO DRAW THE POLYGON\n\n" + \
              "Select points in the figure by enclosing them within a polygon.\n" +
              "Press the 'esc' key to start a new polygon.\n" +
              "Try holding the 'shift' key to move all of the vertices.\n" +
              "Try holding the 'ctrl' key to move a single vertex.")

        lineprops = dict(color='r', linestyle='-', linewidth=2, alpha=0.5)
        markerprops = dict(marker='o', markersize=7, mec='r', mfc='r',
                           alpha=0.5)

        self.ps = PolygonSelector(self.ax, self.onselect, lineprops=lineprops,
                                  markerprops=markerprops)

        plt.show()

    def onselect(self, verts):
        self.verts = verts[:]

    def getPolygon(self):
        return self.verts

# Define callback function to be called when there is a mouse click

