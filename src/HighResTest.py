"""
Created on Mon Nov 14 11:38:41 2019
This is for looking at High resolution neural recording image.
@author: xinweiy
"""
import matplotlib.pyplot as plt
import numpy as np


class Multi_Slice_Viewer(object):
    def __init__(self, volume, points=None, part=False, fig_idx=1):
        super(Multi_Slice_Viewer, self).__init__()
        self.remove_keymap_conflicts({'j', 'k'})
        self.points = points
        fig = plt.figure(fig_idx)
        self.ax = fig.add_subplot(111)
        #fig, ax = plt.subplots()
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2
        self.ax.imshow(volume[self.ax.index])
        self.ax.texts = []
        self.ax.set_title('z= '+str(self.ax.index))
        self.im_neurons_plane, = self.ax.plot([], [], '+', markersize=3, c='red')

        if self.points is not None:
            pt_idx = np.where(self.points[:, 2] == self.ax.index)[0]
            print(pt_idx)
            for idx in pt_idx:
                self.ax.text(self.points[idx, 1], self.points[idx, 0], str(idx), fontsize=10, color='black')
            self.im_neurons_plane.set_xdata(self.points[pt_idx, 1])
            self.im_neurons_plane.set_ydata(self.points[pt_idx, 0])


        fig.canvas.mpl_connect('key_press_event', self.process_key)
        fig.canvas.mpl_connect('scroll_event', self.onscroll)
        if not part:
            plt.show()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()

    def onscroll(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.button == 'up':
            self.next_slice(ax)
        else:
            self.previous_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        ax.set_title('z= '+str(ax.index))
        ax.texts = []
        if self.points is not None:
            pt_idx = np.where(self.points[:, 2] == ax.index)[0]

            for idx in pt_idx:
                ax.text(self.points[idx, 1], self.points[idx, 0], str(idx), fontsize=10, color='black')

            self.im_neurons_plane.set_xdata(self.points[pt_idx, 1])
            self.im_neurons_plane.set_ydata(self.points[pt_idx, 0])

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        ax.set_title('z= '+str(ax.index))
        ax.texts = []
        if self.points is not None:
            pt_idx = np.where(self.points[:, 2] == ax.index)[0]

            for idx in pt_idx:
                ax.text(self.points[idx, 1], self.points[idx, 0], str(idx), fontsize=10, color='black')

            self.im_neurons_plane.set_xdata(self.points[pt_idx, 1])
            self.im_neurons_plane.set_ydata(self.points[pt_idx, 0])

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

