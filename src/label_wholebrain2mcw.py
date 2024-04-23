#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:31:51 2019
This script is to generate the pictures used to label the neurons in MCW.
The images of neurons are shown.(Template and Worm to be assigned in different window)
Human need to type in the label for the worm to be labelled.
@author: xinweiy
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from PyQt5.QtCore import pyqtRemoveInputHook
from mpl_toolkits.mplot3d import Axes3D
from mistofrutta.plt.hyperstacks import IndexTracker
from mistofrutta.geometry.rotations import rotate_3D_image
import pickle
import os
import argparse
from HighResFlow import worm_data_loader
import scipy.io as sio
import cv2


def plot_box(ax, x, y, z):
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.2 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.2 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


class Label_Neuron(IndexTracker):
    # the class used to label neurons in a worm.
    # this class shows figures of neuron positions and colors.
    # human type in label of neurons by comparing template and the worm.
    def __init__(self, worm, order="z", cmap='', colors=['blue', 'cyan', 'green', 'orange', 'red', 'magenta'], \
                 Overlay=[], user='manual'):
        # Initialize the figure for showing one worm
        # worm is the istance for worm data.
        # A is the image of worms
        # order is the order of channel and z in the array.
        # color is the color used for each channel.
        # overlay
        self.has_been_closed = False
        self.user = user
        # switch the z and channel if necessary
        self.worm = worm
        # show multiple in same image with rgb color.
        self.show_multiple_channel = True
        # the mode of figure
        # mode 1, z stack images of 3 channels in RGB.
        # mode 2, z max projection of 3 channels in RGB.
        # mode 3, z stack images of all individual channel.
        self.figure_mode = 0
        # show the label of already matched neurons.
        self.show_label = 2
        self.show_focus_ind = True
        # A hidden mode: to modify the colormap of current channel intensity.
        self.modify_colormap = False
        self.scale_blue = 1
        self.scale_green = 1
        self.scale_red = 1
        self.ch = 0
        self.cb = None
        # the string of instruction for user interaction.
        self.instruction = 'ctrl+p: enter point selection mode. ' + \
                           'alt+z: change the mode of figure(z stack, z projection)\n' + \
                           'alt+o: change the mode of showing the label. ' + \
                           'ctrl+i: turn on/off the index of focus\n' + \
                           'ctrl+g: save current matching. ' + \
                           'Scroll/up-down: navigate in z\n' + \
                           'ctrl+v: go to specific volume ' + \
                           'ctrl+r: go to rotate current image ' + \
                           'ctrl+l: go to flip the image\n' + \
                           'ctrl+k: switch z dir ' + \
                           'A-D/left-right: change volume(take some seconds). ' + \
                           'W-S: change color scaling'
        self.instruction2 = 'anterior-posterior axis is along x, worm is already straightened.\n' + \
                            'To change the channel(color) go to the other figure and use key a or d\n'

        # A = worm.img_norm

        self.num_v = self.worm.pSNew.shape[1]
        self.vidx = min(100, self.num_v) # set the default volume index to 100.
        self.do_flip = False

        # load image
        # load specific volume
        #rec = self.worm.worm_loader.get_frame_aligned(self.vidx, channel='red')
        # # load image
        # A = rec['image']


        # Initialize the figure to show worms.
        self.fig = plt.figure(1, figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('button_press_event', self.onbuttonpress)
        self.fig.canvas.mpl_connect('close_event', self.onclose)

        # print(self.xlim, self.ylim, self.zlim)
        # self.plot_neuron_3d()
        # self.z = 15
        # neurons_on_plane = self.find_neuron_around_focus()
        # self.im_neurons_on_plane, = self.ax.plot(neurons_on_plane[:,1],neurons_on_plane[:,0],'o',markersize=1,c='w')
        self.im_neurons_above_plane, = self.ax.plot([], [], '|', markersize=3, c='red')
        self.im_neurons_below_plane, = self.ax.plot([], [], '_', markersize=3, c='red')


        # load image
        # load specific volume
        # here we load unaligned image
        rec = self.worm.worm_loader.load_frame(self.vidx)
        size_2 = rec['volume'].shape[2] // 2
        A = rec['volume'][:, :, :size_2]
        A[A < 0] = 0

        # set current points and labels
        self.worm.load_worm_coord(self.vidx)
        if order == "zc": A = np.swapaxes(A, 0, 1)
        #rec = self.worm.worm_loader.get_frame_aligned(self.vidx, channel='red')
        self.sign_pos = False
        if rec['z'][-1] - rec['z'][0] > 0:
            self.z_flip = self.sign_pos
        else:
            self.z_flip = not self.sign_pos


        # set current points and labels
        self.worm.load_worm_coord(self.vidx)


        if self.z_flip:
            A = np.flip(A, axis=0)
            self.worm.X[:, 2] = A.shape[0] - 1 - self.worm.X[:, 2]

        super().__init__(self.ax, A, cmap=cmap, colors=colors, Overlay=Overlay)
        self.X_max = np.max(self.X, axis=0)
        # For the hidden colormap modification mode
        self.X_backup = np.copy(self.X)
        self.X_max_backup = np.copy(self.X_max)


    def image_update(self, do_load=False, do_rotate=False, do_flip=False, angle=0):
        # load image
        # load specific volume
        if do_load:
            rec = self.worm.worm_loader.load_frame(self.vidx)
            size_2 = rec['volume'].shape[2] // 2
            A = rec['volume'][:, :, :size_2]
            self.dimensions = A.shape
            self.slices = self.dimensions[-3]
            if rec['z'][-1] - rec['z'][0] > 0:
                self.z_flip = self.sign_pos
            else:
                self.z_flip = not self.sign_pos

            A[A < 0] = 0
            self.X_backup = np.copy(A)
            self.X = A
            # set current points and labels
            self.worm.load_worm_coord(self.vidx)
            if self.z_flip:
                self.X = np.flip(self.X, axis=0)
                self.worm.X[:, 2] = self.X.shape[0] - 1 - self.worm.X[:, 2]

        if do_flip:
            # flip the image
            self.X = np.flip(self.X, axis=(0, 1))
            self.worm.X[:, 0] = self.X.shape[1] - 1 - self.worm.X[:, 0]
            self.worm.X[:, 2] = self.X.shape[0] - 1 - self.worm.X[:, 2]

        if do_rotate and angle != 0:
            # rotate the image
            # rotate the image
            row, col = self.X.shape[1:]
            # center = tuple(np.array([row, col]) // 2)
            center = tuple(np.array([col, row]) // 2)

            A_rotate = []
            for img in self.X:
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

                new_img = cv2.warpAffine(img, rot_mat, (col, row))

                A_rotate.append(new_img)

            # do it for every volume
            A_rotate = np.array(A_rotate)
            self.X = A_rotate

            x = np.copy(self.worm.X[:, 0]) - center[1]
            y = np.copy(self.worm.X[:, 1]) - center[0]
            theta = angle / 180. * np.pi
            x_r = np.cos(theta) * x - np.sin(theta) * y
            y_r = np.sin(theta) * x + np.cos(theta) * y
            self.worm.X[:, 0] = x_r + center[0]
            self.worm.X[:, 1] = y_r + center[1]



        if do_flip or do_load or do_rotate:
            self.X_max = np.max(self.X, axis=0)
            # For the hidden colormap modification mode
            self.X_backup = np.copy(self.X)
            self.X_max_backup = np.copy(self.X_max)

    def onbuttonpress(self, event):
        if self.selectpointsmode:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                # find the segmented neuron that cloest to the select point.
                # use the l1 distance is fast.
                if self.parent_term is not None:
                    title = self.fig.canvas.get_window_title()
                    self.fig.canvas.set_window_title(title+"*")
                    os.system("wmctrl -a "+self.parent_term)


                dist = np.abs(self.worm.X[:, 0] - iy) + np.abs(self.worm.X[:, 1] - ix) + np.abs(
                    self.worm.X[:, 2] - self.z)
                # get the cloest neuron
                neu_idx = np.argmin(dist)
                print('distance to cloest point:{}'.format(dist[neu_idx]))
                print('selected track index:{}'.format(self.worm.trackIdx[neu_idx]))
                label = input("Enter label for (" + str(int(ix)) + "," + str(int(iy)) + "):  ")


                if dist[neu_idx] < 8:
                    if label == '\d':
                        print('delete label:{} at {}'.format(self.worm.label_human[self.worm.trackIdx[neu_idx]], \
                                                             self.worm.X[neu_idx, :]))
                        self.worm.labeledPoints[self.worm.label_human[self.worm.trackIdx[neu_idx]]].pop()
                        self.worm.label_human.pop(self.worm.trackIdx[neu_idx])
                    else:
                        print('add label {} to neuron at {}'.format(label, self.worm.X[neu_idx, :]))
                        if np.isnan(self.worm.trackIdx[neu_idx]):
                            print('Warning, this neuron is not tracked')
                        else:
                            self.worm.label_human[self.worm.trackIdx[neu_idx]] = label
                            self.worm.labeledPoints[label] = [self.vidx, self.worm.X_ori[neu_idx, 0], self.worm.X_ori[neu_idx, 1], self.worm.X_ori[neu_idx, 2]]
                    if self.parent_term is not None:
                        os.system("wmctrl -a "+title+"*")
                        self.fig.canvas.set_window_title(title)

                else:
                    print('No segmented neuron found in the neighbor')



            self.update()
    def onclose(self, event):
        self.worm.save_match()
        print('match saved')
        self.has_been_closed = True


    def onkeypress(self, event):
        # handle the event of a key press.


        if not self.selectpointsmode:
            if event.key == 'd' or event.key == 'right':
                self.vidx = (self.vidx + 1) % self.num_v
                self.image_update(do_load=True, do_rotate=False, do_flip=self.do_flip)

            elif event.key == 'a' or event.key == 'left':
                self.vidx = (self.vidx - 1) % self.num_v
                self.image_update(do_load=True, do_rotate=False, do_flip=self.do_flip)
            elif event.key == 'alt+z':
                # change mode of figure.
                # total number of mode is 3.
                self.figure_mode = np.mod(self.figure_mode + 1, 2)

            elif event.key == 'ctrl+l':
                # change mode of figure.
                # total number of mode is 3.
                self.do_flip = not self.do_flip
                self.image_update(do_load=False, do_rotate=False, do_flip=True)
            elif event.key == 'ctrl+k':
                # change mode of figure.
                # total number of mode is 3.
                self.sign_pos = not self.sign_pos
                self.image_update(do_load=True, do_rotate=False, do_flip=self.do_flip)

            elif event.key == 'ctrl+r':
                # rotate the image
                # lab = input("Label for pt " + str(cl) + " (blank to keep " + self.manual_labels[cl[0]][cl[1]] + "): ")
                # if lab != "": self.manual_labels[cl[0]][cl[1]] = lab
                angle = input("the angle to rotate(-180 ~ 180):")
                angle = float(angle)
                self.image_update(do_load=False, do_rotate=True, do_flip=False, angle=angle)
            elif event.key == 'ctrl+v':
                # go to specific volume
                v = int(input("the volume to go:"))
                self.vidx = v % self.num_v
                self.image_update(do_load=True, do_rotate=False, do_flip=self.do_flip)


            elif event.key == 'alt+o':
                self.show_label = (self.show_label + 1) % 3

            elif event.key == 'ctrl+i':
                self.show_focus_ind = not self.show_focus_ind

            elif event.key == 'ctrl+g':
                # save the current matching
                self.worm.save_match()
                print('match saved')

            elif event.key == 'ctrl+c':
                # colormap change mode.
                self.modify_colormap = not self.modify_colormap



        # The hidden mode. to adjust the colormap of rgb picture.
        if self.modify_colormap:
            if event.key == '1':
                # reduce the blue channel brightness.
                self.scale_blue *= 0.8

            elif event.key == '4':
                # reduce the blue channel brightness.
                self.scale_blue *= 1.2

            elif event.key == '5':
                # reduce the blue channel brightness.
                self.scale_green *= 1.2

            elif event.key == '2':
                # reduce the blue channel brightness.
                self.scale_green *= 0.8

            elif event.key == '3':
                # reduce the blue channel brightness.
                self.scale_red *= 0.8

            elif event.key == '6':
                # reduce the blue channel brightness.
                self.scale_red *= 1.2

            self.X = np.copy(self.X_backup)
            self.X[0] = self.scale_blue * self.X[0]
            self.X[1] = self.scale_green * self.X[1]
            self.X[3] = self.scale_red * self.X[3]
            self.X_max = np.copy(self.X_max_backup)
            self.X_max[0] = self.scale_blue * self.X_max[0]
            self.X_max[1] = self.scale_green * self.X_max[1]
            self.X_max[3] = self.scale_red * self.X_max[3]
            self.X[self.X > 1] = 1
            self.X_max[self.X_max > 1] = 1
            print('scale red:', str(self.scale_red))
            print('scale blue:', str(self.scale_blue))
            print('scale green:', str(self.scale_green))
        super().onkeypress(event)

    def plot_neuron_3d(self):
        if self.cb is not None:
            self.cb.remove()
        self.fig2.clear()
        self.ax2 = Axes3D(self.fig2)
        # self.ax2.clear()

        # print(self.worm.X_brt.shape)
        # sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=var_list, cmap='Reds', alpha=0.75, marker='o')
        # plt.colorbar(sc, ax=ax)
        sc = self.ax2.scatter(self.worm.X_s[:, 0], self.worm.X_s[:, 1], self.worm.X_s[:, 2],
                              c=self.worm.X_brt[:, self.ch], cmap='Reds', s=100, alpha=0.9, marker='o')
        # if self.cb is None:
        self.cb = plt.colorbar(sc, ax=self.ax2)
        if self.show_label:
            for i in range(len(self.worm.X)):
                if self.worm.label_human[i] != ' ':
                    self.ax2.text(self.worm.X_s[i, 0], self.worm.X_s[i, 1], self.worm.X_s[i, 2],
                                  str(self.worm.label_human[i]), fontsize=8, color='black')
                elif self.show_label == 2:
                    self.ax2.text(self.worm.X_s[i, 0], self.worm.X_s[i, 1], self.worm.X_s[i, 2],
                                  str(self.worm.label[i]), fontsize=8, color='black')

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_zlabel('z')
        self.ax2.set_title(self.instruction2 + 'current channel: {}'.format(self.ch))
        self.ax2.auto_scale_xyz(self.xlim, self.ylim, self.zlim)
        self.fig2.canvas.draw()
        # self.ax2.set_aspect("equal")
        # plot_box(self.ax2, self.xlim, self.ylim, self.zlim)

    def update(self):
        # update the plot of worms according to user input.
        # clear the texts
        self.ax.texts = []

        if self.figure_mode == 0:
            # mode 0, z stack images of 3 channels in RGB.
            #      img = np.copy(self.X[:, self.z, ...])
            #      img = np.moveaxis(img,[0,1,2],[2,0,1])
            #      self.im.set_data(img[:,:,[3,1,0]])
            self.im.set_data(self.X[self.z, ...])
            # self.im.set_cmap('')
            self.ax.set_xlabel('slice :{}, volume index :{}'.format(str(self.z), self.vidx))
            self.im.set_cmap('viridis')
            # show the mode in title
            title_mode = 'Mode 1: single plane of image\n'

        elif self.figure_mode == 1:
            # mode 1, z max projection of 3 channels in RGB.
            #      img = np.max(self.X, axis=1)
            #      img = np.moveaxis(img,[0,1,2],[2,0,1])
            #      self.im.set_data(img[:,:,[3,1,0]])
            self.im.set_data(self.X_max)
            # self.im.set_cmap('')
            #self.ax.set_xlabel('slice ' + str(self.z))
            self.ax.set_xlabel('volume index :{}'.format(self.vidx))
            self.im.set_cmap('viridis')
            # show the mode in title
            title_mode = 'Mode 2: z max projection.\n'


        # plot the label of already matched neurons.
        if self.show_label:
            # get the neurons that are on the current focus plane.
            if self.figure_mode == 0:
                neurons_on_plane, neurons_on_plane_idx = self.find_neuron_around_focus(plane_depth=0,
                                                                                       method='around')
            else:
                neurons_on_plane_idx = np.arange(len(self.worm.X))

            # text the label in the image
            if len(neurons_on_plane_idx):
                for i in np.nditer(neurons_on_plane_idx):
                    # show the current label
                    if self.worm.trackIdx[i] in self.worm.label_human:
                        self.ax.text(self.worm.X[i, 1] - 2, self.worm.X[i, 0] - 2, self.worm.label_human[self.worm.trackIdx[i]],
                                     fontsize=8, color='red')
                    elif self.show_label == 2:
                        # also show label for unlabelled tracked neuron.
                        if not np.isnan(self.worm.trackIdx[i]):
                            self.ax.text(self.worm.X[i, 1] - 2, self.worm.X[i, 0] - 2, int(self.worm.trackIdx[i]),
                                         fontsize=8, color='white')

        # find the neuron on focus
        if self.show_focus_ind and self.figure_mode == 0:
            neurons_above_plane, _ = self.find_neuron_around_focus(plane_depth=1, method='above')
            neurons_below_plane, _ = self.find_neuron_around_focus(plane_depth=1, method='below')
        else:
            neurons_above_plane = np.empty((1, 2))
            neurons_below_plane = np.empty((1, 2))
        self.im_neurons_above_plane.set_xdata(neurons_above_plane[:, 1])
        self.im_neurons_above_plane.set_ydata(neurons_above_plane[:, 0])
        self.im_neurons_below_plane.set_xdata(neurons_below_plane[:, 1])
        self.im_neurons_below_plane.set_ydata(neurons_below_plane[:, 0])

        # change the title to instruct user interaction.
        if self.selectpointsmode == True:
            self.ax.set_title("Select points mode. Press ctrl+p to switch back to normal mode")
        else:
            self.ax.set_title(title_mode + self.instruction)

        self.im.axes.figure.canvas.draw()
        # plt.show()

    def find_neuron_around_focus(self, plane_depth=0, method='around'):
        # this function return the neuron on the focus plane

        # get the z coordinate of all neurons
        z = self.worm.X[:, 2]
        # find neurons around focus plane
        if method == 'around':
            # sandwich
            on_focus_idx = np.where((z >= self.z - plane_depth) * (z <= self.z + plane_depth))[0]
        elif method == 'above':
            on_focus_idx = np.where((z >= self.z) * (z <= self.z + plane_depth))[0]
        elif method == 'below':
            on_focus_idx = np.where((z >= self.z - plane_depth) * (z <= self.z))[0]
        else:
            on_focus_idx = np.array([])

        return self.worm.X[on_focus_idx, :], on_focus_idx


class Worm_data(object):
    # this class is to perform the analysis on worm data.
    def __init__(self, folder):
        # this is to load the data for whole-brain imager
        self.folder = folder
        self.worm_loader = worm_data_loader(folder)
        # load the whole-brain imaging analysis results.
        psNew_p = os.path.join(folder, 'pointStatsNew.mat')
        pSNew = sio.loadmat(psNew_p)['pointStatsNew']
        names = pSNew.dtype.names
        pS_dictName_New = dict()
        for i, name in enumerate(names): pS_dictName_New[name] = i
        self.pSNew = pSNew
        self.pS_dictName_New = pS_dictName_New
        # load previous human annotated results.
        self.load_match()

    def load_worm_coord(self, vidx):
        trackIdx = self.pSNew[0, vidx-1][self.pS_dictName_New['trackIdx']][:, 0]
        X_raw = self.pSNew[0, vidx-1][self.pS_dictName_New['rawPoints']]
        #print(vidx, self.pSNew[0, vidx-1][self.pS_dictName_New['stackIdx']])
        self.X = X_raw - 1
        self.X_ori = np.copy(self.X)
        self.trackIdx = trackIdx
        return self.X, trackIdx

    def load_match(self):
        # load the former match.
        match_folder = self.folder
        match_file = os.path.join(match_folder, 'match_wbi2mcw.txt')
        if os.path.exists(match_file):
            with open(match_file, "rb") as fp:  # Pickling
                match = pickle.load(fp)
                fp.close()
            self.label_human = match['match_label']
            self.labeledPoints = match['labeled_pt']
            print('load match results from {}'.format(match_file))
        else:
            self.label_human = dict()
            self.labeledPoints = dict()

    def save_match(self):
        # choose the template folder to store the match
        folder = self.folder
        # dict match to store match information
        match = dict()
        # store match (label_wbi: label_mcw)
        match['match_label'] = self.label_human
        # store annotated pts (vidx, zidx, x, y)
        match['labeled_pt'] = self.labeledPoints

        # save match
        filename = os.path.join(folder, 'match_wbi2mcw.txt')
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(match, fp)
            fp.close()
        print('save match results to {}'.format(filename))

    def normalize_color_image(self, channel=[0, 2, 4], template='../data/neuropal/20190109_162235.txt'):
        # normalize the color in the given channel. Then return the new image with those channels.
        # use color align method.(see test_for_cpd)

        # load the neuron information from template.
        with open(template, "rb") as f:
            cervello = pickle.load(f)
            temp_brt = cervello['Brightness']

        # normalize the color channel one by one
        num_cha = self.X_brt.shape[1]
        img_norm = list()
        for i in range(num_cha):
            if i in channel:
                temp_c = temp_brt[:, i]
                worm_c = self.X_brt[:, i]
                # get the linear transformation
                _, z, scale = self.preprocess.align_color(temp_c, worm_c, get_trans=True, scale_thd=0.8)

                # transform the image.
                tmp_img = self.image[:, i, :, :]
                tmp_img = (tmp_img * z[0] + z[1]) / (scale + 1)
                tmp_img[tmp_img > 1] = 1
                tmp_img[tmp_img < 0] = 0
                img_norm.append(tmp_img)

        # get the image in the correct format(z,x,y,channel)
        img_norm = np.array(img_norm)
        return np.moveaxis(img_norm, [0, 1, 2, 3], [1, 0, 2, 3])


# main function to get the images.
if __name__ == "__main__":
    # This code is to compare the neurons of two worm(one template, the other to be aligned.)
    # first load the worm with a worm instance.
    # template worm
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="/projects/LEIFER/PanNeuronal/20210305/BrainScanner20210305_152429",
                        type=str)
    args = parser.parse_args()
    folder = args.folder
    if folder[-1] != '/':
        folder = folder + '/'
    term_title = "mc_"+"_".join(folder.split("/")[-2].split("ner")[1:])
    print("\x1b]2;"+term_title+"\x07",end="\r")

    worm_tem = Worm_data(args.folder)  # 20190314_162708  20190403_160854 20190109_160553
    fig_worm_tem = Label_Neuron(worm_tem, order='z', colors=['blue', 'green', 'white', 'red'])
    fig_worm_tem.parent_term = term_title
    # show the image
    plt.show()
    while True:
        if fig_worm_tem.has_been_closed:
            fig_worm_tem.worm.save_match()
            plt.close('all')
            break
        fig_worm_tem.update()





