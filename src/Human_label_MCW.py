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
from pairwise_cpd import preprocess_mcw
from plot_two_matched_worm import create_letter_label
import os
import argparse


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
    def __init__(self, worm, order="cz", cmap='', colors=['blue', 'cyan', 'green', 'orange', 'red', 'magenta'], \
                 Overlay=[], user='manual'):
        # Initialize the figure for showing one worm
        # worm is the istance for worm data.
        # A is the image of worms
        # order is the order of channel and z in the array.
        # color is the color used for each channel.
        # overlay
        self.user = user
        # switch the z and channel if necessary
        self.worm = worm
        # show multiple in same image with rgb color.
        self.show_multiple_channel = True
        # the mode of figure
        # mode 1, z stack images of 3 channels in RGB.
        # mode 2, z max projection of 3 channels in RGB.
        # mode 3, z stack images of all individual channel.
        self.figure_mode = 2
        # show the label of already matched neurons.
        self.show_label = 1
        self.show_focus_ind = True
        # A hidden mode: to modify the colormap of current channel intensity.
        self.modify_colormap = False
        self.scale_blue = 1
        self.scale_green = 1
        self.scale_red = 1
        self.ch = 0
        self.cb = None
        # the string of instruction for user interaction.
        self.instruction = 'ctrl+p to enter point selection mode. ' + \
                           'ctrl+m to change the mode of figure(z stack, z projection)\n' + \
                           'ctrl+l to change the mode of showing the label. ' + \
                           'ctrl+i to turn on/off the index of focus\n' + \
                           'ctrl+g to save current matching. ' + \
                           'Scroll/up-down to navigate in z\n' + \
                           'A-D/left-right to change channel. ' + \
                           'W-S to change color scaling'
        self.instruction2 = 'anterior-posterior axis is along x, worm is already straightened.\n' + \
                            'To change the channel(color) go to the other figure and use key a or d\n'

        # A = worm.img_norm
        A = worm.image
        A[A < 0] = 0
        if order == "zc": A = np.swapaxes(A, 0, 1)
        # Initialize the figure to show worms.
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('button_press_event', self.onbuttonpress)
        self.fig2 = plt.figure(2)
        # self.ax2 = Axes3D(self.fig2)
        # self.ax2.autoscale(enable=False)
        # self.fig2.canvas.mpl_connect('key_press_event', self.onkeypress)
        # X_s_std = np.std(self.worm.X_s, axis=0) * 2
        X_s_mean = np.mean(self.worm.X_s, axis=0)
        self.xlim = np.array([-50, 100]) + X_s_mean[0]
        self.ylim = np.array([-15, 15]) + X_s_mean[1]
        self.zlim = np.array([-15, 15]) + X_s_mean[2]
        # print(self.xlim, self.ylim, self.zlim)
        # self.plot_neuron_3d()
        # self.z = 15
        # neurons_on_plane = self.find_neuron_around_focus()
        # self.im_neurons_on_plane, = self.ax.plot(neurons_on_plane[:,1],neurons_on_plane[:,0],'o',markersize=1,c='w')
        self.im_neurons_above_plane, = self.ax.plot([], [], '|', markersize=4, c='red')
        self.im_neurons_below_plane, = self.ax.plot([], [], '_', markersize=4, c='red')
        super().__init__(self.ax, A, cmap=cmap, colors=colors, Overlay=Overlay)
        self.X_max = np.max(self.X, axis=1)
        # For the hidden colormap modification mode
        self.X_backup = np.copy(self.X)
        self.X_max_backup = np.copy(self.X_max)

    def onbuttonpress(self, event):
        if self.selectpointsmode:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                label = input("Enter label for (" + str(int(ix)) + "," + str(int(iy)) + "):  ")

                self.worm.labeledPoints[label] = [self.ch, self.z, iy, ix]
                # find the segmented neuron that cloest to the select point.
                # use the l1 distance is fast.
                dist = np.abs(self.worm.X[:, 0] - iy) + np.abs(self.worm.X[:, 1] - ix) + np.abs(
                    self.worm.X[:, 2] - self.z)
                # get the cloest neuron
                neu_idx = np.argmin(dist)
                print('distance to cloest point:{}'.format(dist[neu_idx]))

                if dist[neu_idx] < 8:
                    if label == ' ':
                        print('delete label:{} at {}'.format(self.worm.label_human[neu_idx], \
                                                             self.worm.X[neu_idx, :]))
                    else:
                        print('add label {} to neuron at {}'.format(label, self.worm.X[neu_idx, :]))
                    self.worm.label_human[neu_idx] = label
                else:
                    print('No segmented neuron found in the neighbor')

            self.update()

    def onkeypress(self, event):
        # handle the event of a key press.
        if not self.selectpointsmode:
            if event.key == 'ctrl+m':
                # change mode of figure.
                # total number of mode is 3.
                self.figure_mode = np.mod(self.figure_mode + 1, 3)

            elif event.key == 'ctrl+l':
                self.show_label = (self.show_label + 1) % 3

            elif event.key == 'ctrl+i':
                self.show_focus_ind = not self.show_focus_ind

            elif event.key == 'ctrl+g':
                # save the current matching
                self.worm.save_match(template=self.user)
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
            self.im.set_data(np.moveaxis(self.X[:, self.z, ...], [0, 1, 2], [2, 0, 1])[:, :, [3, 1, 0]])
            # self.im.set_cmap('')
            self.ax.set_xlabel('slice ' + str(self.z))
            # show the mode in title
            title_mode = 'Mode 1: z stack images of 3 channels in RGB.\n'

        elif self.figure_mode == 1:
            # mode 1, z max projection of 3 channels in RGB.
            #      img = np.max(self.X, axis=1)
            #      img = np.moveaxis(img,[0,1,2],[2,0,1])
            #      self.im.set_data(img[:,:,[3,1,0]])
            self.im.set_data(np.moveaxis(self.X_max, [0, 1, 2], [2, 0, 1])[:, :, [3, 1, 0]])
            # self.im.set_cmap('')
            self.ax.set_xlabel('slice ' + str(self.z))
            # show the mode in title
            title_mode = 'Mode 2: z max projection of 3 channels in RGB.\n'

        elif self.figure_mode == 2:
            # mode 2, z stack images of all individual channel.
            # update the image with the input.
            if len(self.dimensions) == 4:
                self.im.set_data(self.X[self.ch, self.z, ...])  # /self.scale[self.ch])
            else:
                self.im.set_data(self.X[self.z, ...])  # /self.scale[self.ch])

            # update the colormap
            if self.maxX * self.scale[self.ch] > self.im.norm.vmin:
                norm = colors.Normalize(vmin=self.im.norm.vmin, vmax=self.maxX * self.scale[self.ch])
                self.im.set_norm(norm)
            self.ax.set_xlabel('slice ' + str(self.z) + "   channel " + str(self.ch))
            self.im.set_cmap(self.Cmap[self.ch])
            # show the mode in title
            title_mode = 'Mode 3: z stack images of all individual channel.\n'

        # plot the label of already matched neurons.
        if self.show_label:
            # get the neurons that are on the current focus plane.
            neurons_on_plane, neurons_on_plane_idx = self.find_neuron_around_focus(plane_depth=0, method='around')

            # text the label in the image
            if len(neurons_on_plane_idx):
                for i in np.nditer(neurons_on_plane_idx):
                    if self.worm.label_human[i] != ' ':
                        self.ax.text(self.worm.X[i, 1], self.worm.X[i, 0], self.worm.label_human[i], fontsize=10,
                                     color='red')
                    elif self.show_label == 2:
                        self.ax.text(self.worm.X[i, 1], self.worm.X[i, 0], self.worm.label[i], fontsize=10, color='red')

        # find the neuron on focus
        if self.show_focus_ind:
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
        self.plot_neuron_3d()
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
    def __init__(self, worm_time, worm_strain='neuropal', user=None):
        # get the folder name and strain type of the worm to be analyzed.
        # worm_time is the time of recording that is taken
        self.worm_time = worm_time
        self.worm_strain = worm_strain
        self.user = user
        self.preprocess = preprocess_mcw(method='align', worm=worm_strain)
        self.loadData(worm_time)

        if worm_strain == 'neuropal':
            self.img_norm = self.normalize_color_image(channel=[0, 2, 3, 4], \
                                                       template='/tigress/LEIFER/multicolor/Public_annotation/data/neuropal/20190109_162235.txt')
        else:
            self.img_norm = self.normalize_color_image(channel=[0, 2, 3, 4], \
                                                       template='/tigress/LEIFER/multicolor/Public_annotation/data/mcw/20190602_110558.txt')

    def loadData(self, worm_time):
        # load the data about the worm specified by user.
        # load the worm image, this is in the raw data folder.
        image_file = '/tigress/LEIFER/multicolor/data/' + worm_time[
                                                          0:8] + '/' + 'multicolorworm_' + worm_time + '/unmixed_fr.npy'
        self.image = np.load(image_file)

        # load the neurons segmented from the image.
        # this is in the neuropal or mcw folder.
        if self.worm_strain == 'neuropal':
            seg_folder = '/tigress/LEIFER/multicolor/Public_annotation/data/neuropal/'
        else:
            seg_folder = '/tigress/LEIFER/multicolor/Public_annotation/data/mcw/'

        with open(seg_folder + worm_time + '.txt', "rb") as f:
            cervello = pickle.load(f)
            f.close()
        self.X = np.copy(cervello['Neuron'])
        self.X_brt = np.copy(cervello['Brightness'])
        self.X_s = np.copy(cervello['sNeuron'])

        # load the former match.
        match_folder = os.path.join('/tigress/LEIFER/multicolor/Public_annotation/annotation', 'template_' + self.user)
        match_file = os.path.join(match_folder, worm_time) + '.txt'
        self.initial_label()
        if os.path.exists(match_file):
            with open(match_file, "rb") as fp:  # Pickling
                match = pickle.load(fp)
                fp.close()
            self.label_human = match['match_label']
            self.labeledPoints = match['labeled_pt']
        else:
            num_neurons = len(self.X)
            self.label_human = [' '] * num_neurons
            self.labeledPoints = dict()

    #  def rotate_img(self, img, theta, ux, uy, uz):
    #    # rotate the 3d image.
    #    rotate_3D_image(A, theta, ux, uy, uz, x0=0.0, y0=0.0, z0=0.0)

    def initial_label(self):
        # Initialize the label for each neurons.
        num_neurons = len(self.X)
        self.label = list(range(num_neurons))
        # if self.template:
        #   letter_dict = create_letter_label()
        #   # label is a list of label
        #   self.label = letter_dict[:num_neurons]
        # else:
        #   # 'o' represent not matched.
        #   # for non-template neuron, all of them are initialized as not matched.
        #   self.label = ['o']*num_neurons

    def save_match(self, template='manual'):
        # choose the template folder to store the match
        folder = '/tigress/LEIFER/multicolor/Public_annotation/annotation/template_' + template
        if not os.path.exists(folder):
            os.mkdir(folder)
        # dict match to store match information
        match = dict()
        # store template
        match['template'] = template
        # store worm name
        match['worm'] = self.worm_time
        match['worm_strain'] = self.worm_strain
        match['match_label'] = self.label_human
        match['labeled_pt'] = self.labeledPoints
        match['neuron_pos_s'] = self.X_s
        match['neuron_pos_raw'] = self.X
        # save match
        filename = os.path.join(folder, match['worm']) + '.txt'
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(match, fp)
            fp.close()

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
    parser.add_argument("--user", default="manual",
                        type=str)
    parser.add_argument("--worm_time", default='20190602_094802', type=str)
    parser.add_argument("--strain", default='mcw', type=str)
    args = parser.parse_args()
    user = args.user
    worm_time = args.worm_time
    worm_strain = args.strain
    worm_tem = Worm_data(worm_time, worm_strain=worm_strain,
                         user=user)  # 20190314_162708  20190403_160854 20190109_160553
    fig_worm_tem = Label_Neuron(worm_tem, order='zc', colors=['blue', 'green', 'white', 'red'], user=user)
    # show the image
    plt.show()



