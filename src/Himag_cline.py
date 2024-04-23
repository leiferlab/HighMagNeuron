"""
This file is to get centerline from purely himag images.
"""

import matplotlib.pyplot as plt
import shapely.geometry as geom
from utils import get_curve_representation, line_quadratic, smooth_2d_pts, polygon
import os
import glob
import pickle
import scipy.io as sio
from skimage import io
import cv2
from cpd_nonrigid_sep import register_nonrigid
from datetime import date
import argparse
from HighResTest import Multi_Slice_Viewer
from cpd_rigid_sep import register_translation, register_rigid
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import skimage.morphology as skmorp
from HighResFlow import worm_data_loader, warp_flow
from scipy.optimize import linear_sum_assignment
import time

def find_hung_match_average(neurons, tmp, dim=2):
    if dim is None:
        dim = neurons.shape[1]

    dis = np.sum((tmp[:, np.newaxis, :dim] - neurons[np.newaxis, :, :dim]) ** 2, axis=2)
    row, col = linear_sum_assignment(dis)
    mean_dis = np.mean(np.sqrt(dis[row, col]))
    return mean_dis

def find_min_match(neurons, tmp, dim=2):
    if dim is None:
        dim = neurons.shape[1]
    dis = neurons[:, np.newaxis, :dim] - tmp[:, :dim]
    dis = np.sqrt(np.sum(dis ** 2, axis=2))
    idx = np.argmin(dis, axis=1)
    x_idx = np.arange(neurons.shape[0])
    dis_min = dis[x_idx, idx]
    return idx, dis_min


class worm_cline(object):
    #This is a class for storing the centerline of worm
    def __init__(self, Centerline, degree=3, num_pt=100, length_lim=300):
        self.num_pt = num_pt
        self.update_cline(Centerline)
        self.degree = degree
        self.length_lim = length_lim


    def update_cline(self, Centerline, degree=3):
        # extend the Centerline on head and tail.
        Centerline = Centerline[:, :2]
        # self.head_pt = Centerline[0]
        # self.tail_pt = Centerline[-1]
        # head_dir = Centerline[0, :] - Centerline[1, :]
        # tail_dir = Centerline[-1, :] - Centerline[-2, :]
        # head_dir = head_dir / np.sqrt(np.sum(head_dir ** 2))
        # tail_dir = tail_dir / np.sqrt(np.sum(tail_dir ** 2))
        # head_ext = self.head_pt + 100 * head_dir
        # tail_ext = self.tail_pt + 100 * tail_dir

        self.Centerline = smooth_2d_pts(Centerline, degree=degree, num_point_out=self.num_pt)

        # Cline_ext = np.copy(self.Centerline)
        # Cline_ext[0, :] = head_ext
        # Cline_ext[-1, :] = tail_ext

        self.shCenterline = geom.LineString(self.Centerline)

        self.length = self.shCenterline.length

        head = self.shCenterline.interpolate(0)
        self.head_pt = np.array([head.x, head.y])
        head_next = self.shCenterline.interpolate(20)
        self.head_dir = self.norm_dir(np.array([head_next.x - head.x, head_next.y - head.y]))


        tail = self.shCenterline.interpolate(self.length)
        self.tail_pt = np.array([tail.x, tail.y])
        tail_prev = self.shCenterline.interpolate(self.length - 20)
        self.tail_dir = self.norm_dir(np.array([tail.x - tail_prev.x, tail.y - tail_prev.y]))
        self.straight_origin = 0

    def norm_dir(self, cur_dir):
        # normalize direction.
        return cur_dir / (np.sqrt(np.sum(cur_dir ** 2)) + 1e-6)


    def autofluorescence_mask(self, pts_s, inten, keep_idx=None, plot=False):
        # try not to include those autofluorescence, close to body and dim
        # find x > ? and inten < ?
        # overlap with keep_idx
        inten = inten - np.min(inten)
        num_lim = 150
        if keep_idx is None:
            keep_idx = np.arange(len(pts_s))
        if len(pts_s) <= num_lim:
            return keep_idx
        # only handle situation of >150

        x = np.copy(pts_s[:, 0])
        x.sort()
        x_threshold = max(150, x[num_lim-1])
        # take 20 percentile as color threshold
        c_threshold = np.percentile(inten, 10) + 50
        mask = (pts_s[:, 0] > x_threshold) * (inten < c_threshold)

        new_keep = np.array([idx for idx in keep_idx if not mask[idx]])
        if plot:
            plt.scatter(pts_s[:, 0], inten, c='red')
            plt.scatter(pts_s[new_keep, 0], inten[new_keep], c='green')
            plt.show()
        return new_keep

    def head_orient_cline(self, method='straight'):
        head_pt = self.shCenterline.interpolate(0)
        head_pt = np.array([head_pt.x, head_pt.y])
        body_pt = self.shCenterline.interpolate(120)
        body_pt = np.array([body_pt.x, body_pt.y])
        orient_dir = body_pt - head_pt

        orient_dir /= np.sqrt(np.sum(orient_dir ** 2)) + 1e-5
        num_pt = self.num_pt
        #orient_cl = np.arange(num_pt) / (num_pt - 1) * self.length * orient_dir + head_pt
        orient_cl = np.repeat(orient_dir[np.newaxis, :], num_pt, axis=0) * np.arange(num_pt)[:, np.newaxis] / (num_pt - 1) * self.length + head_pt

        if method == 'straight':
            out = worm_cline(orient_cl)
        elif method == 'combine':
            cl = (orient_cl + self.Centerline) * 0.5
            out = worm_cline(cl)
        # plt.scatter(orient_cl[:, 0], orient_cl[:, 1], c='red')
        # plt.scatter(self.Centerline[:, 0], self.Centerline[:, 1], c='yellow')
        # plt.scatter(cl[:, 0], cl[:, 1], c='green')
        # plt.show()

        # tail_pt = head_pt + self.length * orient_dir
        # orient_cl = np.array([head_pt, tail_pt])
        return out

    def cut_head(self, neurons):
        # this function limit the cline to where the points are.
        # cut the cline to fit the neurons,
        # first one has s 0, length is last s
        neurons_s = self.straighten(neurons)
        num_pt = self.num_pt
        tail_s = self.straighten(np.array([self.tail_pt]))
        max_s = min(tail_s[0, 0], self.length_lim)
        max_s = max(50, max_s)
        min_s = max(0, neurons_s[:, 0].min())

        close_idx = np.where(np.abs(neurons_s[:, 1]) < 25)[0]
        if len(close_idx) > 1:
            sort_s = np.sort(neurons_s[close_idx, 0])
            diff_s = np.diff(sort_s)
            s_next = sort_s[1:]

            s_can = np.where((diff_s > 30) * (s_next < 100))[0]
            if len(s_can) > 0:
                print('get rid of tip piece')
                min_s = s_next[s_can.min()]


            step = (max_s - min_s) / (num_pt - 1)
            output = np.arange(min_s, max_s + step / 2, step)
            #output = self.length * np.arange((num_pt)) / (num_pt - 1) - ref_s + ref_s_new
            output = np.stack((output, np.zeros(num_pt)), axis=0).T
            new_cline = self.project(output)
        else:
            new_cline = self.Centerline
        return new_cline


    def cpd_cline_update(self, neurons1, neurons2, degree=3, num_pt=None):
        # transform the current centerline with respect to the pair of neurons
        # with cpd, neuron2 is cpd transform of neuron1, result cline is the same
        # cpd result of current cline.
        #num_neuron = neurons1.shape[0]
        if num_pt is None:
            num_pt = self.num_pt

        if neurons1.shape[0] < 10:
            return self.Centerline

        neurons1_s = self.straighten(neurons1)
        neurons2_s = self.straighten(neurons2)

        ref_s = np.min(neurons1_s[:, 0])

        neurons_diff = neurons2_s - neurons1_s

        # build a diff moving average.
        s_min, s_max = neurons1_s[:, 0].min(), neurons1_s[:, 0].max()

        num_new_s = 30
        s_step = (s_max - s_min) / (num_new_s - 1)
        s_step = 10
        s_array = np.arange(s_min, s_max + s_step / 2, s_step)

        s_list = list()
        diff_list = list()

        num_eval = 2
        for s in s_array:
            pos_idx = np.where((np.abs(neurons1_s[:, 0] - s) < s_step) * (neurons1_s[:, 1] > 0))[0]
            neg_idx = np.where((np.abs(neurons1_s[:, 0] - s) < s_step) * (neurons1_s[:, 1] <= 0))[0]
            if len(pos_idx) > num_eval:
                s_list.append(s)
                diff_list.append(np.mean(neurons_diff[pos_idx, :], axis=0))

            if len(neg_idx) > num_eval:
                if len(pos_idx) > num_eval:
                    diff_list[-1] = 0.5 * (diff_list[-1] + np.mean(neurons_diff[neg_idx, :], axis=0))
                else:
                    s_list.append(s)
                    diff_list.append(np.mean(neurons_diff[neg_idx, :], axis=0))

        if len(s_list) < 5:
            return self.Centerline

        diff_list = np.array(diff_list)
        s_list = np.array(s_list)

        x = s_list
        s_list_min, s_list_max = s_list.min(), s_list.max()
        step = (s_list_max - s_list_min) / (num_pt - 1)
        x_new = np.arange(s_list_min, s_list_max + step / 2, step)

        f1 = interp1d(x, diff_list[:, 0], fill_value='extrapolate')
        f2 = interp1d(x, diff_list[:, 1], fill_value='extrapolate')

        new_cline_s = np.zeros((num_pt, 2))
        diff_new = np.zeros((num_pt, 2))
        diff_new_s = np.zeros((num_pt, 2))
        new_cline_s_o = np.zeros((num_pt, 2))

        diff_new[:, 0] = f1(x_new)
        diff_new[:, 1] = f2(x_new)

        win_length = 20
        win = max(3, int(np.ceil(win_length / step)))
        win = win + 1 if not (win // 2 == 0) else win

        # diff_new_s[:, 0] = savgol_filter(diff_new[:, 0], window_length=win, polyorder=2)
        # diff_new_s[:, 1] = savgol_filter(diff_new[:, 1], window_length=win, polyorder=2)
        diff_new_s[:, 0] = gaussian_filter1d(diff_new[:, 0], sigma=win, mode='reflect')
        diff_new_s[:, 1] = gaussian_filter1d(diff_new[:, 1], sigma=win, mode='reflect')

        new_cline_s[:, 0] = diff_new_s[:, 0] + x_new
        new_cline_s[:, 1] = diff_new_s[:, 1]


        cline_tmp = worm_cline(new_cline_s)
        neurons2_s_new = cline_tmp.straighten(neurons2_s)

        ref_s_new = np.min(neurons2_s_new[:, 0])

        new_s_min = ref_s_new - ref_s
        new_s_max = max(new_s_min + 50, np.max(neurons2_s_new[:, 0]))
        new_s_max = min(self.length_lim, new_s_max)
        step = (new_s_max - new_s_min) / (num_pt - 1)
        output = np.arange(new_s_min, new_s_max + step / 2, step)
        #output = self.length * np.arange((num_pt)) / (num_pt - 1) - ref_s + ref_s_new
        output = np.stack((output, np.zeros(num_pt)), axis=0).T
        new_cline_s_reg = cline_tmp.project(output)
        # plt.scatter(neurons1_s[:, 0], neurons1_s[:, 1], c='red')
        #
        # plt.scatter(neurons2_s[:, 0], neurons2_s[:, 1], c='blue')
        # plt.scatter(new_cline_s_reg[:, 0], new_cline_s_reg[:, 1], c='blue', marker='+')
        # plt.show()
        new_cline = self.project(new_cline_s_reg)

        # plt.scatter(neurons1[:, 0], neurons1[:, 1], c='red')
        # plt.scatter(self.Centerline[:, 0], self.Centerline[:, 1], c='red', marker='+')
        # plt.scatter(neurons2[:, 0], neurons2[:, 1], c='blue')
        # plt.scatter(new_cline[:, 0], new_cline[:, 1], c='blue', marker='+')
        # plt.show()

        # new_cline_f = worm_cline(new_cline)
        # neurons2_s_new = new_cline_f.straighten(neurons2)
        #print('neuron1 median:{}, neuron2 median:{}'.format(np.median(neurons1_s[:, 0]), np.median(neurons2_s_new[:, 0])))

        return new_cline


    def get_dir(self, s):
        # This function get the direction of worm based on the length(straightened coordinate x)
        if s <= 0:
            cur_dir = self.head_dir
        elif s <= self.length / 2:
            point_1 = self.shCenterline.interpolate(s)
            point_2 = self.shCenterline.interpolate(s + 1)
            cur_dir = np.array([point_2.x - point_1.x, point_2.y - point_1.y])
        elif s <= self.length:
            point_1 = self.shCenterline.interpolate(s - 1)
            point_2 = self.shCenterline.interpolate(s)
            cur_dir = np.array([point_2.x - point_1.x, point_2.y - point_1.y])
        else:
            cur_dir = self.tail_dir

        cur_dir = self.norm_dir(cur_dir)
        return cur_dir

    def straighten(self, Neurons):
        assert len(Neurons.shape) == 2
        #N = Neurons.shape[0]

        sNeurons = list()
        # Straighten the brain. The z coordinate does not change.
        for neuron in Neurons:
            point = geom.Point(neuron[0], neuron[1])

            # Calculate distance from the centerline and longitudinal position
            # along the centerline. Apart from the sign of y, these are the
            # coordinates in the straightened frame of reference.
            x = self.shCenterline.project(point)
            if x <= 0:
                point_coord = np.array([point.x, point.y])
                point_head = point_coord - self.head_pt

                x_out = np.dot(self.head_dir, point_head)
                y_out = self.head_dir[0] * point_head[1] - self.head_dir[1] * point_head[0]
            elif x < self.length:

                y = self.shCenterline.distance(point)

                # Find the coordinates of the projection of the neuron on the
                # centerline, in the original frame of reference.
                a = self.shCenterline.interpolate(x)
                # Find the vector going from the projection of the neuron on the
                # centerline to the neuron itself.
                vpx = point.x - a.x
                vpy = point.y - a.y

                # Move along the line in the positive direction.
                cur_dir = self.get_dir(x)
                vx, vy = cur_dir[0], cur_dir[1]

                # Calculate the cross product v x vp. Its sign is the sign of y.
                s = np.sign(vx * vpy - vy * vpx)
                x_out = x
                y_out = s * y
            else:
                point_coord = np.array([point.x, point.y])
                point_tail = point_coord - self.tail_pt

                x_out = np.dot(self.tail_dir, point_tail) + self.length
                y_out = self.tail_dir[0] * point_tail[1] - self.tail_dir[1] * point_tail[0]


            sNeurons.append([x_out, y_out])

        if Neurons.shape[1] > 2:
            sNeurons = np.hstack((np.array(sNeurons), Neurons[:, 2:]))
        else:
            sNeurons = np.array(sNeurons)
        # straight origin is the coordinate of start point in straightened coordinate system.
        sNeurons[:, 0] += self.straight_origin
        return sNeurons

    def dir_ortho(self, cur_dir):
        ortho = np.copy(cur_dir)
        ortho[0] = -cur_dir[1]
        ortho[1] = cur_dir[0]
        return ortho

    def update_straight_origin(self, straight_origin):
        self.straight_origin = straight_origin

    def project(self, sNeurons):
        assert len(sNeurons.shape) == 2
        # straight_origin is the start point of line in straight coordinate
        sNeurons = np.copy(sNeurons)
        sNeurons[:, 0] -= self.straight_origin
        Neurons = list()
        for neuron in sNeurons:

            if neuron[0] < 0:
                head_dir_ortho = self.dir_ortho(self.head_dir)
                point_out = self.head_pt + neuron[0] * self.head_dir + neuron[1] * head_dir_ortho
            elif neuron[0] < self.length:
                a = self.shCenterline.interpolate(neuron[0])
                point_out = np.array([a.x, a.y])
                cur_dir = self.get_dir(neuron[0])
                cur_dir_ortho = self.dir_ortho(cur_dir)
                point_out = point_out + cur_dir_ortho * neuron[1]
            else:
                tail_dir_ortho = self.dir_ortho(self.tail_dir)
                point_out = self.tail_pt + (neuron[0] - self.length) * self.tail_dir + neuron[1] * tail_dir_ortho
            Neurons.append(point_out)
        if sNeurons.shape[1] > 2:
            Neurons = np.hstack((np.array(Neurons), sNeurons[:, 2:]))
        else:
            Neurons = np.array(Neurons)

        return Neurons


    def mask_with_template(self, Neurons, tp_s=None, straight=True, show=False):
        if straight:
            # need to do straighting.
            Neurons = self.straighten(Neurons)


        x_threshold_head_ext = [-25, -5]
        y_threshold_head_ext = [-20, 20]

        mask_head_ext = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head_ext, y_threshold_head_ext)

        x_threshold_head = [-5, 100]
        y_threshold_head = [-40, 40]

        mask_head = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head, y_threshold_head)

        x_threshold_body = [100, self.length + 50]
        y_threshold_body = [-100, 100]
        mask_body = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_body, y_threshold_body)


        # also keep the neurons that are close to current neurons.
        mask = (mask_body + mask_head_ext + mask_head)

        if tp_s is not None:
            bad_idx = np.where(np.logical_not(mask))[0]
            bad_pts = Neurons[bad_idx, :]
            dis = bad_pts[:, np.newaxis, :2] - tp_s[:, :2]
            dis = np.sqrt(np.sum(dis ** 2, axis=2))
            dis_min = np.min(dis, axis=1)
            good_idx = np.where(dis_min < 15)[0]
            mask[bad_idx[good_idx]] = 1


        keep_idx = self.find_all_neighbor(Neurons, mask, dis_neigh=10)

        if show:
            plt.scatter(tp_s[:, 0], tp_s[:, 1], c='green')
            plt.scatter(Neurons[:, 0], Neurons[:, 1], c='yellow')
            plt.scatter(Neurons[keep_idx, 0], Neurons[keep_idx, 1], c='red', marker='+')
            plt.show()
            print('stop')

        return keep_idx

    def find_all_neighbor(self, Neurons_s, mask, max_iter=5, dis_neigh=10):
        increase_sz = 1
        keep_idx = np.where(mask > 0)[0]
        bad_idx = np.where(mask == 0)[0]
        mask = np.copy(mask)
        i = 0
        if len(keep_idx) < Neurons_s.shape[0] and len(keep_idx) > 0:
            while increase_sz > 0 and i < max_iter and len(bad_idx) > 0:

                bad_pts = Neurons_s[bad_idx, :2]
                good_pts = Neurons_s[keep_idx, :2]

                dis = bad_pts[:, np.newaxis, :2] - good_pts[:, :2]
                dis = np.sqrt(np.sum(dis ** 2, axis=2))
                dis_min = np.min(dis, axis=1)

                new_good_idx = bad_idx[np.where(dis_min < dis_neigh)[0]]
                mask[new_good_idx] = True
                increase_sz = len(new_good_idx)
                keep_idx = new_good_idx
                bad_idx = np.where(mask == 0)[0]
                i += 1
        keep_idx = np.where(mask)[0]
        return keep_idx

    def mask_with_threshold(self, s_array, y_array, threshold_x, threshold_y):
        mask_x = (s_array > threshold_x[0]) * (s_array < threshold_x[1])
        mask_y = (y_array > threshold_y[0]) * (y_array < threshold_y[1])
        return mask_x * mask_y



    def mask_neurons(self, Neurons, straight=True):
        # mask out some neurons that are far from the centerline.
        if straight:
            # need to do straighting.
            Neurons = self.straighten(Neurons)

        # def outside_with_threshold(s_array, threshold_x, threshold_y)
        #     mask_x = (s_array > threshold_x[0]) * (s_array < threshold_x[1])
        #     mask_y = (s_array <= threshold_y[0]) + (s_array >= threshold_y[1])
        #     return mask_x * mask_y

        # initial with False.



        mask_out = Neurons[:, 0] < -1e6

        x_threshold_head_ext = [-25, -5]
        y_threshold_head_ext = [-20, 20]

        mask_head_ext = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head_ext, y_threshold_head_ext)

        x_threshold_head = [-5, 100]
        y_threshold_head = [-50, 50]

        mask_head = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_head, y_threshold_head)

        x_threshold_body = [100, self.length + 50]
        y_threshold_body = [-100, 100]
        mask_body = self.mask_with_threshold(Neurons[:, 0], Neurons[:, 1], x_threshold_body, y_threshold_body)


        # also keep the neurons that are close to current neurons.
        cur_mask = mask_body + mask_head_ext + mask_head
        keep_idx = self.find_all_neighbor(Neurons, cur_mask, dis_neigh=10)
        return keep_idx

    def generate_cover_mask(self, pts_s, bbox=[0, -30, 50, 30], radius=3, show=False):
        pts_new = np.copy(pts_s)
        pts_new[:, 0] -= bbox[0]
        pts_new[:, 1] -= bbox[1]

        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1

        mask = np.zeros((width, height))

        pt_idx = np.where((pts_s[:, 0] >= bbox[0]) * (pts_s[:, 0] < bbox[2]) * (pts_s[:, 1] >= bbox[1]) * \
                          (pts_s[:, 1] < bbox[3]))[0]


        pt_contain = np.rint(np.copy(pts_s[pt_idx, :])).astype(np.int16)
        pt_contain[:, 0] -= bbox[0]
        pt_contain[:, 1] -= bbox[1]

        mask[pt_contain[:, 0], pt_contain[:, 1]] = 1
        selem = skmorp.disk(radius=radius)
        mask_dilate = skmorp.binary_dilation(mask, selem)

        if show:
            plt.imshow(mask_dilate)
            plt.show()
        num_cover = np.sum(mask_dilate)

        return num_cover

    def human_annotate_direction(self):
        anno_file = os.path.join(self.ori_folder, 'anno.file')
        if os.path.exists(anno_file):
            with open(anno_file, 'rb') as fp:
                anno_d = pickle.load(fp)
                fp.close()
        else:
            anno_d = dict()
            anno_d['flip_list'] = np.array([False] * (self.max_stack + 1))

        anno_idx_start = int(input("type the volume index, as [start idx, end idx], you want to flip, start idx: (type negative number to stop)"))
        while anno_idx_start >= 0:
            anno_idx_end = int(input("type the volume index, as [start idx, end idx], you want to flip, end idx(included): "))
            if anno_idx_end > self.max_stack:
                print('end index exceeds length of video')
            else:
                anno_d['flip_list'][anno_idx_start:anno_idx_end+1] = True

            anno_idx_start = int(input(
                "type the volume index, as [start idx, end idx], you want to flip, start idx: (type negative number to stop)"))

        with open(anno_file, 'wb') as fp:
            pickle.dump(anno_d, fp)
            fp.close()




    def evaluate_orientation_hung(self, Neurons_all, tp_s, good_idx, scale=200, straight=True):
        Neurons_all = np.copy(Neurons_all)
        if straight:
            # need to do straighting.
            Neurons_all = self.straighten(Neurons_all)
        Neurons = np.copy(Neurons_all)
        Neurons[:, 2] = Neurons[:, 2] - np.median(Neurons[:, 2])
        tp_s_check = np.copy(tp_s)
        # get
        tp_s_check[:, 2] = tp_s_check[:, 2] - np.median(tp_s_check[:, 2])
        keep_idx = np.where((tp_s[:, 0] >= 0) * (np.abs(tp_s[:, 1]) < 60))[0]
        tp_s_check = tp_s_check[keep_idx, :]

        tp_s_inv = np.copy(tp_s_check)
        tp_s_inv[:, 0] = self.length - tp_s_inv[:, 0]
        tp_s_inv[:, 1] = -tp_s_inv[:, 1]
        tp_s_inv[:, 1] = tp_s_inv[:, 1] - np.median(tp_s_inv[:, 1]) + np.median(Neurons[:, 1])
        # match the center of tp_s_inv and template

        mean_dis_ori = find_hung_match_average(Neurons, tp_s_check, dim=3)
        mean_dis_inv = find_hung_match_average(Neurons, tp_s_inv, dim=3)
        print(mean_dis_ori, mean_dis_inv)
        #print('Run time:{}'.format(time.time() - tic))
        # plt.scatter(Neurons[:, 0], Neurons[:, 1], c='black')
        # plt.scatter(tp_s_inv[:, 0], tp_s_inv[:, 1], c='red')
        # plt.scatter(tp_s_check[:, 0], tp_s_check[:, 1], c='green')
        # plt.show()

        turn = True if mean_dis_inv < mean_dis_ori else False
        return turn

    def count_number_head(self, Neurons, head_l, head_w, c_w=5):
        Neurons_abs1 = np.abs(Neurons[:, 1] - np.median(Neurons[:, 1]))
        head_part = (Neurons[:, 0] < head_l) * (Neurons[:, 0] >= 0) * (Neurons_abs1 < head_w) * (Neurons_abs1 > c_w)
        num_head = np.sum(head_part)
        return num_head


    def evaluate_orientation(self, Neurons_all, tp_s, good_idx, scale=200, straight=True):
        # evaluate the orientation of current cline.
        #turn = True
        try:
            ori_score = 0

            hung_turn = self.evaluate_orientation_hung(Neurons_all, tp_s, good_idx, scale=scale, straight=straight)
            if not hung_turn:
                ori_score += 1

            Neurons_all = np.copy(Neurons_all)
            if straight:
                # need to do straighting.
                Neurons_all = self.straighten(Neurons_all)

            Neurons = np.copy(Neurons_all[good_idx, :])

            # make Neurons center in y dimension.
            Neurons[:, 1] -= np.median(Neurons[:, 1])

            tp_s_check = np.copy(tp_s)
            tp_s_check[:, 2] = tp_s_check[:, 2] - np.median(tp_s_check[:, 2])
            tp_s_check[:, 1] = tp_s_check[:, 1] - np.median(tp_s_check[:, 1])
            #plt.hist(tp_s_check[:, 0])
            #plt.show()
            mid_length = np.median(tp_s_check[:, 0])
            front_mask = tp_s_check[:, 0] < mid_length
            width_percent = np.percentile(np.abs(tp_s_check[front_mask, 1]), 80) + 5
            #print('mid_length:{}, width:{}'.format(mid_length, width_percent))

            head_width = max(32, int(width_percent))
            head_length = min(max(70, mid_length - 50), self.length / 2)
            head_length = int(max(20, head_length))


            if True:
                # see if there are neurons in front of head tip
                out_head = np.sum(Neurons_all[:, 0] < 0)
                out_tail = np.sum(Neurons_all[:, 0] > self.length)
                print('out head num:{}, out tail num:{}'.format(out_head, out_tail))
                if out_head < 5 or out_tail > 5:
                    ori_score += 1


            if True:
                print('head_l:{}, head_w:{}'.format(head_length, head_width))
                num_head = self.count_number_head(Neurons, head_length, head_width, c_w=0)

                Neurons_tmp = np.copy(Neurons)
                Neurons_tmp[:, 0] = self.length - Neurons_tmp[:, 0]
                #head_part = np.where((Neurons[:, 0] < head_length) * (Neurons[:, 0] >= 0))[0]
                # if len(head_part):
                #     width_m = np.median(Neurons[head_part, 1])
                #     mask_head = (np.abs(Neurons[head_part, 1] - width_m) < head_width)
                #     num_head = len(np.where(mask_head)[0])
                # else:
                #     num_head = 0
                #tail_part = (Neurons[:, 0] > self.length-head_length) * (Neurons[:, 0] < self.length) * (Neurons_abs1 < head_width) * (Neurons_abs1 > 10)
                num_tail = self.count_number_head(Neurons_tmp, head_length, head_width, c_w=0)
                # tail_part = (Neurons[:, 0] > self.length-head_length) * (Neurons[:, 0] < self.length)
                # if len(tail_part):
                #     width_m = np.median(Neurons[tail_part, 1])
                #     mask_tail = (np.abs(Neurons[tail_part, 1] - width_m) < head_width)
                #     num_tail = len(np.where(mask_tail)[0])
                # else:
                #     num_tail = 0

                if num_head >= num_tail:
                    #turn = False
                    ori_score += 1

            head_0 = int(max(0, np.min(Neurons[:, 0])))
            test_length = int(max(50, min(120, (self.length - head_0) / 2)))

            if True:
                # find if there is a huge gap
                step = 3
                test_idx = test_length // step
                s_test = np.arange(0, self.length + step, step)
                num_c = s_test.shape[0]
                check_c = np.zeros((num_c, 2))
                check_c[:, 0] = s_test

                idx, dis_min = find_min_match(check_c, Neurons)

                dis_thd = test_idx * 15 / 40
                num_front = np.sum(dis_min[:test_idx] < dis_thd)
                num_back = np.sum(dis_min[-test_idx:] < dis_thd)

                if num_front >= (num_back + test_idx // 10):
                    #turn = False
                    ori_score += 1


            cover_front = self.generate_cover_mask(Neurons, bbox=[head_0, -head_width, head_0 + test_length, head_width], show=False)
            cover_back = self.generate_cover_mask(Neurons, bbox=[int(self.length - test_length), -head_width, int(self.length), head_width], show=False)
            if cover_front * 0.8 > cover_back:
                #turn = False
                ori_score += 1

            if num_back > 0.8 * test_idx:
                factor = 1.5
            else:
                factor = 1
            if True:
                # match template to current pts
                dis_thd = 15

                Neurons[:, 2] = Neurons[:, 2] - np.median(Neurons[:, 2])
                # get

                keep_idx = np.where((tp_s[:, 0] >= 0) * (np.abs(tp_s[:, 1]) < 50))[0]
                tp_s_check = tp_s_check[keep_idx, :]
                tp_s_inv = np.copy(tp_s_check)

                tp_s_inv[:, 0] = self.length - tp_s_inv[:, 0]
                tp_s_inv[:, 1] = -tp_s_inv[:, 1]

                idx, dis_min = find_min_match(tp_s_check, Neurons)
                num_ori = np.sum(dis_min < dis_thd)
                dis_ori = np.mean(dis_min)

                idx, dis_min = find_min_match(tp_s_inv, Neurons)
                # use translation to align the points.
                # pt_standard = standard_neurons(Neurons, scale)
                # Neurons_cpd = pt_standard.transform(Neurons)
                # tp_inv_cpd = pt_standard.transform(tp_s_inv)
                #
                # tp_inv_cpd_t, ts = register_translation(tp_inv_cpd, Neurons_cpd, w=0.1)
                # idx, dis_min = find_min_match(tp_inv_cpd_t, Neurons_cpd)
                # dis_min = dis_min * scale

                num_inv = np.sum(dis_min < dis_thd)
                dis_inv = np.mean(dis_min)
                # if num_ori >= num_inv:
                #     turn = False
                if dis_ori * factor <= dis_inv:
                    #turn = False
                    ori_score += 1
            # check ventral cord
            # ventral_cord = (Neurons[:, 0] < 120) * (Neurons[:, 0] >= 0) * (np.abs(Neurons[:, 1]) < 30)
            # num_head = len(np.where(head_part)[0])

            if True:
                print('head num:{}, tail num:{}'.format(num_head, num_tail))
                print('pos num:{}, neg num:{}'.format(num_ori, num_inv))
                print('dis pos:{}, dis inv:{}'.format(dis_ori, dis_inv))
                print('num front:{}, num back:{}'.format(num_front, num_back))
                print('cover front:{}, cover back:{}'.format(cover_front, cover_back))
                print('ori_score:{}'.format(ori_score))


            # test delete
            # plt.scatter(Neurons[:, 0], Neurons[:, 1], c='black')
            # plt.scatter(tp_s_inv[:, 0], tp_s_inv[:, 1], c='red')
            # plt.scatter(tp_s_check[:, 0], tp_s_check[:, 1], c='green')
            # plt.show()


            if ori_score < 2:
                turn = True
            else:
                turn = False
            return turn
        except:
            return False

class standard_neurons(object):
    def __init__(self, neurons, scale=200):
        self.neurons_median = np.median(neurons, axis=0)
        self.scale = scale

    def transform(self, pts):
        pts = np.copy(pts)
        dim = pts.shape[1]
        pts = pts - self.neurons_median[np.newaxis, :dim]
        pts = pts / self.scale
        return pts

    def transform_inv(self, pts):
        pts = np.copy(pts)
        dim = pts.shape[1]
        pts = pts * self.scale
        pts = pts + self.neurons_median[np.newaxis, :dim]
        return pts


class Neuron_cline(object):
    def __init__(self, worm_folder, template_path='/projects/LEIFER/communalCode/HighMagNeuron/template/atlas_1.txt', date_mode='auto', temp_align='none'):
        # initiate
        self.lamb = 1e3
        self.beta = 0.25
        self.degree = 3
        self.scale = 200
        if worm_folder[-1] == '/':
            worm_folder = worm_folder[:-1]

        self.worm_folder = worm_folder
        self.template_align = temp_align

        # make a folder to save neuron_cline
        self.output_folder = os.path.join(worm_folder, 'neuron_cline_result')
        self.cline_init_folder = os.path.join(self.output_folder, 'init')
        self.cline_init_img_folder = os.path.join(self.output_folder, 'init_img')
        self.cline_tp_folder = os.path.join(self.output_folder, 'cline_template')
        self.cline_folder = os.path.join(self.output_folder, 'cline')
        self.ori_folder = os.path.join(self.output_folder, 'ori')
        self.cline_img = os.path.join(self.output_folder, 'cline_img')
        self.neuron_img = os.path.join(self.output_folder, 'neuron_img')
        self.straight_img = os.path.join(self.output_folder, 'straight_img')
        if date_mode == 'auto':
            today = date.today()
            date_time = today.strftime("%Y%m%d")
            self.jeff_f = os.path.join(self.worm_folder, 'CLstraight_{}'.format(date_time))
            if not os.path.exists(self.jeff_f):
                os.mkdir(self.jeff_f)
        else:
            # use last one of CLstraight
            jeff_f = sorted(glob.glob1(self.worm_folder, 'CLstraight*'))[-1]
            self.jeff_f = os.path.join(self.worm_folder, jeff_f)

        self.flow_f = os.path.join(self.worm_folder, 'flow_folder')

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(self.straight_img):
            os.mkdir(self.straight_img)
            # mkdir for cline_init
        if not os.path.exists(self.neuron_img):
            os.mkdir(self.neuron_img)
        if not os.path.exists(self.cline_init_folder):
            os.mkdir(self.cline_init_folder)
        if not os.path.exists(self.cline_init_img_folder):
            os.mkdir(self.cline_init_img_folder)
        if not os.path.exists(self.cline_tp_folder):
            os.mkdir(self.cline_tp_folder)
        if not os.path.exists(self.cline_folder):
            os.mkdir(self.cline_folder)
        if not os.path.exists(self.cline_img):
            os.mkdir(self.cline_img)
        if not os.path.exists(self.ori_folder):
            os.mkdir(self.ori_folder)

        self.load_folder(worm_folder)
        # load the template worm.
        with open(template_path, 'rb') as f_temp:
            self.template = pickle.load(f_temp)
            f_temp.close()

        self.template_cline = worm_cline(self.template['cline_atlas'])

    def load_folder(self, folder):
        # load time of volumes
        self.worm_folder = folder

        hiResData = sio.loadmat(os.path.join(folder, 'hiResData.mat'))
        names = hiResData['dataAll'].dtype.names
        dict_name = dict()
        for i, name in enumerate(names): dict_name[name] = i

        stackIdx = hiResData['dataAll'][0][0][dict_name['stackIdx']]
        min_stack, max_stack = np.min(stackIdx), np.max(stackIdx)
        min_stack = max(1, min_stack)

        self.hiResData = hiResData
        self.min_stack = min_stack
        self.max_stack = max_stack

        # find volume folder and neuron folder
        self.volume_folder = os.path.join(folder, 'aligned_volume')
        self.worm_loader = worm_data_loader(folder)
        self.neuron_folder = os.path.join(folder, 'neuron_pt')

        init_file = os.path.join(folder, 'neuron_cline_init.pt')
        if not os.path.exists(init_file):
            init_v = self.init_cline()
        else:
            with open(init_file, 'rb') as fp:
                init_v = pickle.load(fp)
                fp.close()

        self.init_v = init_v

    def init_cline(self):
        # find the frames that we need human annotation for cline.
        # step 1, find the frames that we can't find neurons.(problematic volume)
        init_v = dict()
        init_v['bad_volume'] = list()
        bad_buffer = 10
        patience = bad_buffer

        for stackIdx in range(self.min_stack, self.max_stack + 1):
            # find out all bad frames
            neuron_name = os.path.join(self.neuron_folder, self.worm_folder[-15:] + '_{}.mat'.format(stackIdx))

            if os.path.exists(neuron_name):
                with open(neuron_name, "rb") as f_neu:
                    neurons = pickle.load(f_neu)
                    f_neu.close()
                if neurons is None:
                    print('stack {} is bad, neuron file is None'.format(stackIdx))
                    init_v['bad_volume'].append(stackIdx)
                    continue

                num_neu = len(neurons['pts_max'])
                if num_neu < 20:# or num_neu > 250:
                    # too few neurons.
                    print('stack:{} is bad, num neuron:{}'.format(stackIdx, num_neu))
                    init_v['bad_volume'].append(stackIdx)
                    continue

                num_plane = len(neurons['Z'])
                if num_plane < 20:
                    print('stack:{} is bad, only {} planes'.format(stackIdx, num_plane))
                    init_v['bad_volume'].append(stackIdx)
                    continue

                inten_median = np.median(neurons['max_inten'])
                print('stack:{}, median intensity of neurons:{}'.format(stackIdx, inten_median))
                if inten_median < 100:
                    print('stack:{} is bad, median intensity of neurons:{}'.format(stackIdx, inten_median))
                    init_v['bad_volume'].append(stackIdx)
                    continue

                pts_max = np.array(neurons['pts_max'])
                fig = plt.figure(1)
                ax = fig.add_subplot(111)
                ax.scatter(pts_max[:, 1], pts_max[:, 2], c='green')
                ax.set_aspect('equal')
                ax.set_xlim([0, 511])
                ax.set_ylim([0, 511])
                fig.savefig(os.path.join(self.neuron_img, 'neuron_{}.png'.format(stackIdx)))
                plt.clf()

            else:
                init_v['bad_volume'].append(stackIdx)
                print('stack:{} is bad, no neuron file'.format(stackIdx))

        start_idx = 1
        # patience_good = 12
        # patience_bad = 12
        # win = 12
        # # to avoid false positive and false negative, we add a buffer for bad volumes.
        # bad_num = np.zeros(self.max_stack + 1)
        # for stackIdx in init_v['bad_volume']:
        #     for i in range(win):
        #         if stackIdx - i >= 0:
        #             bad_num[stackIdx - i] += 1

        while start_idx in init_v['bad_volume']:
            start_idx += 1

        init_v['start_idx'] = start_idx
        init_v['end_idx'] = self.max_stack
        # check which volume needs human annotation.

        # for idx in init_v['bad_volume']:
        #     if not ((idx + 1) in init_v['bad_volume']):
        #         if idx + 1 > init_v['start_idx'] and idx + 1 <= self.max_stack:
        #             init_v['anno_volume'].append(idx + 1)
        print('check {} to see if you want to annotate some volume'.format(self.neuron_img))
        anno_h = int(input("If you want to enter annotation volume press 1, otherwise 0(only annotate first volume)"))

        if anno_h:
            init_v['anno_volume'] = [init_v['start_idx']]
            anno_idx = int(input("type the volume index you want to annotate, type negative number to end: "))
            while anno_idx >= 0:
                if anno_idx in init_v['bad_volume']:
                    print('this is a bad volume, abandom')
                else:
                    init_v['anno_volume'].append(anno_idx)
                anno_idx = int(input("type the volume index you want to annotate, type negative number to end: "))
        else:
            init_v['anno_volume'] = [init_v['start_idx']]

        self.annotate_volume(init_v['anno_volume'])

        # print('We need to annotate {} volumes.'.format(len(init_v['anno_volume'])))
        #
        # # Do the human annotation.
        # init_v['Centerline'] = dict()
        # init_v['Polygon'] = dict()
        # for idx in init_v['anno_volume']:
        #     align_file = os.path.join(self.volume_folder, self.worm_folder[-15:] + '_{}.pkl'.format(idx))
        #     with open(align_file, 'rb') as f_align:
        #         volume = pickle.load(f_align)
        #         f_align.close()
        #     print('current volume index:{}'.format(idx))
        #     cl = line_quadratic(volume['image'])
        #     init_v['Centerline'][idx] = cl.getLine()
        #
        #     poly = polygon(volume['image'])
        #     init_v['Polygon'][idx] = poly.getPolygon()

        side = int(input("If ventral cord is on left, type 1, otherwise 0 "))
        print('side is:{}'.format(side))
        init_v['side'] = side

        # save the init_v
        init_v_name = os.path.join(self.worm_folder, 'neuron_cline_init.pt')
        with open(init_v_name, 'wb') as f:
            pickle.dump(init_v, f)
            f.close()

        return init_v

    def add_template(self):
        # add some template if the current result is not satisfying.
        cur_anno = glob.glob1(self.cline_tp_folder, 'anno_tp*')
        cur_anno = sorted([int(s.split('.')[0].split('_')[-1]) for s in cur_anno])
        print('current template:{}'.format(cur_anno))

        anno_list = list()
        anno_idx = int(input("type the volume index you want to annotate, type negative number to end: "))
        while anno_idx >= 0:
            if anno_idx in self.init_v['bad_volume']:
                print('this is a bad volume, abandom')
            else:
                anno_list.append(anno_idx)
            anno_idx = int(input("type the volume index you want to annotate, type negative number to end: "))
        self.annotate_volume(anno_list)
        cur_anno = glob.glob1(self.cline_tp_folder, 'anno_tp*')
        cur_anno = sorted([int(s.split('.')[0].split('_')[-1]) for s in cur_anno])
        print('current template:{}'.format(cur_anno))


    def annotate_volume(self, anno_list):
        print('We need to annotate {} volumes.'.format(len(anno_list)))

        for idx in anno_list:
            temp_dict = dict()
            align_file = os.path.join(self.volume_folder, self.worm_folder[-15:] + '_{}.pkl'.format(idx))
            with open(align_file, 'rb') as f_align:
                volume = pickle.load(f_align)
                f_align.close()

            print('current volume index:{}'.format(idx))
            cl = line_quadratic(volume['image'])
            temp_dict['Centerline'] = cl.getLine()

            poly = polygon(volume['image'])
            temp_dict['Polygon'] = poly.getPolygon()

            temp_file = os.path.join(self.cline_tp_folder, 'anno_tp_{}.pt'.format(idx))
            with open(temp_file, 'wb') as temp_f:
                pickle.dump(temp_dict, temp_f)
                temp_f.close()

    def load_neuron_pt(self, neurons):
        # load the raw coordinate from neuron dict
        pts_out = np.array(neurons['pts_max'])
        tmp_z = neurons['Z'][pts_out[:, 0], 0] * 47.619
        pts_out = pts_out.astype(np.float)
        pts_out[:, 0] = tmp_z
        pts_out = pts_out[:, [1, 2, 0]]

        return pts_out


    def fine_tune_multiple(self, start_idx, end_idx, tp_all, flip_mode='auto'):
        # load template
        tp_path = os.path.join(self.cline_tp_folder, 'tp_{}.pt'.format(tp_all))

        with open(tp_path, 'rb') as f_tp:
            tp_dict = pickle.load(f_tp)
            f_tp.close()
        tp_all = tp_dict['tp_neurons_s']
        pt_standard = standard_neurons(tp_all, self.scale)
        tp_all_cpd = pt_standard.transform(tp_all)

        tp_idx = None

        # get flip orientation frames.
        flip_list = glob.glob1(self.ori_folder, '*.pt')
        flip_idx = sorted([int(idx.split('.')[0]) for idx in flip_list])
        flip_back = np.array([False] * (self.max_stack + 1))
        if len(flip_idx) > 0:
            flip_diff = np.diff(np.array(flip_idx))
            back_idx = np.where(flip_diff < 18)[0]
            # frames close to the flip we need to check
            for back_idx_i in back_idx:
                flip_back[flip_idx[back_idx_i]:flip_idx[back_idx_i+1]] = True


        for idx in range(start_idx, end_idx):
            cline_init_file = os.path.join(self.cline_init_folder, 'init{}.pt'.format(idx))
            neuron_name = os.path.join(self.neuron_folder, self.worm_folder[-15:] + '_{}.mat'.format(idx))
            if os.path.exists(neuron_name):
                with open(neuron_name, "rb") as f_neu:
                    neurons = pickle.load(f_neu)
                    f_neu.close()
            else:
                continue

            if os.path.exists(cline_init_file):
                with open(cline_init_file, 'rb') as fp_c_init:
                    cline_dict = pickle.load(fp_c_init)
                    fp_c_init.close()
            else:
                continue

            tp_idx_cur = cline_dict['template_idx']
            if tp_idx_cur != tp_idx:
                tp_path = os.path.join(neu_cline.cline_tp_folder, 'tp_{}.pt'.format(tp_idx_cur))
                tp_idx = tp_idx_cur
                with open(tp_path, 'rb') as f_tp:
                    tp_dict = pickle.load(f_tp)
                    f_tp.close()
                tp = tp_dict['tp_neurons_s']
                # pt_standard = standard_neurons(tp, self.scale)
                # tp_cpd = pt_standard.transform(tp)

            anno_file_dir = os.path.join(self.ori_folder, 'anno.file')
            if flip_mode == 'auto':
                cline_f = worm_cline(cline_dict['cline'])
                if os.path.exists(anno_file_dir):
                    with open(anno_file_dir, 'rb') as fp:
                        anno_d = pickle.load(fp)
                        fp.close()
                else:
                    anno_d = np.array([False] * (self.max_stack + 1))

                if anno_d[idx]:
                    print('Annotation: flip at volume:{}'.format(idx))
                    cline_f = worm_cline(cline_dict['cline'][::-1, :])
                else:
                    if flip_back[idx]:
                        # check if we need to flip the centerline.
                        pts_ori_s = cline_f.straighten(cline_dict['pts'])
                        pts_ori_s[:, 2] -= np.median(pts_ori_s[:, 2]) + np.median(tp[:, 2])
                        pts_inv_s = np.copy(pts_ori_s)
                        pts_inv_s[:, 0] = cline_f.length - pts_inv_s[:, 0]
                        pts_inv_s[:, 1] *= -1

                        pt_standard = standard_neurons(tp, self.scale)
                        tp_cpd = pt_standard.transform(tp)
                        pts_ori_s_cpd = pt_standard.transform(pts_ori_s)
                        pts_inv_s_cpd = pt_standard.transform(pts_inv_s)
                        ori_trans = register_nonrigid(tp_cpd, pts_ori_s_cpd, w=0.1, lamb=self.lamb,
                                                     beta=self.beta)
                        inv_trans = register_nonrigid(tp_cpd, pts_inv_s_cpd, w=0.1, lamb=self.lamb,
                                                     beta=self.beta)
                        ori_dis = find_hung_match_average(ori_trans, tp_cpd, dim=None)
                        inv_dis = find_hung_match_average(inv_trans, tp_cpd, dim=None)
                        if np.mean(ori_dis) > np.mean(inv_dis):
                            print('flip at volume:{}'.format(idx))
                            cline_f = worm_cline(cline_dict['cline'][::-1, :])
            elif flip_mode == 'anno':
                if os.path.exists(anno_file_dir):
                    with open(anno_file_dir, 'rb') as fp:
                        anno_d = pickle.load(fp)
                        fp.close()
                    if anno_d['flip_list'][idx]:
                        cline_f = worm_cline(cline_dict['cline'][::-1, :])
                    else:
                        cline_f = worm_cline(cline_dict['cline'])
                else:
                    cline_f = worm_cline(cline_dict['cline'])
            elif flip_mode == 'human':
                print('Annotation: flip at volume:{}'.format(idx))

                cline_f = worm_cline(cline_dict['cline'][::-1, :])
            else:
                cline_f = worm_cline(cline_dict['cline'])


            pts_ori = cline_dict['pts']
            cline, keep_idx = self.fine_tune(pts_ori, cline_f, tp, mask_check=True)

            # cpd translation align.
            jeff_dict = dict()
            cline_f_new = worm_cline(cline)

            pts_s_all = cline_f_new.straighten(pts_ori)
            # get rid of autofluorescence
            keep_idx = cline_f_new.autofluorescence_mask(pts_s_all, np.array(neurons['mean_inten']), keep_idx)

            if len(keep_idx) < 10:
                print('Only found {} Neurons, abandom'.format(len(keep_idx)))
                continue

            pts_s_ori = np.copy(pts_s_all)
            pts_s_ori = pts_s_ori[keep_idx, :]

            pts_s = np.copy(pts_s_ori)
            pts_s[:, 2] += -np.median(pts_s[:, 2]) + np.median(tp[:, 2])

            pts_s_cpd = pt_standard.transform(pts_s)
            pts_rigid_cpd, ts = register_translation(tp_all_cpd, pts_s_cpd, w=0.1)

            pts_rigid = pt_standard.transform_inv(pts_rigid_cpd)

            _, dis_min = find_min_match(tp, pts_rigid)
            tp_ratio = np.sum(dis_min < 10) / tp.shape[0]

            ts_align_s = pts_rigid[0, :] - pts_s_ori[0, :]

            z_median = np.median(pts_rigid[:, 2])
            pts_rigid[:, 2] -= z_median
            pts_rigid[:, 2] = pts_rigid[:, 2] / 47.619 * 30

            size_s = [401, 168, 128]
            pts_rigid += np.array([[size_s[0] // 6, size_s[1] // 2, size_s[2] // 2]])

            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.scatter(pts_rigid[:, 0], pts_rigid[:, 1], c='blue')
            ax.set_aspect('equal')
            ax.set_xlim([0, size_s[0]])
            ax.set_ylim([0, size_s[1]])
            fig.savefig(os.path.join(self.straight_img, 'straight_{}.png'.format(idx)))
            plt.clf()

            # make sure pts_rigid lie in the size_s
            mask = (pts_rigid[:, 0] >= 0) * (pts_rigid[:, 0] < size_s[0] - 1)
            mask *= (pts_rigid[:, 1] >= 0) * (pts_rigid[:, 1] < size_s[1] - 1)
            mask *= (pts_rigid[:, 2] >= 0) * (pts_rigid[:, 2] < size_s[2] - 1)
            mask_idx = np.where(mask)[0]

            pts_rigid = pts_rigid[mask_idx, :]
            keep_idx = keep_idx[mask_idx]

            num_neu = len(keep_idx)
            Rinten = np.array(neurons['mean_inten'], dtype=np.float)
            Rinten_all = np.copy(Rinten)
            Rinten = Rinten[keep_idx].reshape(num_neu, 1)

            Volume = np.array(neurons['area'], dtype=np.float)
            Volume_all = np.copy(Volume)
            Volume = Volume[keep_idx].reshape(num_neu, 1)

            volume_name = os.path.join(self.volume_folder, self.worm_folder[-15:] + '_{}.pkl'.format(idx))
            with open(volume_name, 'rb') as f_volume:
                volume = pickle.load(f_volume)
                f_volume.close()

            z_median = (z_median - ts_align_s[2]) / 47.619
            # get z index for each plane.
            z_plane = (np.arange(size_s[2]) - size_s[2] // 2) / 30 + z_median
            z_list = list()
            z_min, z_max = np.min(neurons['Z']), np.max(neurons['Z'])
            for z in z_plane:
                if z < z_min:
                    z_list.append(-2)
                elif z > z_max:
                    z_list.append(-1)
                else:
                    z_list.append(np.argmin(np.abs(z - neurons['Z'])))

            pts_show = np.rint(pts_rigid)

            map_cord = straighten_img(cline_f_new, size=size_s)
            # straighten image
            img_s_jeff = get_straighten_img(map_cord, volume['image'], z_list, ts_align_s, pts_show, bordermode='const')
            img_s_jeff = img_s_jeff.astype(np.float32)
            img_file = os.path.join(self.jeff_f, 'image{:05}.tif'.format(idx))
            io.imsave(img_file, img_s_jeff)

            # flow for unstraightened image
            with open(os.path.join(self.flow_f, self.worm_folder[-15:] + "_{}.pkl".format(idx)), "rb") as f_flow:
                flow_dict = pickle.load(f_flow)
                f_flow.close()

            num_z = volume['image'].shape[0]
            dsizex = volume['image'].shape[1]
            dsizey = volume['image'].shape[2]
            #flow_scale = volume['image'].shape[0]
            flow_x_list = list()
            flow_y_list = list()
            flow_z_list = list()
            for i in range(num_z):
                if i in flow_dict:
                    flow_cur = np.copy(flow_dict[i])
                    hf, wf = flow_cur.shape[:2]
                    # flow         = -flow
                    flow_cur[:, :, 0] -= np.arange(wf)
                    flow_cur[:, :, 1] -= np.arange(hf)[:, np.newaxis]

                    flow_scale = volume['image'].shape[1] / flow_cur.shape[0]
                    flow_full = cv2.resize(src=flow_cur * flow_scale, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                                           interpolation=cv2.INTER_AREA)
                else:
                    #print('center:{}'.format(i))
                    flow_full = np.zeros((dsizex, dsizey, 2))
                hf, wf = flow_full.shape[:2]
                # flow         = -flow
                flow_full[:, :, 0] += np.arange(wf)
                flow_full[:, :, 1] += np.arange(hf)[:, np.newaxis]
                # on purpose flip x and y back
                flow_x_list.append(flow_full[:, :, 1])
                flow_y_list.append(flow_full[:, :, 0])
                flow_z_list.append(i * np.ones(flow_full.shape[:2]))


            trans_x = get_straighten_img(map_cord, np.array(flow_x_list), z_list, ts_align_s, pts_show)
            trans_y = get_straighten_img(map_cord, np.array(flow_y_list), z_list, ts_align_s, pts_show)
            trans_z = get_straighten_img(map_cord, np.array(flow_z_list), z_list, ts_align_s, pts_show)

            # get unstraightened coordinate.
            raw_p_idx = np.rint(pts_rigid).astype(np.uint16)
            raw_p = np.array([trans_x[raw_p_idx[:, 2], raw_p_idx[:, 0], raw_p_idx[:, 1]],
                              trans_y[raw_p_idx[:, 2], raw_p_idx[:, 0], raw_p_idx[:, 1]],
                              trans_z[raw_p_idx[:, 2], raw_p_idx[:, 0], raw_p_idx[:, 1]]])
            raw_p = raw_p.T
            trans_x = np.rint(np.moveaxis(trans_x, [0, 1, 2], [2, 0, 1])).astype(np.uint16)
            trans_y = np.rint(np.moveaxis(trans_y, [0, 1, 2], [2, 0, 1])).astype(np.uint16)
            trans_z = np.rint(np.moveaxis(trans_z, [0, 1, 2], [2, 0, 1])).astype(np.uint16)
            # get base image correct
            mask_img = np.zeros(volume['image'].shape)
            for b_idx, bbox in enumerate(neurons['bbox']):
                mask_img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = neurons['mask'][b_idx]

            base_img = get_straighten_img(map_cord, mask_img, z_list, ts_align_s, pts_show, bordermode='const')
            base_img = base_img > 0.1
            base_img = np.moveaxis(base_img, [0, 1, 2], [2, 0, 1])

            #Multi_Slice_Viewer(base_img)
            jeff_file = os.path.join(self.jeff_f, 'pointStats{:05}.mat'.format(idx))
            jeff_dict['ts_align_s'] = ts_align_s
            jeff_dict['pts_s'] = pts_s_all
            jeff_dict['straightPoints'] = pts_rigid.astype(np.float) + 1
            jeff_dict['keep_idx'] = keep_idx
            jeff_dict['pts'] = pts_ori
            jeff_dict['tp_ratio'] = tp_ratio
            jeff_dict['Rintensities'] = Rinten
            jeff_dict['Volume'] = Volume
            jeff_dict['pointIdx'] = np.arange(1, num_neu + 1).astype(np.float).reshape(num_neu, 1)
            jeff_dict['rawPoints'] = raw_p + 1
            jeff_dict['stackIdx'] = idx
            jeff_dict['transformx'] = trans_x + 1
            jeff_dict['transformy'] = trans_y + 1
            jeff_dict['transformz'] = trans_z + 1
            jeff_dict['baseImg'] = base_img

            jeff_out_dict = dict()
            jeff_out_dict['pointStats'] = jeff_dict

            sio.savemat(jeff_file, jeff_out_dict, do_compression=True)

            # plt.scatter(pts_s[:, 0], pts_s[:, 1], c='red')
            # plt.scatter(pts_rigid[:, 0], pts_rigid[:, 1], c='green')
            # plt.scatter(tp[:, 0], tp[:, 1], c='yellow')
            # plt.show()
            # generate the file for jeff's registration.

            cline_dict_new = dict()
            cline_dict_new['cline'] = cline
            cline_dict_new['pts'] = cline_dict['pts']
            cline_dict_new['pts_s'] = pts_s_all
            cline_dict_new['keep_idx'] = keep_idx
            cline_dict_new['Riten'] = Rinten_all
            cline_dict_new['Volume'] = Volume_all
            cline_dict_new['tp_ratio'] = tp_ratio
            cline_file = os.path.join(self.cline_folder, 'cline{}.pt'.format(idx))
            with open(cline_file, 'wb') as fp_c:
                pickle.dump(cline_dict_new, fp_c)
                fp_c.close()

            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.scatter(pts_ori[keep_idx, 0], pts_ori[keep_idx, 1], c='green')
            ax.scatter(pts_ori[:, 0], pts_ori[:, 1], c='red', marker='+')
            # ax.scatter(tp_trans[:, 0], tp_trans[:, 1], c='yellow')
            # ax.scatter(Neurons_tp_proj_rigid[:, 0], Neurons_tp_proj_rigid[:, 1], c='blue')

            ax.scatter(cline_dict['cline'][:, 0], cline_dict['cline'][:, 1], c='red')
            ax.scatter(cline_dict['cline'][0, 0], cline_dict['cline'][0, 1], c='blue', marker='+')
            ax.scatter(cline[:, 0], cline[:, 1], c='blue')
            ax.scatter(cline[0, 0], cline[0, 1], c='yellow', marker='+')
            ax.set_aspect('equal')
            ax.set_xlim([0, 511])
            ax.set_ylim([0, 511])
            # ax.scatter(tp_rigid_cline_new[:, 0], tp_rigid_cline_new[:, 1], c='black')
            # ax.scatter(tp_rigid_cline_new[0, 0], tp_rigid_cline_new[0, 1], c='red', marker='+')
            fig.savefig(os.path.join(self.cline_img, 'cline_{}.png'.format(idx)))
            plt.clf()


    def fine_tune(self, pts_ori, cline_f, tp, mask_check=True, max_iter=5):
        # fine tune the centerline with template neurons.
        cline_diff = 1e5
        i_iter = 0
        #cline_ori = cline_f.Centerline
        #keep_idx = np.arange(pts_ori.shape[0])
        cline1 = cline_f.Centerline

        while cline_diff > 100 and i_iter < max_iter:
            cline_old = cline_f.Centerline
            if mask_check:
                keep_idx = cline_f.mask_with_template(pts_ori, tp, straight=True)
                pts = np.copy(pts_ori[keep_idx, :])
            else:
                pts = np.copy(pts_ori)
                keep_idx = np.arange(pts_ori.shape[0])

            if len(keep_idx) == 0:
                break

            bad_ratio = 1 - len(keep_idx) / pts_ori.shape[0]
            bad_ratio = max(0.1, min(bad_ratio, 0.25))
            w = bad_ratio

            tp_proj = cline_f.project(tp)
            tp_proj[:, 2] += -np.median(tp_proj[:, 2]) + np.median(pts[:, 2])

            pt_standard = standard_neurons(pts, self.scale)
            pts_cpd = pt_standard.transform(pts)
            tp_proj_cpd = pt_standard.transform(tp_proj)

            tp_trans = register_nonrigid(tp_proj_cpd, pts_cpd, w=w, lamb=self.lamb, beta=self.beta)

            tp_trans = pt_standard.transform_inv(tp_trans)
            cline1 = cline_f.cpd_cline_update(tp_trans, pts)
            cline_diff = np.sum(np.sqrt(np.sum((cline1[:, :2] - cline_old[:, :2]) ** 2, axis=1)))
            #print(cline_diff)
            i_iter += 1
            cline_f = worm_cline(cline1)

        if mask_check:
            keep_idx = cline_f.mask_with_template(pts_ori, tp, straight=True)
        else:
            keep_idx = np.arange(pts_ori.shape[0])

        # plt.scatter(cline_ori[:, 0], cline_ori[:, 1], c='red')
        # plt.scatter(cline1[:, 0], cline1[:, 1], c='green')
        # plt.scatter(pts_ori[:, 0], pts_ori[:, 1], c='blue')
        # plt.scatter(pts[:, 0], pts[:, 1], c='red', marker='+')
        # plt.show()

        return cline1, keep_idx

    def setup_template(self, Cl, cur_pts, Polygon=None, keep_z=False):
        # set up template
        cur_cline = Cl[:, [1, 0]]
        # for human annotated data, degree is 2.
        cur_cline = smooth_2d_pts(cur_cline, degree=3, num_point_out=100)
        template_cline = worm_cline(cur_cline)
        Neurons_tp = np.copy(cur_pts)
        # plt.scatter(cur_cline[:, 0], cur_cline[:, 1], c='red')
        # plt.scatter(cur_pts[:, 0], cur_pts[:, 1], c='blue')
        # plt.show()

        if Polygon is None:
            keep_idx = template_cline.mask_neurons(Neurons_tp, straight=True)
            mask_check = True
        else:
            new_p = [(p[1], p[0]) for p in Polygon]
            polygonMask = geom.polygon.Polygon(new_p)
            keep_idx = list()
            for n in range(Neurons_tp.shape[0]):
                neuron = Neurons_tp[n]
                point = geom.Point(neuron[0], neuron[1])
                if polygonMask.contains(point):
                    keep_idx.append(n)
            keep_idx = np.array(keep_idx)
            Neurons_tp = Neurons_tp[keep_idx, :]
            keep_idx = np.arange(Neurons_tp.shape[0])
            mask_check = False

        # Neurons_tp_s is the template worm(another standard worm)
        if self.template_align == 'fine':
            new_cline, keep_idx = self.fine_tune(Neurons_tp, template_cline, self.Neurons_tp_s, mask_check=mask_check, max_iter=5)
        # or only make translation.
        elif self.template_align == 'translation':
            pt_standard = standard_neurons(Neurons_tp, self.scale)
            Neurons_tp_cpd = pt_standard.transform(Neurons_tp)
            Neurons_tp_stand_cpd = pt_standard.transform(template_cline.project(self.Neurons_tp_s))
            _, ts = register_translation(Neurons_tp_stand_cpd, Neurons_tp_cpd, w=0.1)
            new_cline = np.copy(cur_cline)
            new_cline[:, 0] -= self.scale * ts[0]
            new_cline[:, 1] -= self.scale * ts[1]
        else:
            new_cline = cur_cline

        template_cline = worm_cline(new_cline)
        Neurons_tp = Neurons_tp[keep_idx, :]
        Neurons_tp_s = template_cline.straighten(Neurons_tp)
        if not keep_z:
            Neurons_tp[:, 2] -= np.median(Neurons_tp[:, 2])
            Neurons_tp_s[:, 2] = Neurons_tp[:, 2]
        return template_cline, Neurons_tp, Neurons_tp_s


    def find_cline(self, start_idx, end_idx):
        # find the centerline for the worm across time.
        # take use of the continuity in time domain.

        # load template(human annotated, but from a gold standard dataset)
        Neurons_atlas = self.template['Neuron_atlas']
        cline_atlas = self.template['cline_atlas']

        self.Neurons_tp = Neurons_atlas
        self.Neurons_tp_s = self.template_cline.straighten(self.Neurons_tp)

        self.ref_pt_idx = 20
        self.ref_s = self.Neurons_tp_s[self.ref_pt_idx, 0]

        # get annotated volumes
        anno_volume = glob.glob1(self.cline_tp_folder, 'anno_tp*')
        anno_volume = sorted([int(s.split('.')[0].split('_')[-1]) for s in anno_volume])
        template_idx = anno_volume[0]

        if self.init_v['side'] != self.template['side']:
            self.Neurons_tp_s[:, [1, 2]] *= -1
        Neurons_tp_s_median = np.median(self.Neurons_tp_s, axis=0)
        self.Neurons_tp_s[:, 2] -= Neurons_tp_s_median[2]

        patience_turn = 0 # patience for flip head and tail.
        # go throught time, update centerline.
        for idx in range(start_idx, end_idx):
            print('current idx:{}'.format(idx))

            neuron_name = os.path.join(self.neuron_folder, self.worm_folder[-15:] + '_{}.mat'.format(idx))
            if os.path.exists(neuron_name):
                with open(neuron_name, "rb") as f_neu:
                    neurons = pickle.load(f_neu)
                    f_neu.close()
            else:
                continue

            # get the centerline from last frame or annotation.
            volume_name = os.path.join(self.volume_folder, self.worm_folder[-15:] + '_{}.pkl'.format(idx))
            with open(volume_name, 'rb') as f_volume:
                volume = pickle.load(f_volume)
                f_volume.close()

            image = np.mean(volume['image'], axis=0)
            image = image - image.mean()
            image[image < 0] = 0

            if (idx in self.init_v['bad_volume']):
                cur_cline = last_cline
                new_cline = cur_cline
                good_idx = []
                pts_s = []
                cpd_align = False
            else:
                cur_pts = self.load_neuron_pt(neurons)

                if idx in anno_volume:
                    # cur_cline = self.init_v['Centerline'][idx]
                    # cur_cline = cur_cline[:, [1, 0]]
                    # # for human annotated data, degree is 2.
                    # cur_cline = smooth_2d_pts(cur_cline, degree=3, num_point_out=100)
                    # template_cline = worm_cline(cur_cline)
                    # Neurons_tp = np.copy(cur_pts)
                    #
                    # Neurons_tp_s = template_cline.straighten(Neurons_tp)
                    # keep_idx = template_cline.mask_neurons(Neurons_tp_s, straight=False)
                    #
                    # Neurons_tp = Neurons_tp[keep_idx, :]
                    # Neurons_tp_s = Neurons_tp_s[keep_idx, :]
                    # Neurons_tp[:, 2] -= np.median(Neurons_tp[:, 2])
                    # Neurons_tp_s[:, 2] = Neurons_tp[:, 2]

                    # renew the template.
                    anno_file = os.path.join(self.cline_tp_folder, 'anno_tp_{}.pt'.format(idx))
                    with open(anno_file, 'rb') as anno_f:
                        anno_tmp = pickle.load(anno_f)
                        anno_f.close()
                    template_idx = idx
                    template_cline, Neurons_tp, Neurons_tp_s = self.setup_template(anno_tmp['Centerline'],
                                                                                   cur_pts, anno_tmp['Polygon'])
                    cur_cline = template_cline.Centerline
                    # save the template
                    template_dict = dict()
                    template_dict['tp_cline'] = cur_cline
                    template_dict['tp_neurons'] = Neurons_tp
                    template_dict['tp_neurons_s'] = Neurons_tp_s
                    template_file = os.path.join(self.cline_tp_folder, 'tp_{}.pt'.format(idx))
                    with open(template_file, 'wb') as f_tp:
                        pickle.dump(template_dict, f_tp)
                        f_tp.close()

                    fig = plt.figure(1)
                    ax = fig.add_subplot(111)
                    ax.scatter(cur_cline[:, 0], cur_cline[:, 1], c='green')
                    ax.scatter(cur_cline[0, 0], cur_cline[0, 1], c='red', marker='+')
                    ax.scatter(Neurons_tp[:, 0], Neurons_tp[:, 1], c='blue')
                    ax.set_aspect('equal')
                    ax.set_xlim([0, 511])
                    ax.set_ylim([0, 511])
                    fig.savefig(os.path.join(self.cline_tp_folder, 'temp_{}.png'.format(template_idx)))
                    plt.clf()

                    cpd_align = True
                else:
                    # use optic flow to transform cline from previous frame.
                    show = False
                    # move point by using the optical flow estimated from image.
                    pts_guess = align_centerline(image, last_image, last_pts_good, show)

                    cur_cline = last_cline_f.cpd_cline_update(last_pts_good, pts_guess)
                    # plt.scatter(cur_pts[:, 0], cur_pts[:, 1], c='red')
                    # plt.scatter(last_cline[:, 0], last_cline[:, 1], c='blue')
                    # plt.scatter(cur_cline[:, 0], cur_cline[:, 1], c='green')
                    # #plt.scatter(last_cline_f.Centerline[:, 0], last_cline_f.Centerline[:, 1], c='black')
                    # plt.show()
                    cpd_align = True

            if cpd_align:
                cline_f = worm_cline(cur_cline)
                # mask out neurons that are not accept(outliers)
                tp_s = np.copy(Neurons_tp_s)
                good_idx = cline_f.mask_with_template(cur_pts, tp_s, straight=True, show=False)

                if len(good_idx) < 10:
                    good_idx = np.arange(cur_pts.shape[0])

                bad_ratio = 1 - len(good_idx) / cur_pts.shape[0]
                bad_ratio = max(0.1, min(bad_ratio, 0.25))
                w = bad_ratio
                w_nonrigid = bad_ratio

                cur_pts_good = cur_pts[good_idx, :]

                last_image = image

                # standardize coordinate for cpd
                pt_standard = standard_neurons(cur_pts_good, self.scale)
                cur_pts_good_cpd = pt_standard.transform(cur_pts_good)

                # use cpd rigid transformation to get template match current volume
                cline_head_f = cline_f.head_orient_cline(method='combine')
                #cline_head_f = cline_f

                Neurons_tp_proj = cline_head_f.project(Neurons_tp_s)
                #cur_median = np.median(cur_pts_good, axis=0)
                Neurons_tp_proj[:, 2] += -np.median(Neurons_tp_proj[:, 2]) + np.median(cur_pts_good[:, 2])
                tp_proj_cpd = pt_standard.transform(Neurons_tp_proj)

                # use rotation.

                cur_pts_rigid, r, ts, sigma2_new = register_rigid(tp_proj_cpd, cur_pts_good_cpd, w=w,
                                                                          fix_scale=True)

                # not use rotation when rotate more than 90 degree

                if r[0][0] > 0:
                    # cur_pts_rigid, ts = register_translation(tp_proj_cpd, cur_pts_good_cpd, w=w)
                    #
                    # r = np.eye(3)
                    m = tp_proj_cpd.shape[0]
                    Neurons_tp_proj_rigid = np.dot((tp_proj_cpd - np.matlib.repmat(np.transpose(ts), m, 1)), r)


                    tp_rigid_cline = cline_head_f.Centerline
                    tp_rigid_cline_cpd = pt_standard.transform(tp_rigid_cline)
                    m_tp_cline = tp_rigid_cline.shape[0]
                    r_2d = r[:2, :2]
                    ts_2d = ts[:2, :]
                    tp_rigid_cline_new_cpd = np.dot((tp_rigid_cline_cpd - np.matlib.repmat(np.transpose(ts_2d), m_tp_cline, 1)), r_2d)
                    tp_rigid_cline_new = pt_standard.transform_inv(tp_rigid_cline_new_cpd)
                    cline_head_f_rigid = worm_cline(tp_rigid_cline_new)
                else:
                    # not use any rigid transformation.
                    Neurons_tp_proj_rigid = tp_proj_cpd
                    cline_head_f_rigid = cline_head_f

                # tp_trans = register_nonrigid(cur_pts_good_cpd, Neurons_tp_proj_rigid, w=w_nonrigid, lamb=lamb,
                #                              beta=beta)
                tp_trans = register_nonrigid(Neurons_tp_proj_rigid, cur_pts_good_cpd, w=w_nonrigid, lamb=self.lamb, beta=self.beta)

                tp_trans = pt_standard.transform_inv(tp_trans)
                Neurons_tp_proj_rigid = pt_standard.transform_inv(Neurons_tp_proj_rigid)

                cline1 = cline_head_f_rigid.cpd_cline_update(tp_trans, cur_pts_good)

                new_cline = cline1
                new_cline_f = worm_cline(new_cline)
                new_cline = new_cline_f.cut_head(cur_pts_good)
                new_cline_f = worm_cline(new_cline)
                good_idx = new_cline_f.mask_with_template(cur_pts, tp_s, straight=True)
                cur_pts_good = cur_pts[good_idx, :]
                # post analysis

                # cur_pts_s = new_cline_f.straighten(cur_pts)
                # Neurons_tp_proj_rigid_s = new_cline_f.straighten(Neurons_tp_proj_rigid)
                #
                #turn = new_cline_f.evaluate_orientation(cur_pts, Neurons_tp_s, good_idx, self.init_v['side'])
                #turn = new_cline_f.evaluate_orientation_hung(cur_pts, Neurons_tp_s, good_idx, self.init_v['side'])

                turn = new_cline_f.evaluate_orientation(cur_pts, Neurons_tp_s, good_idx, self.init_v['side'])
                #print(turn, turn2)
                #turn = turn * turn2
                #print(turn)
                if turn:
                    patience_turn += 1
                else:
                    patience_turn = 0

                if patience_turn > 1:
                    flip_start = max(start_idx, idx - patience_turn + 1)
                    for flip_idx in range(flip_start, idx+1):
                        flip_file = os.path.join(self.ori_folder, '{}.pt'.format(flip_idx))
                        with open(flip_file, 'wb') as f_flip:
                            tmp = dict()
                            pickle.dump(tmp, f_flip)
                            f_flip.close()
                    patience_turn = 0
                    new_cline = cline1[::-1, :]
                    new_cline_f = worm_cline(new_cline)

                    good_idx = new_cline_f.mask_with_template(cur_pts, tp_s, straight=True)
                    cur_pts_good = cur_pts[good_idx, :]
                print('patience:{}'.format(patience_turn))
                last_cline = new_cline
                new_cur_pts_good = cur_pts_good
                last_pts_good = new_cur_pts_good
                last_cline_f = new_cline_f


                # save new cline and other
                cline_dict = dict()
                cline_dict['cline'] = new_cline
                cline_dict['pts'] = cur_pts

                cline_dict['flip_orient'] = turn
                cline_dict['template_idx'] = template_idx
                cline_init_file = os.path.join(self.cline_init_folder, 'init{}.pt'.format(idx))
                with open(cline_init_file, 'wb') as fp_c_init:
                    pickle.dump(cline_dict, fp_c_init)
                    fp_c_init.close()

                # plot the cline for test.
                cline_folder = self.cline_init_img_folder
                if not os.path.exists(cline_folder):
                    os.mkdir(cline_folder)
                fig = plt.figure(1)
                ax = fig.add_subplot(111)
                ax.scatter(cur_pts_good[:, 0], cur_pts_good[:, 1], c='green')
                ax.scatter(cur_pts[:, 0], cur_pts[:, 1], c='red', marker='+')
                #ax.scatter(tp_trans[:, 0], tp_trans[:, 1], c='yellow')
                #ax.scatter(Neurons_tp_proj_rigid[:, 0], Neurons_tp_proj_rigid[:, 1], c='blue')
                ax.scatter(new_cline[:, 0], new_cline[:, 1], c='blue')
                ax.scatter(new_cline[0, 0], new_cline[0, 1], c='red', marker='+')
                ax.set_aspect('equal')
                ax.set_xlim([0, 511])
                ax.set_ylim([0, 511])
                #ax.scatter(tp_rigid_cline_new[:, 0], tp_rigid_cline_new[:, 1], c='black')
                #ax.scatter(tp_rigid_cline_new[0, 0], tp_rigid_cline_new[0, 1], c='red', marker='+')
                fig.savefig(os.path.join(cline_folder, 'cline_{}.png'.format(idx)))
                plt.clf()

def straighten_img(cline_f, size=[401, 168, 128]):
    # get the point along centerline.
    pts_s = np.zeros((size[0], 2))
    pts_s[:, 0] = np.arange(size[0]) - size[0] // 6

    pts_cline = cline_f.project(pts_s)
    dir_list = [cline_f.get_dir(s) for s in pts_s[:, 0]]

    dir_array = np.array(dir_list)
    w_arr = np.arange(size[1]) - size[1] // 2

    x_cord = pts_cline[:, 0:1] - w_arr[np.newaxis, :] * dir_array[:, 1:2]
    y_cord = pts_cline[:, 1:2] + w_arr[np.newaxis, :] * dir_array[:, 0:1]
    map_cord = np.array([x_cord, y_cord])
    map_cord = np.moveaxis(map_cord, [0, 1, 2], [2, 0, 1])
    map_cord = map_cord[:, :, [1, 0]].astype(np.float32)
    return map_cord


def get_straighten_img(map_cord, volume_img, z_list, ts, pts, bordermode='rep'):
    # # get the straightened image with respect to the cline.
    ts = np.rint(ts).astype(np.int64)
    num_z = volume_img.shape[0]
    img_s = list()
    if bordermode == 'rep':
        mode = cv2.BORDER_REPLICATE
    else:
        mode = cv2.BORDER_CONSTANT
    for z in range(num_z):
        tmp = cv2.remap(volume_img[z], map_cord, None, cv2.INTER_LINEAR, borderMode=mode)
        tmp = np.roll(tmp, shift=(ts[0], ts[1]), axis=(0, 1))
        img_s.append(tmp)

    img_s_jeff = list()
    for z in z_list:
        if z == -2:
            img_s_jeff.append(img_s[0])
        else:
            img_s_jeff.append(img_s[z])

    img_s_jeff = np.array(img_s_jeff)
    #Multi_Slice_Viewer(img_s_jeff, pts)
    # Multi_Slice_Viewer(volume_img)
    return img_s_jeff




def align_centerline(image, last_image, cline_init, show=False):
    # align the image to the former image
    # initialize template if it is not initiated yet
    cline_atlas = np.copy(cline_init)
    cline_atlas = np.rint(np.clip(cline_atlas, a_min=0, a_max=511)).astype(np.uint16)

    def deepflow_align(image, last_image, cline_atlas, scale=4):

        dsizex = image.shape[0]
        dsizey = image.shape[1]

        image_half = cv2.resize(src=image, dsize=(dsizey // scale, dsizex // scale), fx=0.0, fy=0.0, \
                              interpolation=cv2.INTER_AREA)
        last_image_half = cv2.resize(src=last_image, dsize=(dsizey // scale, dsizex // scale), fx=0.0, fy=0.0, \
                              interpolation=cv2.INTER_AREA)

        # # # here I want to first try use phase correlate to calculate the translation.
        # pad_l = 5
        # border_v = np.mean(last_image_half)
        # last_image_half = cv2.GaussianBlur(last_image_half, (pad_l, pad_l), cv2.BORDER_REFLECT)
        # image_half = cv2.GaussianBlur(image_half, (pad_l, pad_l), cv2.BORDER_REFLECT)
        pad_l = 5
        last_image_half_pad = cv2.copyMakeBorder(last_image_half, pad_l, pad_l, pad_l, pad_l, cv2.BORDER_CONSTANT, value=0)
        image_half_pad = cv2.copyMakeBorder(image_half, pad_l, pad_l, pad_l, pad_l, cv2.BORDER_CONSTANT, value=0)

        dp = cv2.phaseCorrelate(last_image_half_pad, image_half_pad)
        if dp[1] < 0.3:
            dp = ((0, 0), 0)
        translation_matrix = np.float32([[1, 0, dp[0][0]], [0, 1, dp[0][1]]])
        num_rows, num_cols = last_image_half.shape[:2]
        last_image_half_t = cv2.warpAffine(last_image_half, translation_matrix, (num_cols,num_rows))

        # test_pt = np.array([[num_rows // 2, num_cols//2, 0], [num_rows // 2 + dp[0][1], num_cols // 2 + dp[0][0], 0]])
        # Multi_Slice_Viewer(np.array([last_image_half, last_image_half_t, image_half]), test_pt)
        #
        # # check the linear polar image
        # def get_linearpolar(image):
        #     sx, sy = image.shape
        #     out = cv2.linearPolar(image, (sx/2, sy/2), np.sqrt(sx ** 2 + sy ** 2) / 2,
        #                           cv2.WARP_FILL_OUTLIERS)
        #     return out
        # last_polar = get_linearpolar(last_image_half)
        # cur_polar = get_linearpolar(image_half)
        # dp = cv2.phaseCorrelate(last_polar, cur_polar)
        # print(dp)
        # Multi_Slice_Viewer(np.array([last_polar, cur_polar]))


        of_estim = cv2.optflow.createOptFlow_DeepFlow()
        flow = of_estim.calc(last_image_half_t, image_half, None)
        #flow = of_estim.calc(last_image_half, image_half, None)
        flow = flow * scale
        flow_full = cv2.resize(src=flow, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                           interpolation=cv2.INTER_AREA)

        #image_warp = warp_flow(image, flow_full)

        #Multi_Slice_Viewer(np.array([image, image_warp, last_image]))
        #flow_full += scale / 2
        #flow_full_show = np.moveaxis(flow_full, [0, 1, 2], [1, 2, 0]) * 10
        #Multi_Slice_Viewer(np.moveaxis(flow_full, [0, 1, 2], [1, 2, 0]), part=False, fig_idx=1)
        #Multi_Slice_Viewer(np.array([image, last_image]), part=False, fig_idx=2)
        #image_stack = np.concatenate((np.array([image, last_image]), np.abs(flow_full_show)), axis=0)
        #Multi_Slice_Viewer(image_stack)

        #flow_coord = np.copy(flow_full)
        flow_x = flow_full[:, :, 1]
        flow_y = flow_full[:, :, 0]
        cline_atlas = cline_atlas.astype(np.int32)
        #print(cline_atlas[:10, 0])
        cline_atlas[:, 0] += int(dp[0][1] * scale)
        cline_atlas[:, 1] += int(dp[0][0] * scale)
        cline_atlas[:, 0] = np.clip(cline_atlas[:, 0], a_min=0, a_max=flow_x.shape[0]-1)
        cline_atlas[:, 1] = np.clip(cline_atlas[:, 1], a_min=0, a_max=flow_x.shape[1] - 1)
        #print(cline_atlas[:10, 0])
        dcline_x = flow_x[tuple(cline_atlas[:, :2].T)]
        dcline_y = flow_y[tuple(cline_atlas[:, :2].T)]
        dcline = np.array([dcline_x, dcline_y]).T

        dcline[:, 0] += dp[0][1] * scale
        dcline[:, 1] += dp[0][0] * scale
        return dcline

    # #first round , shift the image based on average cline_atlas
    # dcline = deepflow_align(image, last_image, cline_atlas, scale=4)
    # dcline_mean = np.mean(dcline, axis=0)
    # print(dcline_mean)
    #
    # # shift the image
    # dcline_mean = np.rint(dcline_mean).astype(np.int16)
    # last_image_new = np.roll(last_image, shift=(dcline_mean[0], dcline_mean[1]), axis=[0, 1])
    #
    # cline_atlas_tl = np.copy(cline_atlas)
    # cline_atlas_tl[:, :2] = cline_atlas[:, :2] + dcline_mean[np.newaxis, :]
    # dcline_fine = deepflow_align(image, last_image_new, cline_atlas_tl, scale=4)
    #
    # cline_atlas_new = np.copy(cline_atlas_tl)
    # cline_atlas_new = cline_atlas_new.astype(np.float32)
    # cline_atlas_new[:, 0] += dcline_fine[:, 0]
    # cline_atlas_new[:, 1] += dcline_fine[:, 1]

    dcline_test = deepflow_align(image, last_image, cline_atlas, scale=4)
    cline_atlas_test = np.copy(cline_init)
    cline_atlas_test = cline_atlas_test.astype(np.float32)
    cline_atlas_test[:, 0] += dcline_test[:, 0]
    cline_atlas_test[:, 1] += dcline_test[:, 1]
    cline_atlas_test[:, 0] = np.clip(cline_atlas_test[:, 0], a_min=0, a_max=image.shape[0] - 1)
    cline_atlas_test[:, 1] = np.clip(cline_atlas_test[:, 1], a_min=0, a_max=image.shape[1] - 1)
    cline_atlas_new = cline_atlas_test

    if show:
        image_show = np.array([last_image, image])
        pt_show_0 = np.hstack((np.rint(cline_atlas[:, :2]), np.zeros((cline_atlas.shape[0], 1))))
        #pt_show_1 = np.hstack((np.rint(cline_atlas_tl[:, :2]), np.ones((cline_atlas_tl.shape[0], 1))))
        #pt_show_2 = np.hstack((np.rint(cline_atlas_new[:, :2]), 2 * np.ones((cline_atlas_new.shape[0], 1))))
        pt_show_3 = np.hstack((np.rint(cline_atlas_test[:, :2]), np.ones((cline_atlas_test.shape[0], 1))))
        pt_show = np.vstack((pt_show_0, pt_show_3))
        pt_show = pt_show.astype(np.uint16)
        Multi_Slice_Viewer(image_show, pt_show)
    return cline_atlas_new




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='/projects/LEIFER/PanNeuronal/Xinwei_test/free_AML32/BrainScanner20170424_105620',
                        type=str)
    parser.add_argument("--method", default='0', type=str)
    parser.add_argument("--get_cline", default=1, type=int)
    parser.add_argument("--start_idx", default=2, type=int)
    parser.add_argument("--end_idx", default=100, type=int)
    parser.add_argument("--date_mode", default='auto', type=str)
    parser.add_argument("--add_temp", action='store_true', help='to add template')
    parser.add_argument("--anno_direction", action='store_true', help='to annotate direction')
    parser.add_argument("--temp_align", default='none', type=str)
    parser.add_argument("--flip_mode", default='auto', type=str)
    args = parser.parse_args()

    neu_cline = Neuron_cline(args.folder, date_mode=args.date_mode)
    if args.add_temp:
        neu_cline.add_template()

    if args.anno_direction:
        neu_cline.human_annotate_direction()

    if args.method == '1' or args.method == 'all':
        neu_cline.find_cline(start_idx=args.start_idx, end_idx=args.end_idx)

    if args.method == '2' or args.method == 'all':
        anno_volume = glob.glob1(neu_cline.cline_tp_folder, 'anno_tp*')
        anno_volume = sorted([int(s.split('.')[0].split('_')[-1]) for s in anno_volume])
        tp_all = anno_volume[0]

        neu_cline.fine_tune_multiple(start_idx=args.start_idx, end_idx=args.end_idx, tp_all=tp_all, flip_mode=args.flip_mode)