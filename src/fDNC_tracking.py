"""
Tracking the whole-brain imaging recording with fDNC.
"""

import os
import scipy.io as sio
from HighResFlow import worm_data_loader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from model_utils import NIT_Registration
import torch


def predict_label(model, temp_pos, temp_label, test_pos, temp_color=None, test_color=None,
                  cuda=True, topn=5):

    # put template worm data and test worm data into a batch
    pt_batch = list()
    color_batch = list()

    pt_batch.append(temp_pos[:, :3])
    pt_batch.append(test_pos[:, :3])# here we can add more test worm if provided as a list.
    if temp_color is not None and test_color is not None:
        color_batch.append(temp_color)
        color_batch.append(test_color)
    else:
        color_batch = None
    data_batch = dict()
    data_batch['pt_batch'] = pt_batch
    data_batch['color'] = color_batch
    data_batch['match_dict'] = None
    data_batch['ref_i'] = 0

    model.eval()
    pt_batch = data_batch['pt_batch']
    with torch.no_grad():
        _, output_pairs = model(pt_batch, match_dict=None, ref_idx=data_batch['ref_i'], mode='eval')
    # p_m is the match of worms to the worm0
    i = 1
    p_m = output_pairs['p_m'][i].detach().cpu().numpy()
    num_neui = len(pt_batch[i])
    p_m = p_m[:num_neui, :]

    #p_m = p_m[:, :-1] + color_m * 1
    num_tmp = p_m.shape[1] - 1
    p_m = np.hstack((p_m[:, :-1], np.repeat(p_m[:, -1:], 20, axis=1)))
    row, col = linear_sum_assignment(-p_m)

    prob_m = softmax(p_m, axis=1)

    # most probable label
    test_label = [(-2, 0)] * num_neui
    for row_i in range(len(row)):
        if prob_m[row[row_i], col[row_i]] > 0.5 and col[row_i] < num_tmp:
            test_label[row[row_i]] = (temp_label[col[row_i]], prob_m[row[row_i], col[row_i]])
    #candidates list
    #p_m_sortidx = np.argsort(-p_m, axis=1)
    candidate_list = []
    #     for row_i, rank_idx in enumerate(p_m_sortidx[:, :topn]):
    #         cur_list = [(temp_label[idx], prob_m[row_i, idx])  for idx in rank_idx]
    #         candidate_list.append(cur_list)
    return test_label, candidate_list


def load_pt(pS_pts, cur_z, side, scale=200):
    z_idx = pS_pts[:, 2] - 1
    z_idx = z_idx.astype(np.int)
    z = cur_z[z_idx]
    pS_pts[:, 2] = z * 47.619
    if side == 0:
        pS_pts[:, [0, 2]] *= -1

    pS_pts -= np.median(pS_pts, axis=0)
    pS_pts /= scale
    return pS_pts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='/projects/LEIFER/PanNeuronal/behavior_annotation/AML310/BrainScanner20200130_105254_XP', type=str)
    parser.add_argument("--ref_idx", default=100, type=int)

    args = parser.parse_args()

    cuda = True
    model = NIT_Registration(input_dim=3, n_hidden=128, n_layer=6, p_rotate=0, feat_trans=0, cuda=True)
    device = torch.device("cuda:0" if cuda else "cpu")
    # load trained model
    # fDNC model path
    model_path = "/projects/LEIFER/Xinwei/github/fDLC_Neuron_ID/model/model.bin"
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)


    # load the pointStats file.
    ps_p = os.path.join(args.folder, 'PointsStats.mat')
    ps = sio.loadmat(ps_p)['pointStats']
    names = ps.dtype.names
    ps_dictName = dict()
    for i, name in enumerate(names): ps_dictName[name] = i

    # load the template pointStatsNew file.
    psNew_p = './pointStatsNew_temp.mat'
    psNew = sio.loadmat(psNew_p)['pointStatsNew']
    psNew = np.repeat(psNew, ps.shape[1], axis=1)
    names = psNew.dtype.names
    ps_New_dictName = dict()
    for i, name in enumerate(names): ps_New_dictName[name] = i

    # # psNew_p = os.path.join(folder, 'pointStatsNew.mat')
    # # pSNew = sio.loadmat(psNew_p)['pointStatsNew']
    # psNew_p = os.path.join(args.folder, 'PointsStats2_old.mat')
    # pSNew = sio.loadmat(psNew_p)['pointStats2']
    # pSNew_ori = np.copy(pSNew)
    #
    # names = pSNew.dtype.names
    # pS_dictName_New = dict()
    # for i, name in enumerate(names): pS_dictName_New[name] = i


    worm_loader = worm_data_loader(args.folder)
    # side = 1 if on left side
    side = 1
    pts = np.copy(ps[0, args.ref_idx][ps_dictName['rawPoints']])
    stackIdx = ps[0, args.ref_idx][ps_dictName['stackIdx']][0, 0]
    z_list = worm_loader.load_z(stackIdx)
    temp_pts = load_pt(pts, z_list[:, 0], side)
    temp_label2 = np.arange(len(temp_pts)) + 1.

    for i in range(ps.shape[1]):
        pts = np.copy(ps[0, i][ps_dictName['rawPoints']])

        # copy the information
        psNew[0, i][ps_New_dictName['stackIdx']] = ps[0, i][ps_dictName['stackIdx']]
        psNew[0, i][ps_New_dictName['straightPoints']] = ps[0, i][ps_dictName['straightPoints']]
        psNew[0, i][ps_New_dictName['rawPoints']] = ps[0, i][ps_dictName['rawPoints']]
        psNew[0, i][ps_New_dictName['pointIdx']] = ps[0, i][ps_dictName['pointIdx']]
        psNew[0, i][ps_New_dictName['Rintensities']] = ps[0, i][ps_dictName['Rintensities']]
        psNew[0, i][ps_New_dictName['Volume']] = ps[0, i][ps_dictName['Volume']]


        if len(pts) < 1 or pts.shape[1] < 1:
            trackIdx = np.array([1.0])
            psNew[0, i][ps_New_dictName['trackIdx']] = trackIdx.astype(np.float)
            continue
        #print(i)
        stackIdx = ps[0, i][ps_dictName['stackIdx']][0, 0]
        z_list = worm_loader.load_z(stackIdx)
        pts = load_pt(pts, z_list[:, 0], side)

        test_label_pred, candidate_list = predict_label(model, temp_pts, temp_label2, pts,
                                                        temp_color=None, test_color=None, cuda=True, topn=5)

        trackIdx = [np.nan] * len(pts)
        for neu_i, item in enumerate(test_label_pred):
            if item[0] >= 0:
                trackIdx[neu_i] = item[0]
        trackIdx = np.array(trackIdx)[:, np.newaxis]
        #trackIdx = trackIdx.astype(np.float)

        psNew[0, i][ps_New_dictName['trackIdx']] = trackIdx

    sio.savemat(os.path.join(args.folder, 'pointStatsNew.mat'), {'pointStatsNew':psNew})