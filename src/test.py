from HighResTest import Multi_Slice_Viewer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os
from skimage import io
import scipy.io as sio
from HighResFlow import worm_data_loader
from segmentNet import Detect_From_Deconv


def find_neighbor(image, pt, r=[2, 2]):
    # image should be a 2d image.
    pt_m = np.rint(np.copy(pt)).astype(np.int64)
    min_x = max(0, pt_m[0] - r[0])
    min_y = max(0, pt_m[1] - r[1])
    max_x = min(image.shape[0] - 1, r[0] + pt_m[0])
    max_y = min(image.shape[1] - 1, r[1] + pt_m[1])

    max_v = 0

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if image[x, y] > max_v:
                pt_m[0], pt_m[1] = x, y
                max_v = image[x, y]

    return pt_m, max_v




f_idx = 100
folder = '/projects/LEIFER/PanNeuronal/20210702/BrainScanner20210702_122235'
# sub_name = folder.split('/')[-1][-15:]
# pt_name = os.path.join(folder, 'neuron_pt/{}_{}.mat'.format(sub_name, f_idx))
# with open(pt_name, 'rb') as f:
#     pt_seg = pickle.load(f)
#     f.close()

#pts_m = np.rint(pt_seg['pts_max']).astype(np.int16)
worm_loader = worm_data_loader(folder)
rec = worm_loader.get_frame_aligned(f_idx, channel='red')

volume = rec['image']
ZZ = rec['ZZ']
z_diff = np.mean(np.abs(np.diff(ZZ, axis=0)))
volume = volume - np.mean(volume)
volume[volume < 0] = 0

detect_deconv = Detect_From_Deconv(use_gpu=False)
neurons = detect_deconv.detect_neuron_hessian(volume, z_diff, show=0)
pts_m = np.rint(neurons['pts_max']).astype(np.int16)

img_red = volume
# print('image shape:', img_red.shape)
# plt.imshow(np.max(img_red, axis=0))
# plt.show()

#Multi_Slice_Viewer(img_red)
Multi_Slice_Viewer(img_red, pts_m[:, [1, 2, 0]])