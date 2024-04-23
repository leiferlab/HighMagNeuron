"""
Created on Weds Nov 20 11:38:41 2019
This is for looking at High resolution neural recording image.
@author: xinweiy
"""
import scipy.io as sio
import cv2
import numpy as np
import pickle
import struct
import os
import time
import argparse


def segment_cv(Araw, C, resize=True, buf=None, blur=0.65, thresh=0.003,
               sz_xy=3, sz_z=2):
    '''
    Parameters
    ----------
    Araw: numpy array
        3D-image with axes [x,y,z]
    C: numpy-array
        Filter used to calculate the pure second derivative (e.g., d^2/dx^2)
    resize: Boolean
        Whether to resize the input image to half the xy dimension. Z is not
        affected.
    buf: numpy array
        Buffer to write out some intermediate result, if needed. Shape must be
        the same as Araw if resize is False, otherwise x and y have to be
        halfed.
    blur: float
        Sigma of the Gaussian filter to perform the first blurring.
    sz_xy: int
        Size of the dilation kernel used in xy.
    sz_z: int
        Size of the brute-force local minimum finder used in z.

    Returns
    -------
    Neuron
        A numpy array of shape (3, N) with the xyz coordinates of the N neurons
        found.

    Finds the positions of the neurons in the volume Araw, calculating the 2D
    Hessian in the planes using the filter C after blurring them and then
    looking for local minima via dilation in xy and brute force check in z.
    '''

    # Resize
    Araw = Araw.astype(np.float64)
    planes = Araw.shape[2]
    dsizex = Araw.shape[0] // 2
    dsizey = Araw.shape[1] // 2
    dsize = dsizex
    if resize:
        A = cv2.resize(src=Araw, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                       interpolation=cv2.INTER_AREA)
    else:
        A = Araw

    bound_mask = (A > 300).astype(np.uint8)
    K_b = np.ones((10, 10), np.uint8)
    bound_mask = cv2.erode(bound_mask, K_b, iterations=1)

    # Apply Gaussian blur
    # blur = 0.65 usual
    # blur = 0.4 #L4
    gausize = 3
    if not resize: blur *= 2; gausize = 7
    A = cv2.GaussianBlur(A, (gausize, gausize), blur, blur)

    # Calculate BX = d^2/dx^2, BY = d^2/dy^2, and the trace of the Hessian in
    # the plane (B).
    BX = cv2.sepFilter2D(src=A, ddepth=-1, kernelX=C, kernelY=np.ones(1)) * bound_mask
    BY = cv2.sepFilter2D(src=A, ddepth=-1, kernelX=np.ones(1), kernelY=C) * bound_mask
    B = (BX + BY)

    # Threshold out some noise and mask the trace of the Hessian accordingly
    threshX = thresh * np.min(BX)  # 0.003
    threshY = thresh * np.min(BY)
    Bth = (BX < threshX) * (BY < threshY) * B

    # Find local maxima within the planes: Dilate and look for the pixels that
    # did not change their value. Since a minimum in 3D is also a minimum in 2D,
    # these are candidate neurons.
    # sz_xy = 3
    if not resize: sz_xy = sz_xy * 2 + 1
    
    K = np.ones((sz_xy, sz_xy), np.float64)
    Bdil = -cv2.dilate(-Bth, K)
    Multi_Slice_Viewer(np.moveaxis(np.concatenate((Bth, Bdil), axis=1), [0, 1, 2], [1, 2, 0]))

    # buf[:,:,:] = (B-Bdil+1.0)*(((B-Bdil)==0.0) * B!=0.0)[:,:,:]

    neuron_cand = (((B - Bdil) == 0.0) * B != 0.0).astype(np.uint16)
    Multi_Slice_Viewer(np.moveaxis(neuron_cand, [0, 1, 2], [1, 2, 0]))
    Neuron = np.array(np.where(neuron_cand))
    #Neuron = np.array(np.where((B - Bdil) == 0.0))
    # Instead of differentiating and dilating in z, brute force check if the
    # candidate neurons are points of minima of the 2D Hessian. This corresponds
    # to checking in which plane the xy curvature is maximum and not to looking
    # for the actual minimum in z.
    Neuron2 = []
    # if planes<20:
    #    pz = 1
    # else:
    #    pz = 2
    pz = sz_z
    pxy = sz_xy
    block_z = 1
    block_xy = 1
    for ne in Neuron.T:
        if all(ne != 0) and all(ne != dsize) and ne[2] != planes - 1:
            if neuron_cand[ne[0], ne[1], ne[2]] > 1:
                continue
            el = B[ne[0], ne[1], ne[2]]
            cube = B[ne[0] - pxy:ne[0] + pxy + 1, ne[1] - pxy:ne[1] + pxy + 1, \
                   ne[2] - pz:ne[2] + pz + 1]
            try:
                if el == np.min(cube):
                    # all surrounding neighbor becomes 2(save time)
                    neuron_cand[ne[0] - pxy:ne[0] + pxy + 1, ne[1] - pxy:ne[1] + pxy + 1, \
                    ne[2] - pz:ne[2] + pz + 1] *= 2
                    # closest neighbor needed to be blocked is >2 now
                    neuron_cand[ne[0] - block_xy:ne[0] + block_xy + 1, ne[1] - block_xy:ne[1] + block_xy + 1, \
                    ne[2] - block_z:ne[2] + block_z + 1] *= 2

                    Neuron2.append(ne)
            except:
                pass  # print("A cube failed")
    # Multi_Slice_Viewer(np.moveaxis(neuron_cand, [0, 1, 2], [1, 2, 0]))
    # # second round search, remove bright pts first
    # for ne in Neuron2:
    #     B[ne[0] - pxy:ne[0] + pxy + 1, ne[1] - pxy:ne[1] + pxy + 1, \
    #            ne[2] - pz:ne[2] + pz + 1] *= 0.8
    #
    #     xy_n1, z_n1 = 2, 1 # neighbor pixels * 0.8*0.7
    #     B[ne[0] - xy_n1:ne[0] + xy_n1 + 1, ne[1] - xy_n1:ne[1] + xy_n1 + 1, \
    #            ne[2] - z_n1:ne[2] + z_n1 + 1] *= 0.3
    #     # previous bright point 0.8*0.7*0.3
    #     B[ne[0], ne[1], ne[2]] *= 0.1
    #
    # Neuron = np.array(np.where((neuron_cand <= 2) * (neuron_cand >= 1)))
    # for ne in Neuron.T:
    #     if all(ne != 0) and all(ne != dsize) and ne[2] != planes - 1:
    #
    #         el = B[ne[0], ne[1], ne[2]]
    #         cube = B[ne[0] - pxy:ne[0] + pxy + 1, ne[1] - pxy:ne[1] + pxy + 1, \
    #                ne[2] - pz:ne[2] + pz + 1]
    #         try:
    #             if el == np.min(cube):
    #                 # all surrounding neighbor becomes 2(save time)
    #                 neuron_cand[ne[0] - pxy:ne[0] + pxy + 1, ne[1] - pxy:ne[1] + pxy + 1, \
    #                 ne[2] - pz:ne[2] + pz + 1] += 1
    #                 # closest neighbor needed to be blocked is 3 now
    #                 neuron_cand[ne[0] - block_xy:ne[0] + block_xy + 1, ne[1] - block_xy:ne[1] + block_xy + 1, \
    #                 ne[2] - block_z:ne[2] + block_z + 1] += 1
    #
    #                 Neuron2.append(ne)
    #         except:
    #             pass  # print("A cube failed")



    Neuron2 = np.array(Neuron2).T

    Multi_Slice_Viewer(np.moveaxis(np.concatenate((A, neuron_cand*200, B+400), axis=1), [0,1,2], [1,2,0]), Neuron2.T)
    if resize: Neuron = (Neuron2.T * np.array([2, 2, 1])).T
    if not resize: Neuron = Neuron2

    return Neuron


def warp_flow(img, flow):
    '''
        Applies to img the transformation described by flow.
    '''
    assert len(flow.shape) == 3 and flow.shape[-1] == 2
    flow_coord = np.copy(flow)
    hf, wf = flow.shape[:2]
    # flow         = -flow
    flow_coord[:, :, 0] += np.arange(wf)
    flow_coord[:, :, 1] += np.arange(hf)[:, np.newaxis]
    res = cv2.remap(img, flow_coord, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # flow_coord = flow_coord * 10
    # test = np.array([img, res, flow_coord[:, :, 0], flow_coord[:, :, 1]])
    # Multi_Slice_Viewer(test)
    return res

def stabilize_zstack(volume, im_std, scale=1):
    of_estim = cv2.optflow.createOptFlow_DeepFlow()
    im_std = im_std[:, 0]
    c_idx = np.sum(im_std * np.arange(len(im_std))) / np.sum(im_std)
    c_idx = int(np.rint(c_idx))
    frames = list()
    #frames_low = list()
    flow_prev = None

    # resize the image to reduce resolution
    volume = np.moveaxis(volume, [0, 1, 2], [2, 0, 1])
    planes = volume.shape[2]
    dsizex = volume.shape[0]
    dsizey = volume.shape[1]

    worm_img = cv2.resize(src=volume, dsize=(dsizey // scale, dsizex // scale), fx=0.0, fy=0.0, \
                          interpolation=cv2.INTER_AREA)
    worm_img = np.moveaxis(worm_img, [0, 1, 2], [1, 2, 0])
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    flow_dict = dict()
    for i in range(1, c_idx + 1):
        s_idx = c_idx - i
        t_idx = s_idx + 1

        src_ori = volume[s_idx, :, :]
        src = worm_img[s_idx, :, :]

        tgt = worm_img[t_idx, :, :]
        flow = of_estim.calc(tgt, src, None)

        if flow_prev is not None:
            new_flow_x = warp_flow(flow[:, :, 0], flow_prev)

            new_flow_y = warp_flow(flow[:, :, 1], flow_prev)
            flow_prev[:, :, 0] += new_flow_x
            flow_prev[:, :, 1] += new_flow_y
        else:
            flow_prev = flow

        # check the
        # frame_t = warp_flow(src_ori, cv2.resize(src=flow_prev * scale, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
        #                   interpolation=cv2.INTER_AREA))
        flow_full = cv2.resize(src=flow_prev * scale, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                           interpolation=cv2.INTER_AREA)
        frame_t = warp_flow(src_ori, flow_full)
        #frame_t_low = warp_flow(src, flow_prev)
        flow_dict[s_idx] = np.copy(flow_prev)
        frames.append(frame_t)
        #frames_low.append(frame_t_low)
    frames = frames[::-1]
    #frames_low = frames_low[::-1]

    flow_prev = None
    frames.append(volume[c_idx, :, :])
    #frames_low.append(worm_img[c_idx, :, :])
    for i in range(c_idx, volume.shape[0] - 1):
        s_idx = i + 1
        t_idx = i

        src_ori = volume[s_idx, :, :]
        src = worm_img[s_idx, :, :]
        tgt = worm_img[t_idx, :, :]
        flow = of_estim.calc(tgt, src, None)
        if flow_prev is not None:
            new_flow_x = warp_flow(flow[:, :, 0], flow_prev)
            new_flow_y = warp_flow(flow[:, :, 1], flow_prev)
            flow_prev[:, :, 0] += new_flow_x
            flow_prev[:, :, 1] += new_flow_y
        else:
            flow_prev = flow
        flow_full = cv2.resize(src=flow_prev * scale, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                           interpolation=cv2.INTER_AREA)
        frame_t = warp_flow(src_ori, flow_full)

        #frame_t_low = warp_flow(src, flow_prev)
        flow_dict[s_idx] = np.copy(flow_prev)
        frames.append(frame_t)
        #frames_low.append(frame_t_low)

    # frames_low = np.moveaxis(np.array(frames_low), [0, 1, 2], [2, 0, 1])
    # frames_low = cv2.resize(src=np.array(frames_low), dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
    #                       interpolation=cv2.INTER_AREA)
    # frames_low = np.moveaxis(frames_low, [0, 1, 2], [1, 2, 0])
    #frames_low = np.array(frames_low)
    frames = np.array(frames)
    #Multi_Slice_Viewer(np.concatenate((worm_img, frames_low), axis=2))
    #Multi_Slice_Viewer(worm_img)
    #Multi_Slice_Viewer(np.concatenate((volume, frames), axis=2))

    return frames, flow_dict

class worm_data_loader(object):
    def __init__(self, folder):
        # load the files that are used to load volumes
        hiResData = sio.loadmat(os.path.join(folder, 'hiResData.mat'))
        sts = sio.loadmat(os.path.join(folder, 'startWorkspace.mat'))

        self.flow_folder = os.path.join(folder, 'flow_folder')
        names = hiResData['dataAll'].dtype.names
        dict_name = dict()
        for i, name in enumerate(names): dict_name[name] = i
        self.M_inv = sts['M_matrix_inv'].T
        self.zOffset = sts['zOffset'][0][0]
        self.stackIdx = hiResData['dataAll'][0][0][dict_name['stackIdx']]
        self.minIdx, self.maxIdx = np.min(self.stackIdx), np.max(self.stackIdx)
        self.Z = hiResData['dataAll'][0][0][dict_name['Z']]
        self.imSTD = hiResData['dataAll'][0][0][dict_name['imSTD']]
        try:
            self.back_v = np.mean(hiResData['dataAll'][0][0][dict_name['imAvg']])
        except:
            self.back_v = None
        self.stackIdx = self.stackIdx.reshape(self.stackIdx.shape[0])
        self.folder = folder

        # get the stackIdx that has flash:
        flash_Loc = hiResData['dataAll'][0][0][dict_name['flashLoc']][0]
        self.flash_stack = list()
        range_sz = 1
        for i in range(-range_sz, range_sz+1):
            self.flash_stack += [self.stackIdx[loc - 1] + i for loc in flash_Loc]



    def load_frame(self, volume_idx):
        t = volume_idx
        rec = dict()
        FramesIdx, = np.where(self.stackIdx == t)
        FramesIdx = FramesIdx[FramesIdx >= 0]
        rec['z'] = self.Z[FramesIdx]
        rec['imSTD'] = self.imSTD[FramesIdx]
        FramesIdx += self.zOffset
        mask = FramesIdx >= 0
        rec['z'] = rec['z'][mask]
        rec['imSTD'] = rec['imSTD'][mask]
        FramesIdx = FramesIdx[mask]

        I = len(FramesIdx)
        if os.path.exists(os.path.join(self.folder, "sCMOS_Frames_U16_1024x512.dat")):
            record_f = os.path.join(self.folder, "sCMOS_Frames_U16_1024x512.dat")
            b = 512
            c = 512
        else:
            record_f = os.path.join(self.folder, "sCMOS_Frames_U16_1024x1024.dat")
            b = 600
            c = 600

        #b = 512

        A = np.zeros((I, b, c * 2), dtype=np.uint16)
        framesize = b * (c * 2) * 2

        with open(record_f, 'br') as f:
            for i in np.arange(I):
                frameidx = FramesIdx[i] + 1
                f.seek(framesize * frameidx)  # go to frame
                im = np.array(struct.unpack(str(b * (2 * c)) + 'H', f.read(b * (2 * c) * 2)))
                A[i, :, :] = im.reshape((b, 2 * c))
            # deal with flash
            if t in self.flash_stack:
                mean_array = np.mean(A, axis=(1, 2))
                if self.back_v is None:
                    back_v = np.median(mean_array)
                else:
                    back_v = self.back_v
                flash_z = np.where(mean_array > (back_v + 600))[0]
                if len(flash_z) > 0:
                    min_flashz = max(0, np.min(flash_z) - 1)
                    max_flashz = min(A.shape[0], np.max(flash_z) + 2)
                    A[min_flashz:max_flashz, :, :] = back_v

            rec['volume'] = A
            f.close()

        rec['back_v'] = self.back_v
        rec['volume_idx'] = volume_idx
        return rec

    def get_frame_aligned(self, volume_idx, channel='red'):
        # use the flow file to align the volume
        rec = self.load_frame(volume_idx)
        v_idx = rec['volume_idx']
        # check the flow file.
        flow_file = os.path.join(self.flow_folder,  self.folder[-15:] + "_{}.pkl".format(v_idx))
        if os.path.exists(flow_file):
            with open(flow_file, "rb") as f:
                flow_dict = pickle.load(f)
                f.close()
        else:
            print('does not found {}'.format(flow_file))
            return None

        # align the image.
        num_v = rec['volume'].shape[0]
        size_2 = rec['volume'].shape[2] // 2


        if channel == 'red':
            # red channel
            img = rec['volume'][:, :, :size_2]
        else:
            # green channel
            img = rec['volume'][:, :, size_2:]
        dsizex = img.shape[1]
        dsizey = img.shape[2]

        # align the image frame by frame
        for i in range(num_v):
            if i in flow_dict:
                flow = flow_dict[i]
                scale = dsizex / flow.shape[0]
                hf, wf = flow.shape[:2]
                flow[:, :, 0] = flow[:, :, 0] - np.arange(wf)
                flow[:, :, 1] = flow[:, :, 1] - np.arange(hf)[:, np.newaxis]

                flow_full = cv2.resize(src=flow * scale, dsize=(dsizey, dsizex), fx=0.0, fy=0.0, \
                                       interpolation=cv2.INTER_AREA)
                hf, wf = flow_full.shape[:2]
                # flow         = -flow
                flow_full[:, :, 0] += np.arange(wf)
                flow_full[:, :, 1] += np.arange(hf)[:, np.newaxis]

                if channel != 'red':
                    flow_full = cv2.perspectiveTransform(flow_full, self.M_inv)

                img[i] = cv2.remap(img[i], flow_full, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                # only need to handle green channel.
                if channel != 'red':
                    flow_full = np.zeros((dsizex, dsizey, 2))
                    flow_full[:, :, 0] += np.arange(dsizey)
                    flow_full[:, :, 1] += np.arange(dsizex)[:, np.newaxis]
                    flow_full = cv2.perspectiveTransform(flow_full, self.M_inv)
                    flow_full = flow_full.astype(np.float32)
                    img[i] = cv2.remap(img[i], flow_full, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        volume_dict = dict()
        volume_dict['image'] = img
        volume_dict['ZZ'] = rec['z']
        volume_dict['imSTD'] = rec['imSTD']
        return volume_dict


def stable_volume(folder, v_idx, len_run, save_folder=None):
    worm_loader = worm_data_loader(folder)
    for i in range(v_idx, v_idx + len_run):

        rec = worm_loader.load_frame(volume_idx=i)

        volume = rec['volume']
        print('volume size:{}'.format(volume.shape))
        if len(volume) < 10:
            continue

        size_2 = volume.shape[2] // 2
        img_red = volume[:, :, :size_2]
        # Multi_Slice_Viewer(img_red)
        # exit()
        # This step should have been deleted, but kept for convenience.
        #img_red = np.moveaxis(img_red, [0, 1, 2], [0, 2, 1])
        img_red_aligned, flow_dict = stabilize_zstack(img_red, rec['imSTD'], scale=2)

        volume_dict = dict()
        volume_dict['image'] = img_red_aligned
        volume_dict['ZZ'] = rec['z']
        volume_dict['imSTD'] = rec['imSTD']

        if save_folder is None:
            save_folder = folder
        align_folder = os.path.join(save_folder, 'aligned_volume')
        flow_folder = os.path.join(save_folder, 'flow_folder')
        if not os.path.exists(align_folder):
            os.mkdir(align_folder)
        if not os.path.exists(flow_folder):
            os.mkdir(flow_folder)

        with open(os.path.join(align_folder, folder[-15:] + "_{}.pkl".format(i)), "wb") as f:
            pickle.dump(volume_dict, f)
            f.close()

        for key in flow_dict.keys():
            flow_coord = np.copy(flow_dict[key])
            hf, wf = flow_coord.shape[:2]
            flow_coord[:, :, 0] = flow_coord[:, :, 0] + np.arange(wf)
            flow_coord[:, :, 1] = flow_coord[:, :, 1] + np.arange(hf)[:, np.newaxis]
            flow_dict[key] = flow_coord

        with open(os.path.join(flow_folder,  folder[-15:] + "_{}.pkl".format(i)), "wb") as f:
            pickle.dump(flow_dict, f)
            f.close()

    return img_red_aligned
    


if __name__ == "__main__":
    #folder = "/tigress/LEIFER/PanNeuronal/20191106/BrainScanner20191106_143222"
    #folder = '/tigress/LEIFER/PanNeuronal/20191216/BrainScanner20191216_150705'
    #folder = '/tigress/LEIFER/PanNeuronal/2018/20181210/BrainScanner20181210_172207'
    #folder = '/tigress/LEIFER/PanNeuronal/20191007/BrainScanner20191007_151627'
    #folder = '/tigress/LEIFER/PanNeuronal/20200106/BrainScanner20200106_151101'
    #/ tigress / LEIFER / PanNeuronal / 20191216 / BrainScanner20191216_150705
    #'/tigress/LEIFER/PanNeuronal/2018/20180410/BrainScanner20180410_144953'
    parser = argparse.ArgumentParser()
    #parser.add_argument("--folder", default='/projects/LEIFER/PanNeuronal/Xinwei_test/free_AML32/BrainScanner20170424_105620', type=str)
    parser.add_argument("--folder", default='/projects/LEIFER/PanNeuronal/20221017_msc/BrainScanner20221017_210822',
                        type=str)
    parser.add_argument("--v_idx", default='100', type=int)
    parser.add_argument("--len", default='1', type=int)
    args = parser.parse_args()
    
    tic = time.time()
    img_red_aligned = stable_volume(args.folder, v_idx=args.v_idx, len_run=args.len)
    print('Run time:{}'.format(time.time()-tic))
    print(img_red_aligned.shape)
