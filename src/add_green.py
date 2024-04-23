import argparse
import os
import glob
import pickle
import numpy as np
from HighResFlow import worm_data_loader
from segmentNet import kernel_radius


def crop_image(image, c_pos, crop_size=[5, 9, 9]):
    im_sz = image.shape
    c_pos = np.rint(c_pos).astype(np.uint16)
    back_v = np.mean(image)
    crop_img = np.ones(crop_size) * back_v

    dim0_step = (crop_size[0] - 1) // 2
    dim1_step = (crop_size[1] - 1) // 2
    dim2_step = (crop_size[2] - 1) // 2

    b0_img = np.clip(np.array([c_pos[0] - dim0_step, c_pos[0] + dim0_step + 1]), a_min=0, a_max=im_sz[0])
    b1_img = np.clip(np.array([c_pos[1] - dim1_step, c_pos[1] + dim1_step + 1]), a_min=0, a_max=im_sz[1])
    b2_img = np.clip(np.array([c_pos[2] - dim2_step, c_pos[2] + dim2_step + 1]), a_min=0, a_max=im_sz[2])

    b0_crop = b0_img - c_pos[0] + dim0_step
    b1_crop = b1_img - c_pos[1] + dim1_step
    b2_crop = b2_img - c_pos[2] + dim2_step
    crop_img[b0_crop[0]:b0_crop[1], b1_crop[0]:b1_crop[1], b2_crop[0]:b2_crop[1]] = image[b0_img[0]:b0_img[1], b1_img[0]:b1_img[1], b2_img[0]:b2_img[1]]

    return crop_img

def outlier_mask(crop, values):
    percent_75 = np.percentile(values, 75)
    percent_25 = np.percentile(values, 25)
    int_percent = percent_75 - percent_25
    #print(percent_25, percent_75)
    outlier_scale = 2
    crop_mask = (crop < percent_75 + outlier_scale * int_percent) * (crop > percent_25 - outlier_scale * int_percent)
    return crop_mask

def local_max_search(volume, pt, search_area=[1, 5, 5]):
    # perform a local search of local maxima in 2d
    crop = crop_image(volume, pt, crop_size=search_area)
    max_pos = np.unravel_index(np.argmax(crop, axis=None), search_area)
    origin_pt = (np.array(search_area) - 1) // 2
    pt_new = pt + np.array(max_pos) - origin_pt
    return pt_new


def extract_signal(neurons, worm_loader, v_idx):
    crop_size = [5, 7, 7]
    kernel_mask = kernel_radius(kernel_size=crop_size, r=3.0, dim1_scale=1.4) > 0
    # get the green signal also
    # print('time {}'.format(time.time() - tic))
    volume_dict_g = worm_loader.get_frame_aligned(v_idx, channel='green')

    aligned_folder = os.path.join(worm_loader.folder, 'aligned_volume')
    file_r = worm_loader.folder[-15:] + '_{}.pkl'.format(v_idx)
    aligned_file = os.path.join(aligned_folder, file_r)
    if os.path.exists(aligned_file):
        with open(aligned_file, "rb") as f:
            volume_r = pickle.load(f)
            f.close()
    else:
        volume_r = worm_loader.get_frame_aligned(v_idx, channel='red')


    volume_g = volume_dict_g['image']
    volume_r = volume_r['image']

    num_neurons = len(neurons['pts'])

    neurons['max_inten_green'] = list()
    neurons['mean_inten_green'] = list()

    neurons['max_inten'] = list()
    neurons['mean_inten'] = list()

    neurons['green_img'] = list()
    neurons['red_img'] = list()

    for neu_idx in range(num_neurons):
        crop_r = crop_image(volume_r, neurons['pts_max'][neu_idx], crop_size=crop_size)
        values = crop_r[kernel_mask]
        outlier_mask_r = outlier_mask(crop_r, values)


        pt_max_g = local_max_search(volume_g,  neurons['pts_max'][neu_idx])
        crop_g = crop_image(volume_g, pt_max_g, crop_size=crop_size)
        values = crop_g[kernel_mask]
        outlier_mask_g = outlier_mask(crop_g, values)

        mask_r = kernel_mask * outlier_mask_r
        mask_g = kernel_mask * outlier_mask_g
        values = crop_g[mask_g]
        neurons['max_inten_green'].append(np.max(values))
        neurons['mean_inten_green'].append(np.mean(values))
        # neurons['green_img'].append(crop_image(volume_g, neurons['pts'][neu_idx]))

        values = crop_r[mask_r]
        neurons['max_inten'].append(np.max(values))
        neurons['mean_inten'].append(np.mean(values))
        # neurons['red_img'].append(crop_image(volume_r, neurons['pts'][neu_idx]))
        # Multi_Slice_Viewer(np.concatenate((crop_r, crop_r * mask_r), axis=2))
        # Multi_Slice_Viewer(np.concatenate((crop_g, crop_g * mask_g), axis=2))

    return neurons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='/projects/LEIFER/PanNeuronal/Xinwei_test/free_AML32/BrainScanner20170613_134800',
                        type=str)
    args = parser.parse_args()

    aligned_folder = os.path.join(args.folder, 'aligned_volume')
    neuron_folder = os.path.join(args.folder, 'neuron_pt')
    files = glob.glob1(neuron_folder, '*.mat')

    worm_loader = worm_data_loader(args.folder)
    for file in files:
        #tic = time.time()
        v_idx = int(file.split('.mat')[0].split('_')[-1])

        file_r = args.folder[-15:] + '_{}.pkl'.format(v_idx)
        aligned_file = os.path.join(aligned_folder, file_r)
        with open(aligned_file, "rb") as f:
            volume_r = pickle.load(f)
            f.close()

        with open(os.path.join(neuron_folder, file), 'rb') as f:
            neurons = pickle.load(f)
            f.close()
        neurons = extract_signal(neurons, worm_loader, v_idx)

        with open(os.path.join(neuron_folder, file), "wb") as f:
            pickle.dump(neurons, f)
            f.close()
        #print('finish {}'.format(time.time()-tic))

