from segmentNet import Detect_From_Deconv
import pickle
import os
import numpy as np
import argparse
from HighResFlow import worm_data_loader
from add_green import extract_signal



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

def get_neuron_inten(neurons, volume_r, volume_g):
    num_neurons = len(neurons['pts'])
    neurons['max_inten_green'] = list()
    neurons['mean_inten_green'] = list()
    # rewrite red signal here.
    neurons['max_inten'] = list()
    neurons['mean_inten'] = list()

    for neu_idx in range(num_neurons):
        # get the neuron one by one.
        bbox = neurons['bbox'][neu_idx]
        mask = neurons['mask'][neu_idx]
        subimg_g = volume_g[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        subimg_r = volume_r[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        values = subimg_g[mask]
        neurons['max_inten_green'].append(np.max(values))
        neurons['mean_inten_green'].append(np.mean(values))

        values = subimg_r[mask]
        neurons['max_inten'].append(np.max(values))
        neurons['mean_inten'].append(np.mean(values))
    return neurons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", default='/projects/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_160928', type=str)
    #parser.add_argument("--filename", default='20180410_144953_518.pkl', type=str)
    parser.add_argument("--v_idx", default='2868', type=int)
    parser.add_argument("--len", default='1', type=int)
    parser.add_argument("--num_lim", default=1000, type=int)
    args = parser.parse_args()

    data_folder = args.datafolder
    v_idx = args.v_idx
    len_run = args.len

    worm_loader = worm_data_loader(data_folder)

    aligned_folder = os.path.join(data_folder, 'aligned_volume')
    flow_folder = os.path.join(data_folder, 'flow_folder')

    for i in range(v_idx, v_idx+len_run):
        #filename = filename_list[i].split('/')[-1]
        filename = data_folder[-15:] + '_{}.pkl'.format(i)
        if filename == 'filenames.pkl':
            continue


        aligned_file = os.path.join(aligned_folder, filename)
        flow_file = os.path.join(flow_folder, filename)
        if not os.path.exists(flow_file):
            continue

        with open(aligned_file, "rb") as f:
            volume_dict = pickle.load(f)
            f.close()

        volume = np.copy(volume_dict['image'])
        ZZ = volume_dict['ZZ']
        z_diff = np.mean(np.abs(np.diff(ZZ, axis=0)))
        volume = volume - np.mean(volume)
        volume[volume < 0] = 0

        use_gpu = False
        #device = torch.device("cuda:0" if use_gpu else "cpu")
        detect_deconv = Detect_From_Deconv(use_gpu)
        neurons = detect_deconv.detect_neuron_hessian(volume, z_diff, show=0, worm_green=None)


        if neurons is None:
            print('check stack {}, too many neurons'.format(i))
            continue

        neurons['Z'] = ZZ

        # get the green/red signal also
        neurons = extract_signal(neurons, worm_loader, i)

        # comment out this when used.
        neuron_save_folder = os.path.join(data_folder, 'neuron_pt')
        if not os.path.exists(neuron_save_folder):
            os.mkdir(neuron_save_folder)

        filename = filename.split('.')[0] + '.mat'

        neuron_file = os.path.join(neuron_save_folder, filename)

        if len(neurons['pts']) > args.num_lim:
            inten_copy = neurons['max_inten'].copy()
            inten_copy.sort()
            inten_thd = inten_copy[-(args.num_lim-1)]
            mask = np.array(neurons['max_inten']) >= inten_thd
            neurons['pts'] = [neurons['pts'][i] for i in range(len(mask)) if mask[i]]
            neurons['pts_max'] = [neurons['pts_max'][i] for i in range(len(mask)) if mask[i]]
            neurons['area'] = [neurons['area'][i] for i in range(len(mask)) if mask[i]]
            neurons['max_inten'] = [neurons['max_inten'][i] for i in range(len(mask)) if mask[i]]
            neurons['mean_inten'] = [neurons['mean_inten'][i] for i in range(len(mask)) if mask[i]]
            neurons['label_o'] = [neurons['label_o'][i] for i in range(len(mask)) if mask[i]]
            neurons['bbox'] = [neurons['bbox'][i] for i in range(len(mask)) if mask[i]]
            neurons['mask'] = [neurons['mask'][i] for i in range(len(mask)) if mask[i]]
            neurons['max_inten_green'] = [neurons['max_inten_green'][i] for i in range(len(mask)) if mask[i]]
            neurons['mean_inten_green'] = [neurons['mean_inten_green'][i] for i in range(len(mask)) if mask[i]]


        with open(neuron_file, "wb") as f:
            pickle.dump(neurons, f)
            f.close()
        print('save neuron of {}'.format(neuron_file))





