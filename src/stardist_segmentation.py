import pickle
import os
import numpy as np
import argparse
from HighResFlow import worm_data_loader
from add_green import extract_signal
from csbdeep.utils import Path, normalize
from stardist.models import StarDist3D
from skimage.measure import regionprops


def stardist_segmentation(raw_image):
    model_path = '/projects/LEIFER/communalCode/HighMagNeuron/stardist_model'
    # model_path = '/home/matt/Documents/python/stardist_segmentation/models'
    axis_norm = (0, 1, 2)
    model = StarDist3D(None, name='stardist', basedir=model_path)
    image_normalized = normalize(raw_image, 1, 99.8, axis=axis_norm)
    labels, details = model.predict_instances(image_normalized)

    return labels, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", default='/projects/LEIFER/PanNeuronal/20221017_msc/BrainScanner20221017_210822', type=str)
    #parser.add_argument("--filename", default='20180410_144953_518.pkl', type=str)
    parser.add_argument("--v_idx", default='2868', type=int)
    parser.add_argument("--len", default='10', type=int)
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

        # with open(aligned_file, "rb") as f:
        #     volume_dict = pickle.load(f)
        #     f.close()
        volume_r = worm_loader.get_frame_aligned(i, channel='red')
        volume_g = worm_loader.get_frame_aligned(i, channel='green')

        ZZ = volume_r['ZZ']
        volume_r = volume_r['image']
        volume_g = volume_g['image']
        z_diff = np.mean(np.abs(np.diff(ZZ, axis=0)))

        # segment the neurons with stardist
        labels, details = stardist_segmentation(volume_r)
        # gets information about the image from regions defined in labels
        props_r = regionprops(labels, intensity_image=volume_r)
        props_g = regionprops(labels, intensity_image=volume_g)
        neurons = {}
        neurons['pts'] = [details['points'][i, :] for i in range(details['points'].shape[0])]
        neurons['pts_max'] = [details['points'][i, :] for i in range(details['points'].shape[0])]
        neurons['num_neuron'] = details['points'].shape[0]

        neurons['mask'] = []
        neurons['red_img'] = []
        neurons['green_img'] = []
        neurons['max_inten'] = []
        neurons['mean_inten'] = []
        neurons['label_o'] = []
        neurons['area'] = []
        neurons['bbox'] = []
        neurons['max_inten_green'] = []
        neurons['mean_inten_green'] = []
        for pi in range(len(props_r)):
            neurons['mask'].append(props_r[pi].image)
            neurons['max_inten'].append(props_r[pi].max_intensity)
            neurons['mean_inten'].append(props_r[pi].mean_intensity)
            neurons['label_o'].append(props_r[pi].label)
            neurons['area'].append(props_r[pi].area)
            neurons['bbox'].append(props_r[pi].bbox)

            neurons['max_inten_green'].append(props_g[pi].max_intensity)
            neurons['mean_inten_green'].append(props_g[pi].mean_intensity)

        if neurons is None:
            print('check stack {}, too many neurons'.format(i))
            continue

        with open(aligned_file, "rb") as f:
            volume_dict = pickle.load(f)
            f.close()

        neurons['Z'] = ZZ

        # comment out this when used.
        neuron_save_folder = os.path.join(data_folder, 'neuron_pt')
        if not os.path.exists(neuron_save_folder):
            os.mkdir(neuron_save_folder)

        filename = filename.split('.')[0] + '.mat'

        neuron_file = os.path.join(neuron_save_folder, filename)

        with open(neuron_file, "wb") as f:
            pickle.dump(neurons, f)
            f.close()
        print('save neuron of {}'.format(neuron_file))





