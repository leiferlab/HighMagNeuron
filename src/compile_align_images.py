# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:52:07 2019
@author: yxw
"""
import os
import numpy as np
import argparse
import scipy.io as sio
import glob
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/tigress/LEIFER/PanNeuronal/2018/20180410/BrainScanner20180410_144953_xinwei/',
                    type=str)
args = parser.parse_args()

# check number of file
align_folder = os.path.join(args.folder, 'aligned_volume')
filename_list = glob.glob(os.path.join(align_folder, '*.pkl'))
list_file_name = os.path.join(align_folder, 'filenames.pkl')
with open(list_file_name, 'wb') as f:
    pickle.dump(filename_list, f)
    f.close()