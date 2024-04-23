# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:36:15 2023
This code is modified from yxw's original code submit_pipline.py for the new low mag designe with only the IR camera

The only change here is writing the slurm job runStraightStartNew.sh instead of runStraightStart.sh in the old version

@author: JL
"""
import os
import subprocess
import numpy as np
import argparse
import scipy.io as sio
import pickle
import getpass



parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/tigress/LEIFER/PanNeuronal/2018/20180410/BrainScanner20180410_144953',
                    type=str)
parser.add_argument("--num_align", default='100', type=int)
parser.add_argument("--num_seg", default='50', type=int)
parser.add_argument("--num_lim", default=1000, type=int)
args = parser.parse_args()
username = getpass.getuser()
# make a output folder to save the submit files and output files
out_folder = os.path.join(args.folder, 'output_segment')
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


hiResData = sio.loadmat(os.path.join(args.folder, 'hiResData.mat'))
names = hiResData['dataAll'].dtype.names
dict_name = dict()
for i, name in enumerate(names): dict_name[name] = i

stackIdx = hiResData['dataAll'][0][0][dict_name['stackIdx']]
min_stack, max_stack = np.min(stackIdx), np.max(stackIdx)
min_stack = max(1, min_stack)

randint = np.random.randint(1000)

## compile segmentation
out_file_name = os.path.join(out_folder, "strait_start.out")
err_file_name = os.path.join(out_folder, "strait_start.err")
job_file = os.path.join(out_folder, 'strait_start.job')
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -N 1 # node count\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --ntasks-per-socket=1\n")
    fh.writelines("#SBATCH --cpus-per-task=1\n")
    fh.writelines("#SBATCH --partition=physics\n")
    # fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH -t 0:59:59\n")
    fh.writelines("#SBATCH -J strait{}\n".format(args.folder))
    #fh.writelines("#SBATCH -d singleton\n")
    fh.writelines("#SBATCH --output {}\n".format(out_file_name))
    fh.writelines("#SBATCH --error {}\n".format(err_file_name))
    fh.writelines("# sends mail when process begins, and\n")
    # fh.writelines("#SBATCH --mail-type=end\n")
    # fh.writelines("#SBATCH --mail-user=xinweiy@princeton.edu\n")
    fh.writelines("./runStraightStartNew.sh {} \n".format(args.folder))
    fh.close()
command_str = "sbatch {}".format(job_file)
# print(command_str)
status, jobnum_start = subprocess.getstatusoutput(command_str)
jobnum_start = jobnum_start.split(' ')[-1]
if status == 0:
    print("Job_straight start is {}".format(jobnum_start))
else:
    print("Error submitting Job straight start")


## submit alignment jobs.

for i in range(min_stack, max_stack+1, args.num_align):
    v_idx = i
    v_len = min(v_idx + args.num_align, max_stack+1) - v_idx

    out_file_name = os.path.join(out_folder, "align_{}_{}.out".format(i, v_len))
    err_file_name = os.path.join(out_folder, "align_{}_{}.err".format(i, v_len))
    job_file = os.path.join(out_folder, 'align_{}_{}.job'.format(i, v_len))
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -N 1 # node count\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --ntasks-per-socket=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        #fh.writelines("#SBATCH --partition=physics\n")
        #fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH -d afterok:{} \n".format(jobnum_start))
        fh.writelines("#SBATCH -t 3:59:59\n")
        fh.writelines("#SBATCH -J align{}\n".format(args.folder))
        fh.writelines("#SBATCH --output {}\n".format(out_file_name))
        fh.writelines("#SBATCH --error {}\n".format(err_file_name))
        fh.writelines("# sends mail when process begins, and\n")
        fh.writelines("#when it ends. Make sure you define your email\n")
        fh.writelines("# Load anaconda3 environment\n")
        fh.writelines("source /projects/LEIFER/communalCode/HighMagNeuron/venv/bin/activate\n")
        #fh.writelines("python ./N2S_Multiscale.py\n")
        fh.writelines("python ./HighResFlow.py --folder {} --v_idx {} --len {} \n".format(args.folder, v_idx, v_len))
        #fh.writelines("python ./Noisy_Prob.py\n")
        fh.writelines("deactivate\n")
        fh.close()
    command_str = "sbatch {}".format(job_file)
    #print(command_str)
    status, jobnum = subprocess.getstatusoutput(command_str)
    jobnum = jobnum.split(' ')[-1]
    if status == 0:
        print("Job1 is {}".format(jobnum))
    else:
        print("Error submitting Job alignment")

# compile the names of all aligned volume into a file

out_file_name = os.path.join(out_folder, "align_compile.out")
err_file_name = os.path.join(out_folder, "align_compile.err")
job_file = os.path.join(out_folder, 'align_compile.job')
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -N 1 # node count\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --ntasks-per-socket=1\n")
    fh.writelines("#SBATCH --cpus-per-task=1\n")
    #fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH -t 0:59:59\n")
    fh.writelines("#SBATCH -J align{}\n".format(args.folder))
    #fh.writelines("#SBATCH --partition=physics\n")
    fh.writelines("#SBATCH -d singleton\n")
    fh.writelines("#SBATCH --output {}\n".format(out_file_name))
    fh.writelines("#SBATCH --error {}\n".format(err_file_name))
    fh.writelines("# sends mail when process begins, and\n")
    fh.writelines("# Load anaconda3 environment\n")
    fh.writelines("source /projects/LEIFER/communalCode/HighMagNeuron/venv/bin/activate\n")
    #fh.writelines("python ./N2S_Multiscale.py\n")
    fh.writelines("python ./compile_align_images.py --folder {} \n".format(args.folder))
    #fh.writelines("python ./Noisy_Prob.py\n")
    fh.writelines("deactivate\n")
    fh.close()
command_str = "sbatch {}".format(job_file)
#print(command_str)
status, jobnum_align_compile = subprocess.getstatusoutput(command_str)
jobnum_align_compile = jobnum_align_compile.split(' ')[-1]
if status == 0:
    print("Job_align_compile is {}".format(jobnum_align_compile))
else:
    print("Error submitting Job alignment compile")

# segement the neurons

for i in range(min_stack, max_stack+1, args.num_seg):
    v_idx = i
    v_len = min(v_idx + args.num_seg, max_stack+1) - v_idx

    out_file_name = os.path.join(out_folder, "seg_{}_{}.out".format(i, v_len))
    err_file_name = os.path.join(out_folder, "seg_{}_{}.err".format(i, v_len))
    job_file = os.path.join(out_folder, 'seg_{}_{}.job'.format(i, v_len))
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -N 1 # node count\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --ntasks-per-socket=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        #fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH -t 3:59:59\n")
        #fh.writelines("#SBATCH --partition=physics\n")
        fh.writelines("#SBATCH -J seg{}\n".format(args.folder))
        fh.writelines("#SBATCH -d afterok:{} \n".format(jobnum_align_compile))
        fh.writelines("#SBATCH --output {}\n".format(out_file_name))
        fh.writelines("#SBATCH --error {}\n".format(err_file_name))
        fh.writelines("# Load anaconda3 environment\n")
        fh.writelines("source /projects/LEIFER/communalCode/HighMagNeuron/venv/bin/activate\n")
        #fh.writelines("python ./N2S_Multiscale.py\n")
        fh.writelines("python ./neuron_detect_hess.py --datafolder {} --v_idx {} --len {} --num_lim {}\n".format(args.folder, v_idx, v_len, args.num_lim))
        #fh.writelines("python ./Noisy_Prob.py\n")
        fh.writelines("deactivate\n")
        fh.close()
    command_str = "sbatch {}".format(job_file)
    #print(command_str)
    status, jobnum = subprocess.getstatusoutput(command_str)
    jobnum = jobnum.split(' ')[-1]
    if status == 0:
        print("Job_seg is {}".format(jobnum))
    else:
        print("Error submitting Job segmentation")

## compile segmentation
out_file_name = os.path.join(out_folder, "seg_compile.out")
err_file_name = os.path.join(out_folder, "seg_compile.err")
job_file = os.path.join(out_folder, 'seg_compile.job')
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -N 1 # node count\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --ntasks-per-socket=1\n")
    fh.writelines("#SBATCH --cpus-per-task=1\n")
    #fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH -t 0:59:59\n")
    fh.writelines("#SBATCH -J seg{}\n".format(args.folder))
    fh.writelines("#SBATCH -d singleton\n")
    fh.writelines("#SBATCH --output {}\n".format(out_file_name))
    fh.writelines("#SBATCH --error {}\n".format(err_file_name))
    # fh.writelines("# sends mail when process begins, and\n")
    fh.writelines("#SBATCH --mail-type=end\n")
    fh.writelines("#SBATCH --mail-user={}@princeton.edu\n".format(username))
    #fh.writelines("./runWormCompilePointStats.sh {} \n".format(args.folder))
    fh.close()
command_str = "sbatch {}".format(job_file)
#print(command_str)
status, jobnum = subprocess.getstatusoutput(command_str)
jobnum = jobnum.split(' ')[-1]
if status == 0:
    print("Job_seg_compile is {}".format(jobnum))
else:
    print("Error submitting Job seg compile")
