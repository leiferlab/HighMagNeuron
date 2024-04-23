"""
This file submit cline to della
"""

import os
import subprocess
import numpy as np
import argparse
import scipy.io as sio
import pickle
import time
import glob
import getpass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='/tigress/LEIFER/PanNeuronal/2018/20180410/BrainScanner20180410_144953',
                        type=str)
    parser.add_argument("--method", default='all', type=str)
    parser.add_argument("--num_fine", default=100, type=int)
    parser.add_argument("--flip_mode", default='auto', type=str)
    args = parser.parse_args()
    username = getpass.getuser()
    # find the initial file otherwise generate it on tigressdata
    init_file = os.path.join(args.folder, 'neuron_cline_init.pt')

    if os.path.exists(init_file):
        with open(init_file, 'rb') as f_init:
            init_v = pickle.load(f_init)
    else:
        print('go to tigressdata and run python Himag_cline --folder {}'.format(args.folder))
    #self.output_folder = os.path.join(worm_folder, 'neuron_cline_result')
    cline_tp_folder = os.path.join(args.folder, os.path.join('neuron_cline_result', 'cline_template'))
    anno_volume = glob.glob1(cline_tp_folder, 'anno_tp*')
    anno_volume = sorted([int(s.split('.')[0].split('_')[-1]) for s in anno_volume])

    out_folder = os.path.join(args.folder, 'output_cline')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # the first round of tracking cline across time.
    if args.method == '1' or args.method == 'all':
        for i in range(len(anno_volume)):
            start_idx = anno_volume[i]
            if i < len(anno_volume) - 1:
                end_idx = anno_volume[i + 1]
            else:
                end_idx = init_v['end_idx'] + 1

            out_file_name = os.path.join(out_folder, "cline_{}_{}.out".format(start_idx, end_idx))
            err_file_name = os.path.join(out_folder, "cline_{}_{}.err".format(start_idx, end_idx))
            job_file = os.path.join(out_folder, 'cline_{}_{}.job'.format(start_idx, end_idx))
            with open(job_file, "w") as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH -N 1 # node count\n")
                fh.writelines("#SBATCH --ntasks-per-node=1\n")
                fh.writelines("#SBATCH --ntasks-per-socket=1\n")
                fh.writelines("#SBATCH --cpus-per-task=1\n")
                #fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH -t 23:59:59\n")
                #fh.writelines("#SBATCH --partition=physics\n")
                fh.writelines("#SBATCH -J cline{}\n".format(args.folder))
                fh.writelines("#SBATCH --output {}\n".format(out_file_name))
                fh.writelines("#SBATCH --error {}\n".format(err_file_name))
                fh.writelines("# sends mail when process begins, and\n")
                fh.writelines("#when it ends. Make sure you define your email\n")
                # fh.writelines("#SBATCH --mail-type=end\n")
                # fh.writelines("#SBATCH --mail-user=xinweiy@princeton.edu\n")
                fh.writelines("# Load anaconda3 environment\n")
                fh.writelines("source /projects/LEIFER/communalCode/HighMagNeuron/venv/bin/activate\n")
                #fh.writelines("python ./N2S_Multiscale.py\n")
                fh.writelines("python ./Himag_cline.py --folder {} --method 1 --date_mode last --start_idx {} --end_idx {}\n".format(args.folder, start_idx, end_idx))
                #fh.writelines("python ./Noisy_Prob.py\n")
                fh.writelines("deactivate\n")
                fh.close()
            command_str = "sbatch {}".format(job_file)
            #print(command_str)
            status, jobnum = subprocess.getstatusoutput(command_str)
            jobnum_first = jobnum.split(' ')[-1]
            if status == 0:
                print("Job1 is {}".format(jobnum_first))
            else:
                print("Error submitting Job Cline")

    if args.method == '1' or args.method == 'all':
        ## compile segmentation
        out_file_name = os.path.join(out_folder, "cline_init_compile.out")
        err_file_name = os.path.join(out_folder, "cline_init_compile.err")
        job_file = os.path.join(out_folder, 'cline_init_compile.job')
        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH -N 1 # node count\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --ntasks-per-socket=1\n")
            fh.writelines("#SBATCH --cpus-per-task=1\n")
            #fh.writelines("#SBATCH --gres=gpu:1\n")
            fh.writelines("#SBATCH -t 0:59:59\n")
            #fh.writelines("#SBATCH --partition=physics\n")
            fh.writelines("#SBATCH -J cline{}\n".format(args.folder))
            fh.writelines("#SBATCH -d singleton\n")
            fh.writelines("#SBATCH --output {}\n".format(out_file_name))
            fh.writelines("#SBATCH --error {}\n".format(err_file_name))
            # fh.writelines("# sends mail when process begins, and\n")
            # fh.writelines("#SBATCH --mail-type=end\n")
            # fh.writelines("#SBATCH --mail-user=xinweiy@princeton.edu\n")
            #fh.writelines("./runWormCompilePointStats.sh {} \n".format(args.folder))
            fh.close()
        command_str = "sbatch {}".format(job_file)
        #print(command_str)
        status, jobnum_first = subprocess.getstatusoutput(command_str)
        jobnum_first = jobnum_first.split(' ')[-1]
        if status == 0:
            print("Job_cline_init_compile is {}".format(jobnum_first))
        else:
            print("Error submitting Job cline init compile")


    if args.method == '2' or args.method == 'all':
        # fine tune the centerline.
        min_stack = init_v['start_idx']
        max_stack = init_v['end_idx']
        for i in range(min_stack, max_stack + 1, args.num_fine):
            v_idx = i
            v_last = min(v_idx + args.num_fine, max_stack + 1)

            out_file_name = os.path.join(out_folder, "fine_{}_{}.out".format(i, v_last))
            err_file_name = os.path.join(out_folder, "fine_{}_{}.err".format(i, v_last))
            job_file = os.path.join(out_folder, 'fine_{}_{}.job'.format(i, v_last))
            with open(job_file, "w") as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH -N 1 # node count\n")
                fh.writelines("#SBATCH --ntasks-per-node=1\n")
                fh.writelines("#SBATCH --ntasks-per-socket=1\n")
                fh.writelines("#SBATCH --cpus-per-task=1\n")
                #fh.writelines("#SBATCH --partition=physics\n")
                # fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH -t 3:59:59\n")
                fh.writelines("#SBATCH -J fine{}\n".format(args.folder))
                if args.method == 'all':
                    fh.writelines("#SBATCH -d afterok:{} \n".format(jobnum_first))
                fh.writelines("#SBATCH --output {}\n".format(out_file_name))
                fh.writelines("#SBATCH --error {}\n".format(err_file_name))
                fh.writelines("# sends mail when process begins, and\n")
                fh.writelines("#when it ends. Make sure you define your email\n")
                fh.writelines("# Load anaconda3 environment\n")
                fh.writelines("source /projects/LEIFER/communalCode/HighMagNeuron/venv/bin/activate\n")
                # fh.writelines("python ./N2S_Multiscale.py\n")
                fh.writelines(
                    "python ./Himag_cline.py --folder {} --method 2 --date_mode last --start_idx {} --end_idx {} --flip_mode {}\n".format(args.folder, v_idx, v_last, args.flip_mode))
                # fh.writelines("python ./Noisy_Prob.py\n")
                fh.writelines("deactivate\n")
                fh.close()
            command_str = "sbatch {}".format(job_file)
            # print(command_str)
            status, jobnum = subprocess.getstatusoutput(command_str)
            jobnum_fine = jobnum.split(' ')[-1]
            if status == 0:
                print("Job1 is {}".format(jobnum_fine))
            else:
                print("Error submitting Cline fine tune")


    if args.method == '2' or args.method == 'all' or args.method == '3':
        ## compile segmentation
        out_file_name = os.path.join(out_folder, "cline_compile.out")
        err_file_name = os.path.join(out_folder, "cline_compile.err")
        job_file = os.path.join(out_folder, 'cline_compile.job')
        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH -N 1 # node count\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --ntasks-per-socket=1\n")
            fh.writelines("#SBATCH --cpus-per-task=1\n")
            #fh.writelines("#SBATCH --partition=physics\n")
            #fh.writelines("#SBATCH --gres=gpu:1\n")
            fh.writelines("#SBATCH -t 0:59:59\n")
            fh.writelines("#SBATCH -J fine{}\n".format(args.folder))
            fh.writelines("#SBATCH -d singleton\n")
            fh.writelines("#SBATCH --output {}\n".format(out_file_name))
            fh.writelines("#SBATCH --error {}\n".format(err_file_name))
            fh.writelines("# sends mail when process begins, and\n")
            fh.writelines("#SBATCH --mail-type=end\n")
            fh.writelines("#SBATCH --mail-user={}@princeton.edu\n".format(username))
            fh.writelines("./runWormCompilePointStats.sh {} \n".format(args.folder))
            fh.close()
        command_str = "sbatch {}".format(job_file)
        #print(command_str)
        status, jobnum = subprocess.getstatusoutput(command_str)
        jobnum = jobnum.split(' ')[-1]
        if status == 0:
            print("Job_cline_compile is {}".format(jobnum))
        else:
            print("Error submitting Job cline compile")