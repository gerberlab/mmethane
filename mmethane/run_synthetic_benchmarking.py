import sys
import os
sys.path.append(os.path.abspath(".."))
import subprocess
import itertools
import numpy as np
import time
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--max_jobs', type=int, default=1)
# default = ['both_and','and_metabs','metabs','otus']
parser.add_argument('--cases', type=str, default = ['both_and', 'otus', 'and_otus','metabs','and_metabs'], nargs='+')
parser.add_argument('--scorer', type=str, default=['f1'],nargs='+')
parser.add_argument('--model', type=str, default=['LR','RF','AdaBoost'],nargs='+')
parser.add_argument('--log_transform_otus', type=int, default=[0],nargs='+')
parser.add_argument('--full', type=int, default=[0],nargs='+')
parser.add_argument('--cv_type', type=str, default='kfold')
parser.add_argument('--data_name', type=str, default='')
parser.add_argument('--make_new_data', type=int, default=0)
parser.add_argument('--no_filter', action='store_true')
parser.add_argument('--otu_tr', type=str, choices=['standardize','clr','none','sqrt'],default='sqrt')
# parser.add_argument('--seeds', type=int, default=list(range(10,100)), nargs='+')
parser.add_argument('--seeds', type=int, default=[
    # 0,1,2,3,4,5,6,7,8,9
0,10,20,30,40,50,60,70,80,90,
    1,11,21,31,41,51,61,71,81,91,
                                                  2,12,22,32,42,52,62,72,82,92,
                                                  3,13,23,33,43,53,63,73,83,93,
                                                  4,14,24,34,44,54,64,74,84,94,
                                                  5,15,25,35,45,55,65,75,85,95,
                                                  6,16,26,36,46,56,66,76,86,96,
                                                  7,17,27,37,47,57,67,77,87,97,
                                                  8,18,28,38,48,58,68,78,88,98,

                                                  99,89,79,69,59,49,39,29,19,9
                                                  ], nargs='+')
parser.add_argument('-N','--N_subj', type=int, default=[36,48,64,128,300,1000], nargs='+')
args = parser.parse_args()

# cases = ['both','metabs','both_metabs','and_metabs','or_metabs']

N = args.N_subj
# N=[96]
# N = [128]
path = '../datasets/SEMISYN/processed/'
num_subjs = ' '.join([str(n) for n in N])
seeds = ' '.join([str(s) for s in args.seeds])
# make_data=f"python3 ./semi_synthetic_data.py -N {num_subjs} --seeds {seeds} --cases {' '.join(args.cases)} --path {path}"
# if args.make_new_data:
#     print(make_data)
#     pid_list=[]
#     pid = subprocess.Popen(make_data.split(' '), stdout=sys.stdout, stderr=sys.stderr)
#     pid_list.append(pid)
#     time.sleep(0.1)
#     while sum([x.poll() is None for x in pid_list]) >= 1:
#         time.sleep(1)


pid_list=[]
for n in N:
    for case in args.cases:
        case_path = f'{path}/{case}_{n}'
        for scorer in args.scorer:
            for model in args.model:
                # for s in args.seeds:
                    # if not os.path.isfile(case_path + f'_{s}/mets.pkl'):
                    #     make_data = f"python3 ./semi_synthetic_data.py -N {n} --seeds {s} --cases {case} --path {path}"
                    #     print(make_data)
                    #     syn_ls = []
                    #     pid = subprocess.Popen(make_data.split(' '), stdout=sys.stdout, stderr=sys.stderr)
                    #     syn_ls.append(pid)
                    #     time.sleep(0.1)
                    #     while sum([x.poll() is None for x in syn_ls]) >= 1:
                    #         time.sleep(1)
                if 'metabs' in case:
                    dtype='metabs'
                elif 'both' in case:
                    dtype='otus metabs'
                elif 'otus' in case:
                    dtype='otus'
                run_benchmarker=f"python3 ./benchmarker.py --model {model} --syn_data {case_path} --scorer {scorer} " \
                                f"--dtype {dtype} --cv_type {args.cv_type} --is_synthetic --seed {seeds} " \
                                f"--log_dir logs_semisyn/ --otu_tr {args.otu_tr} --data_name {args.data_name}"
                if args.no_filter:
                    run_benchmarker += " --no_filter"
                print(f'Command {run_benchmarker} sent to benchmarker.py')
                pid = subprocess.Popen(run_benchmarker.split(' '), stdout=sys.stdout, stderr=sys.stderr)
                pid_list.append(pid)
                time.sleep(1)
                while sum([x.poll() is None for x in pid_list]) >= args.max_jobs:
                    time.sleep(1)