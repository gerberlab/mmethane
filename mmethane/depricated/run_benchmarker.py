import sys
import os
sys.path.append(os.path.abspath(".."))
import subprocess
import itertools
import numpy as np
import time
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
parser.add_argument('--max_jobs', type=int, default=5)
parser.add_argument('--type', type=str, default = ['both','taxa','metabs'], nargs='+')
parser.add_argument('--scorer', type=str, default=['f1'],nargs='+')
parser.add_argument('--model', type=str, default=['LR','RF','AdaBoost'],nargs='+')
parser.add_argument('--seeds', type=int, default=list(range(10)),nargs='+')
parser.add_argument('--cv_type', type=str, default='kfold')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--taxa_tr', type=str, choices=['standardize','clr','none','sqrt'], default='clr')
args = parser.parse_args()
# max_jobs = 5
# both=True

dataset_ls = [
    # ('','../datasets/SEMISYN/processed/otus_128_3/seqs.pkl')
    # ('../datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl',
    #  '../datasets/FRANZOSA/processed/franzosa_cts/seqs.pkl'),
    ('../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl',
     '../datasets/ERAWIJANTARI/processed/erawijantari_cts/seqs.pkl'),
    # ('../datasets/WANG/processed/wang_pubchem/mets.pkl',
    #  '../datasets/WANG/processed/wang_cts/seqs.pkl'),
    # ('../datasets/HE/processed/he_pubchem/2_mets.pkl',
    #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # ('../datasets/IBMDB/processed/ibmdb_pubchem/mets.pkl',
    #  '../datasets/IBMDB/processed/ibmdb_cts/seqs.pkl'),
    ('../datasets/CDI/processed/cdi_pubchem/mets.pkl',
     '../datasets/CDI/processed/cdi_cts/seqs.pkl'),

]

seeds = ' '.join([str(s) for s in args.seeds])
for type in args.type:
    pid_list = []
    for model in args.model:
        for met_data, taxa_data in dataset_ls:
            for scorer in args.scorer:

                if type == 'metabs':
                    if 'CDI' in met_data:
                        args.cv_type='loo'
                    else:
                        args.cv_type='kfold'
                    cmd = f"python3 ./benchmarker.py --model {model} --met_data {met_data} --seed {seeds} --scorer {scorer} --dtype metabs --cv_type {args.cv_type} --taxa_tr {args.taxa_tr} --run_name {args.run_name}"

                    # cmd_out = cmd.replace(' ','').split('.py')[-1].replace('/','_').replace('--','__').replace('-','').replace('..','_').replace('.','_')
                    # cmd += f' > {cmd_out}.log 2>&1'
                    print(f'Command {cmd} sent to benchmarker.py')
                    pid = subprocess.Popen(cmd.split(' '), stdout=sys.stdout, stderr=sys.stderr)
                    pid_list.append(pid)
                    time.sleep(1)
                    while sum([x.poll() is None for x in pid_list]) >= args.max_jobs:
                        time.sleep(1)
                else:
                    for log in args.log_transform_otus:
                        if type=='both':
                            if 'CDI' in met_data:
                                args.cv_type = 'loo'
                            else:
                                args.cv_type='kfold'
                            cmd = f"python3 ./benchmarker.py --model {model} --met_data {met_data} --otu_data {taxa_data} --seed {seeds} --scorer {scorer} --dtype metabs otus --cv_type {args.cv_type} --taxa_tr {args.taxa_tr} --run_name {args.run_name}"

                        elif type=='taxa':
                            if 'CDI' in taxa_data:
                                args.cv_type = 'loo'
                            else:
                                args.cv_type='kfold'
                            cmd = f"python3 ./benchmarker.py --model {model} --taxa_data {taxa_data} --seed {seeds} --scorer {scorer} --dtype taxa --cv_type {args.cv_type} --taxa_tr {args.taxa_tr} --run_name {args.run_name}"

                        else:
                            raise ValueError('Specify allowed type of data: metabs, otus, or both')

                        if log==1:
                            cmd += " -log"
                        print(f'Command {cmd} sent to benchmarker.py')
                        pid = subprocess.Popen(cmd.split(' '))
                        pid_list.append(pid)
                        while sum([x.poll() is None for x in pid_list]) >= args.max_jobs:
                            time.sleep(1)


