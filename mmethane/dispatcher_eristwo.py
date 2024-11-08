import os
import sys
sys.path.append(os.path.abspath(".."))
import numpy as np
import subprocess
import argparse
import itertools
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import time
import pandas as pd
from get_results import get_results, eval_rules
# from slurmpy import Slurm

parser = argparse.ArgumentParser()
parser.add_argument('--max_load', default = 10, type=int)
parser.add_argument('--max_jobs', default = 10, type=int)
parser.add_argument('--case', default='', type=str)
parser.add_argument('--plot_all_seeds', default = 0, type=int)
parser.add_argument('--plot_metabolite_structures', default=0, type=int)
parser.add_argument('-r','--only_get_results', default=0, type=int)
parser.add_argument('--num_seeds', default=None, type=int)
parser.add_argument('--seeds', default=[
            1,11,21,31,41,51,61,71,81,91,
                                               0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                               2, 12, 22, 32, 42, 52, 62, 72, 82, 92,
                                               3, 13, 23, 33, 43, 53, 63, 73, 83, 93,
                                               4, 14, 24, 34, 44, 54, 64, 74, 84, 94,
                                               5, 15, 25, 35, 45, 55, 65, 75, 85, 95,
                                               6,16,26,36,46,56,66,76,86,96,
                                               7,17,27,37,47,57,67,77,87,97,
                                               8,18,28,38,48,58,68,78,88,98,
                                               9,19,29,39,49,59,69,79,89,99
                                               ], type=int)
parser.add_argument('--num_threads', default=6, type=int)
parser.add_argument('--change_params', default=[], type=str, nargs='+')
parser.add_argument('--out_path', default='/data/bwh-comppath-full/gerberlab/jen/logs-OCT17/', type=str)
parser.add_argument('--datasets', default=[

    '../datasets/SEMISYN/processed/otus_1000',
    '../datasets/SEMISYN/processed/otus_48',
    '../datasets/SEMISYN/processed/otus_64',
    '../datasets/SEMISYN/processed/otus_128',
    '../datasets/SEMISYN/processed/otus_300',
    '../datasets/SEMISYN/processed/otus_36',
    # #
    # '../datasets/SEMISYN/processed/both_and_1000',
    # '../datasets/SEMISYN/processed/both_and_48',
    # '../datasets/SEMISYN/processed/both_and_36',
    # '../datasets/SEMISYN/processed/both_and_64',
    # '../datasets/SEMISYN/processed/both_and_128',
    # '../datasets/SEMISYN/processed/both_and_300',

    # # #
    # '../datasets/SEMISYN/processed/metabs_1000',
    # '../datasets/SEMISYN/processed/metabs_36',
    # '../datasets/SEMISYN/processed/metabs_48',
    # '../datasets/SEMISYN/processed/metabs_64',
    # '../datasets/SEMISYN/processed/metabs_128',
    # '../datasets/SEMISYN/processed/metabs_300',
    #
    '../datasets/SEMISYN/processed/and_otus_1000',
    '../datasets/SEMISYN/processed/and_otus_48',
    '../datasets/SEMISYN/processed/and_otus_36',
    '../datasets/SEMISYN/processed/and_otus_64',
    '../datasets/SEMISYN/processed/and_otus_128',
    '../datasets/SEMISYN/processed/and_otus_300',

    # '../datasets/SEMISYN/processed/and_metabs_1000',
    # '../datasets/SEMISYN/processed/and_metabs_48',
    # '../datasets/SEMISYN/processed/and_metabs_36',
    # '../datasets/SEMISYN/processed/and_metabs_64',
    # '../datasets/SEMISYN/processed/and_metabs_128',
    # '../datasets/SEMISYN/processed/and_metabs_300',

    # ('../datasets/ERAWIJANTARI/processed/erawijantari_map4/mets.pkl',
    #  '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl'),
    # '../datasets/ERAWIJANTARI/processed/erawijantari_map4/mets.pkl',
    #
    # '../datasets/FRANZOSA/processed/franzosa_map4/mets.pkl',
    # # ('../datasets/FRANZOSA/processed/franzosa_map4/mets.pkl',
    # #  '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl'),
    # #
    # '../datasets/HE/processed/he_map4/2_mets.pkl',
    # # ('../datasets/HE/processed/he_map4/2_mets.pkl',
    # #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # # #
    # '../datasets/IBMDB/processed/ibmdb_map4/mets.pkl',
    # # ('../datasets/IBMDB/processed/ibmdb_map4/mets.pkl',
    # #  '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl'),
    # #
    # #
    # '../datasets/WANG/processed/wang_map4/mets.pkl',
    # # ('../datasets/WANG/processed/wang_map4/mets.pkl',
    # #  '../datasets/WANG/processed/wang_ra/seqs.pkl'),
    #
    # '../datasets/CDI/processed/cdi_map4/mets.pkl',
    # # ('../datasets/CDI/processed/cdi_map4/mets.pkl',
    # #  '../datasets/CDI/processed/cdi_cts/seqs.pkl'),
    # #
    # '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl',
    # ('../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl',
    #  '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl'),
    # '../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl',
    # #
    # '../datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl',
    # ('../datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl',
    #  '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl'),
    # '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl',
    # # #
    # '../datasets/HE/processed/he_pubchem/2_mets.pkl',
    # ('../datasets/HE/processed/he_pubchem/2_mets.pkl',
    #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # '../datasets/HE/processed/he_cts/2_seqs.pkl',
    # #
    # '../datasets/IBMDB/processed/ibmdb_pubchem/mets.pkl',
    # ('../datasets/IBMDB/processed/ibmdb_pubchem/mets.pkl',
    #  '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl'),
    # '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl',
    # #
    # '../datasets/WANG/processed/wang_pubchem/mets.pkl',
    # ('../datasets/WANG/processed/wang_pubchem/mets.pkl',
    #  '../datasets/WANG/processed/wang_ra/seqs.pkl'),
    # '../datasets/WANG/processed/wang_ra/seqs.pkl',
    # #
    # '../datasets/CDI/processed/cdi_pubchem/mets.pkl',
    # ('../datasets/CDI/processed/cdi_pubchem/mets.pkl',
    #  '../datasets/CDI/processed/cdi_cts/seqs.pkl'),
    # '../datasets/CDI/processed/cdi_cts/seqs.pkl',
    #
    # # ('../datasets/ERAWIJANTARI/processed/erawijantari_infomax/mets.pkl',
    # #  '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl'),
    # '../datasets/ERAWIJANTARI/processed/erawijantari_infomax/mets.pkl',
    #
    # '../datasets/FRANZOSA/processed/franzosa_infomax/mets.pkl',
    # # ('../datasets/FRANZOSA/processed/franzosa_infomax/mets.pkl',
    # #  '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl'),
    #
    # '../datasets/HE/processed/he_infomax/2_mets.pkl',
    # # ('../datasets/HE/processed/he_infomax/2_mets.pkl',
    # #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # #
    # '../datasets/IBMDB/processed/ibmdb_infomax/mets.pkl',
    # # ('../datasets/IBMDB/processed/ibmdb_infomax/mets.pkl',
    # #  '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl'),
    # #
    # #
    # '../datasets/WANG/processed/wang_infomax/mets.pkl',
    # # ('../datasets/WANG/processed/wang_infomax/mets.pkl',
    # #  '../datasets/WANG/processed/wang_ra/seqs.pkl'),
    #
    # '../datasets/CDI/processed/cdi_infomax/mets.pkl',
    # # ('../datasets/CDI/processed/cdi_infomax/mets.pkl',
    # # '../datasets/CDI/processed/cdi_cts/seqs.pkl'),
    #
    # # ('../datasets/ERAWIJANTARI/processed/erawijantari_morgan/mets.pkl',
    # #  '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl'),
    # '../datasets/ERAWIJANTARI/processed/erawijantari_morgan/mets.pkl',
    #
    # '../datasets/FRANZOSA/processed/franzosa_morgan/mets.pkl',
    # # ('../datasets/FRANZOSA/processed/franzosa_morgan/mets.pkl',
    # #  '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl'),
    # #
    # '../datasets/HE/processed/he_morgan/2_mets.pkl',
    # # ('../datasets/HE/processed/he_morgan/2_mets.pkl',
    # #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # # #
    # '../datasets/IBMDB/processed/ibmdb_morgan/mets.pkl',
    # # ('../datasets/IBMDB/processed/ibmdb_morgan/mets.pkl',
    # #  '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl'),
    # #
    # #
    # '../datasets/WANG/processed/wang_morgan/mets.pkl',
    # # ('../datasets/WANG/processed/wang_morgan/mets.pkl',
    # #  '../datasets/WANG/processed/wang_ra/seqs.pkl'),
    #
    # '../datasets/CDI/processed/cdi_morgan/mets.pkl',
    # # ('../datasets/CDI/processed/cdi_morgan/mets.pkl',
    # #  '../datasets/CDI/processed/cdi_cts/seqs.pkl'),
    #
    # # ('../datasets/ERAWIJANTARI/processed/erawijantari_mqn/mets.pkl',
    # #  '../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl'),
    # '../datasets/ERAWIJANTARI/processed/erawijantari_mqn/mets.pkl',
    #
    # '../datasets/FRANZOSA/processed/franzosa_mqn/mets.pkl',
    # ('../datasets/FRANZOSA/processed/franzosa_mqn/mets.pkl',
    #  '../datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl'),
    # #
    # '../datasets/HE/processed/he_mqn/2_mets.pkl',
    # # ('../datasets/HE/processed/he_mqn/2_mets.pkl',
    # #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    # # #
    # '../datasets/IBMDB/processed/ibmdb_mqn/mets.pkl',
    # ('../datasets/IBMDB/processed/ibmdb_mqn/mets.pkl',
    #  '../datasets/IBMDB/processed/ibmdb_ra/seqs.pkl'),
    # #
    # #
    # '../datasets/WANG/processed/wang_mqn/mets.pkl',
    # # ('../datasets/WANG/processed/wang_mqn/mets.pkl',
    # #  '../datasets/WANG/processed/wang_ra/seqs.pkl'),
    #
    # '../datasets/CDI/processed/cdi_mqn/mets.pkl',
    # ('../datasets/CDI/processed/cdi_mqn/mets.pkl',
    #  '../datasets/CDI/processed/cdi_cts/seqs.pkl'),

], type=str, nargs='+')
args = parser.parse_args()
job_script = '''#!/bin/bash
'''
slurm_single_script = '''#!/bin/bash
#SBATCH --job-name=mditre_metabolites   # Job name
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjdawkins@bwh.harvard.edu   # Where to send mail	
#SBATCH --mem=200G                   # Job Memory
#SBATCH --output={1}.log    # Standard output and error log
#SBATCH --partition=bwh_comppath,bwh_comppath_long
#SBATCH -t 0-24:00
#SBATCH --cpus-per-task {2}
#SBATCH --exclude=bwhc-dn032

pwd; hostname; date

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/PHShome/jjd65/.conda/envs/mditre/lib

export MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS={2}
export OPENBLAS_NUM_THREADS={2}
export MKL_NUM_THREADS={2}
export VECLIB_MAXIMUM_THREADS={2}
export NUMEXPR_NUM_THREADS={2}

{0}

echo {0}

date

'''

slurm_script = '''#!/bin/bash
#SBATCH --job-name=mditre_metabolites   # Job name
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjdawkins@bwh.harvard.edu   # Where to send mail	
#SBATCH --mem=200G                   # Job Memory
#SBATCH --output=array_%Aa.log    # Standard output and error log
#SBATCH --array=1-{1}%{2}                 # Array range
#SBATCH --partition=bwh_comppath,bwh_comppath_long
#SBATCH -t 0-24:00
#SBATCH --cpus-per-task {3}
#SBATCH --nodelist=bwhc-dn033,bwhc-dn034,bwhc-dn035,bwhc-dn038,bwhc-dn019,bwhc-dn020,bwhc-dn021,bwhc-dn015,bwhc-dn017,bwhc-dn013,bwhc-dn014,bwhc-dn011,bwhc-dn012,bwhc-dn004,bwhc-dn002,bwhc-dn003,bwhc-dn008,bwhc-dn005,bwhc-dn007,bwhc-dn023,bwhc-dn025,bwhc-dn024
# SBATCH --exclude=bwhc-dn032,bwhc-dn036,bwhc-dn004,bwhc-dn005,bwhc-dn006,bwhc-dn009,bwhc-dn010,bwhc-dn015,bwhc-dn018,bwhc-dn019,bwhc-dn022,bwhc-dn023,bwhc-dn026,bwhc-dn028,bwhc-dn030,bwhc-dn032,bwhc-dn033,bwhc-dn034,bwhc-dn035,bwhc-dn036,bwhc-dn037,bwhc-dn038,bwhc-dn039
#pwd; hostname; date

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/PHShome/jjd65/.conda/envs/mditre/lib

export MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS={3}
export OPENBLAS_NUM_THREADS={3}
export MKL_NUM_THREADS={3}
export VECLIB_MAXIMUM_THREADS={3}
export NUMEXPR_NUM_THREADS={3}

{0} --seed $SLURM_ARRAY_TASK_ID

echo $SLURM_ARRAY_TASK_ID

echo {0}

date

'''

slurm_results_script = '''#!/bin/bash
#SBATCH --job-name=mditre_metabolites_results   # Job name
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjdawkins@bwh.harvard.edu   # Where to send mail	
#SBATCH --mem=50G                   # Job Memory
#SBATCH --output=results_%j.log    # Standard output and error log
#SBATCH --partition bwh_comppath,bwh_comppath_long
#SBATCH --dependency=afterok:{1}
pwd; hostname; date

{0} &

echo {0} 

date
'''
pid_list = []



params_to_change = []
param_dict_ = {}
if args.num_seeds is not None:
    seeds = np.arange(args.num_seeds)
else:
    seeds = args.seeds
    args.num_seeds=len(seeds)
    param_dict = {
        "optimizer": ["NAdam"],
        "bc_loss_type": "sum",
        "filter_data": 1,
        'otus_n_d': [10],
        "use_ray": 1,
        "plot_all_seeds": 0,
        "epochs": [3000],
        "parallel": 6,
        # 'annealing_limit':[(0,0.75),(0, 0.7)],
        # ('metabs_lr_alpha','otus_lr_alpha','lr_beta'):[(0.001,0.001,0.001)],
        ('otus_init_clusters', 'metabs_init_clusters'): [
            ('kmeans', 'kmeans'),

        ],
        ('metabs_use_pca', 'otus_use_pca'): [(0, 0)],
        'metabs_lr_alpha': [0.001],
        'patience': [100],
        'metabs_lr_thresh': [0.0005],
        "metabs_min_k_bc": [0.5],
        "otus_lr_thresh": [0.0001],
        'schedule_lr': [0],
        # ('otus_adj_kappa_loss','metabs_adj_kappa_loss'):[(0,0),(1,1)],
        ('otus_adj_detector_loss', 'metabs_adj_detector_loss', 'otus_adj_kappa_loss', 'metabs_adj_kappa_loss'):
            [(0, 0, 0, 0)],
        'adj_rule_loss': [0],
        'kmeans_noise': [1],
        "early_stopping": [1],
        # 'otus_lr_thresh': [0.0001],
        ('otus_adj_n_d', 'metabs_adj_n_d'): [(0, 0)],
        'init_w_LR': [0], 'init_mult': [0.1],
        'eta_min': [0.00001],
        'alpha_init_zeros': [1]
        # 'otus_kappa_mult':[1e3,10,1]

    }
total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__') and not isinstance(v, str)])
total_cases = total_iters * args.num_seeds * len(args.datasets)
# timer=BackgroundTimer()
# timer.my_init(total_cases, args.num_seeds)
# timer.start()

print(total_iters * args.num_seeds * len(args.datasets))
list_keys = []
list_vals = []
cases_finished = 0
job_id = 0
for key, value in param_dict.items():
    if (isinstance(value, list) and len(value) > 1):
        if isinstance(key, str) or isinstance(key, int) or isinstance(key, float):
            params_to_change.extend([key])
        else:
            params_to_change.extend(key)
    if isinstance(key, list) or isinstance(key, tuple):
        if isinstance(key, tuple):
            keyy = list(key)
        for k in keyy:
            if k in args.change_params:
                params_to_change.extend([k])
    else:
        if key in args.change_params:
            params_to_change.extend(key)
    list_keys.append(key)
    if hasattr(value, "__len__") and not isinstance(value, str):
        list_vals.append(value)
    else:
        list_vals.append([value])

zipped_params = list(itertools.product(*list_vals))

pid_list = []
otus = True
metabs = True
for d_tmp in args.datasets:
    if 'SEMISYN' in d_tmp:
        data_str = ''
        if 'both' in d_tmp or 'otus' in d_tmp:
            data_str += ' --data_otu ' + d_tmp + '_REPLACETHIS/seqs.pkl'
        if 'both' in d_tmp or 'metabs' in d_tmp:
            data_str += ' --data_met ' + d_tmp + '_REPLACETHIS/mets.pkl'
        res_data_name = d_tmp.split('/processed/')[-1]
        if 'both' in d_tmp:
            data_str += ' --dtype metabs otus'
        elif 'otus' in d_tmp:
            data_str += ' --dtype otus'
        elif 'metabs' in d_tmp:
            data_str += ' --dtype metabs'
    else:
        if isinstance(d_tmp, tuple):
            # dataset=[]
            data_str = ''
            for dat in list(d_tmp):
                if 'mets' in dat:
                    data_str += ' --data_met ' + dat
                elif 'seqs' in dat:
                    data_str += ' --data_otu ' + dat
                else:
                    ValueError('Unclear if data is metabolomics or otus')
            res_data_name = '/'.join(d_tmp[0].split('.pkl')[0].split('/')[-2:]) + '_' + \
                            d_tmp[1].split('.pkl')[0].split('/')[-1]
            data_str += ' --dtype metabs otus'

        else:
            dataset = d_tmp
            if 'mets' in dataset:
                data_str = ' --data_met ' + dataset + ' --dtype metabs'
                otus = False
            elif 'seqs' in dataset:
                data_str = ' --data_otu ' + dataset + ' --dtype otus'
                metabs = False
            else:
                ValueError('Unclear if data is metabolomics or otus')
            res_data_name = '/'.join(dataset.split('.pkl')[0].split('/')[-2:])

    cmds = []
    for p in zipped_params:
        my_str = "python3 ./lightning_trainer.py" + data_str
        i = 0
        if isinstance(args.case, str):
            if 'SEMISYN' in d_tmp:
                data_name = res_data_name + '_' + args.case + '_REPLACETHIS_'
            else:
                data_name = res_data_name + '_' + args.case
        else:
            data_name = res_data_name
        for il, l in enumerate(list_keys):
            if il - 1 > len(p):
                break

            if isinstance(l, tuple) and isinstance(p[i], tuple):
                if otus is False and any(['otus' in l[ii] for ii in range(len(l))]):
                    i += 1
                    continue
                elif metabs is False and any(['metabs' in l[ii] for ii in range(len(l))]):
                    i += 1
                    continue
                else:
                    for ii in range(len(l)):
                        if hasattr(p[i][ii], "__len__") and not isinstance(p[i][ii], str):
                            pout = [str(pp) for pp in p[i][ii]]
                            my_str = my_str + ' --' + l[ii] + ' ' + ' '.join(pout)
                            if l[ii] in params_to_change:
                                data_name = data_name + '_' + l[ii] + '_'.join(pout)
                        else:
                            fin = p[i][ii]
                            my_str = my_str + ' --' + l[ii] + ' ' + str(fin)
                            if l[ii] in params_to_change:
                                data_name = data_name + '_' + l[ii] + str(fin)
            elif not isinstance(l, tuple) and isinstance(p[i], tuple):
                if otus is False and 'otus' in l:
                    i += 1
                    continue
                elif metabs is False and 'metabs' in l:
                    i += 1
                    continue
                pout = [str(pp) for pp in p[i]]
                my_str = my_str + ' --' + l + ' ' + ' '.join(pout)
                if l in params_to_change:
                    data_name = data_name + '_' + l + '_'.join(pout)
            else:
                if otus is False and 'otus' in l:
                    i += 1
                    continue
                if otus is False:
                    if isinstance(l, tuple):
                        if 'otus' in l[0] or 'otus' in l[1]:
                            i += 1
                            continue
                elif metabs is False and 'metabs' in l:
                    i += 1
                    continue
                lvar = l
                pi_var = p[i]

                try:
                    my_str = my_str + ' --' + l + ' ' + str(p[i])
                except:
                    import pdb;

                    pdb.set_trace()

                if l in params_to_change:
                    data_name = data_name + '_' + l + str(p[i])
            i += 1

        data_name = data_name.replace('.', 'd')
        if len(data_name) > 251:
            data_name = data_name[:250]
        if args.out_path is not None:
            cmd = my_str + ' --data_name ' + data_name + ' --out_path ' + args.out_path + ' --plot_all_seeds ' + str(
                args.plot_all_seeds)
        else:
            cmd = my_str + ' --data_name ' + data_name + ' --plot_all_seeds ' + str(args.plot_all_seeds)

        job_id += 1
        if 'CDI' in cmd:
            cmd = cmd.replace('kfold', 'loo')

        if 'SEMISYN' in cmd:
            for s in seeds:
                if s >= 10:
                    sst = str(s)[-1]
                else:
                    sst=s
                cmd_ = cmd.replace('REPLACETHIS', str(sst))
                cmd_ = cmd_ + f' --seed {s}'
                print(cmd_)
                slurm_job=slurm_single_script.format(cmd_, data_name+str(s), args.num_threads)
                f = open(f"mditre_{job_id}_{s}.job", "w")
                f.write(slurm_job)
                f.close()
                os.system(f"sbatch mditre_{job_id}_{s}.job")

        else:
            if 'CDI' in cmd or 'cdi' in cmd:
                num_threads = 40
            else:
                num_threads=args.num_threads
            slurm_batch_params={'array':f'1-{args.num_seeds}%{args.max_load}',
                                'output':'array_%A-%a',
                                'time':'04:00:00', 'mem':'300G','partition':'bwh_comppath'}
            # slurm_cmd = dict_to_slurm(slurm_batch_params)
            # cmd = slurm_cmd + cmd
            slurm_job = slurm_script.format(cmd, args.num_seeds, args.max_load, num_threads)
            f = open(f"mditre_{job_id}.job", "w")
            f.write(slurm_job)
            f.close()
            os.system(f"sbatch mditre_{job_id}.job")
    # p=subprocess.Popen(["sbatch", f"mditre_{job_id}.job"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # out, err = p.communicate()
    # batch_job_id = out.decode("utf-8").split(' ')[-1].split('\n')[0]
    # s = Slurm(f"mditre_{job_id}.job",
    #           {"mail-type":"FAIL","mail-user":"jjdawkins@bwh.harvard.edu","mem":"300G",
    #            "partition":"bwh_comppath","time":"03:00:00","array":f"0-{args.num_seeds-1}%{args.max_load}"},
    #           log_dir='slurm-logs', date_in_name=False)
    # batch_job_id = s.run(slurm_job)

#     res_command = f'python3 ./get_results.py --logs_dir {args.out_path} --changed_vars {params_to_change} ' \
#                   f'--command_list {cmd} --plot_metabolite_structures {args.plot_metabolite_structures} ' \
#                   f'--case {args.case}'
#     res_paths.append(args.out_path + '/' + cmd.split('--data_name ')[-1].split(' ')[0] + f'/res_to_compare_{args.case}.csv')
#     results_job = slurm_results_script.format(res_command, batch_job_id)
#     f = open(f"results_{job_id}.job","w")
#     f.write(results_job)
#     f.close()
#     p=subprocess.Popen(["sbatch", f"results_{job_id}.job"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     res_out, res_err = p.communicate()
#     res_job_id = res_out.decode("utf-8").split(' ')[-1].split('\n')[0]
# #     # os.system(f"sbatch results_{job_id}.job")
# #     # res_job_id = os.getenv('SLURM_JOB_ID')
# #
# #     # s_res = Slurm(f"sbatch results_{job_id}-{batch_job_id}.job",
# #     #               {"mail-type":"FAIL","mail-user":"jjdawkins@bwh.harvard.edu",
# #     #                "mem":"50G","partition":"bwh_comppath","time":"00:10:00"}, log_dir='slurm-logs', date_in_name=False)
# #     # res_job_id = s_res.run(res_command, depends_on=[batch_job_id])
# #
# #     # res_job_id = tmp_res.decode("utf-8").split(' ')[-1].split('\n')[0]
# #
# #     outer_iter += 1
#     pid_list.append(res_job_id)
# #
# fin_results_script = '''#!/bin/bash
# #SBATCH --job-name=final_results   # Job name
# #SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=jjdawkins@bwh.harvard.edu   # Where to send mail
# #SBATCH --mem=10G                   # Job Memory
# #SBATCH --output=slurm-logs/results_%j.log    # Standard output and error log
# #SBATCH --partition bwh_comppath
# #SBATCH -t 0-0:10
# # SBATCH --dependency=afterany:{1}
# pwd; hostname; date
#
# {0} &
#
# echo {0}
#
# date
# '''
# cmd = f"python3 group_results.py --res_paths {res_paths}"
# #
# res_script = fin_results_script.format(cmd, ':'.join(pid_list))
# f = open(f"fin_results.job", "w")
# f.write(fin_results_script.format(cmd,':'.join(pid_list)))
# f.close()
# os.system("sbatch fin_results.job")
# #
# # # s_res_fin = Slurm("fin_results.job",{"mail-type":"FAIL","mail-user":"jjdawkins@bwh.harvard.edu",
# # #                                      "mem":"10G","partition":"bwh_comppath","time":"00:10:00"},
# # #                    log_dir='slurm-logs', date_in_name=False)
# # # fin_job_id = s_res_fin.run(cmd, depends_on=pid_list)

