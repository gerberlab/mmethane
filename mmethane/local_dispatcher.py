'''
CURRENTLY:
- k=1 for forward binary concrete function that calculates detector & rule selectors
- Post-anneal length is 25% of total epochs instead of 10%
- Variance for kappa is args.multiplier*calculated variance
- Epsilon is currently 10% x min-non-zero value (before was 1)
- Added back in rule and detector BC loss!!!!
- Kappa prior is truncated normal!!
TO CHANGE:
- temperatures for detector selectors?
- epsilon back to 1?
- metabolite_divider to 2?
'''
import sys
import os
sys.path.append(os.path.abspath(".."))
import subprocess
import pickle
import itertools
import numpy as np
import time
import argparse
import datetime
import pandas as pd
import requests
try:
    import pubchempy as pcp
except:
    pass
import re
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import pickle as pkl
import copy
import json
import asyncio
try:
    from rdkit.Chem import Draw,MolFromInchi,AllChem
    from rdkit import Chem, DataStructs
except:
    pass

import matplotlib.pyplot as plt
from utilities.data_utils import read_ctsv
from get_results import get_results, eval_rules

pid_list=[]

# import time
# from threading import Thread

# class BackgroundTimer(Thread):
#     def my_init(self, total_jobs, num_seeds):
#         self.total_jobs = total_jobs
#         self.jobs_completed = 0
#         self.seeds = num_seeds
#
#     def run(self):
#         while 1:
#             print(f'RUNNING CASES {self.jobs_completed-self.seeds}-{self.jobs_completed} OUT OF {self.total_jobs}')
#             time.sleep(60)

async def run_for_seeds(data_name, cmd, seeds, change_vars, args):
    seed_cmds = []
    for s in seeds:
        if int(s) >=10:
            srep = str(s)[-1]
        else:
            srep = s
        cmd_ = cmd.replace('REPLACETHIS', str(srep))
        cmd_ = cmd_ + ' --seed ' + str(s)

        seed_cmds.append(cmd_)
        args2 = cmd_.split(' ')
        if args.only_get_results != 1:
            print(f'Command {cmd_} sent to lightning_trainer.py')
            # with open('log_new.txt', 'w') as f:
            pid = subprocess.Popen(args2)
            pid_list.append(pid)
            time.sleep(5)
            while sum([x.poll() is None for x in pid_list]) >= args.max_load:
                time.sleep(10)

    while sum([x.poll() is None for x in pid_list]) >= 1:
        await asyncio.sleep(1)
    else:
        if 'data_met' in cmd:
            path_to_dataset = cmd.split('--data_met ')[-1].split(' ')[0]
        else:
            path_to_dataset = cmd.split('--data_otu ')[-1].split(' ')[0]
        case_path = args.out_path + '/' + data_name
        results_path = '/'.join(case_path.split('/')[:-1])
        if not os.path.isdir(case_path):
            return [pd.DataFrame()]
        res_df, rename_str = get_results(case_path, args.case, change_vars, path_to_dataset.replace('REPLACETHIS', str(s)), args)
        if rename_str is not None and rename_str!='':
            if not os.path.isdir(case_path + '_' + rename_str):
                os.rename(case_path, case_path + '_' + rename_str)
        return res_df



if __name__=='__main__':
    async def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_load', default =1, type=int)
        parser.add_argument('--case', default = '', type=str)
        parser.add_argument('--plot_all_seeds', default = 0, type=int)
        parser.add_argument('-r','--only_get_results', default = 0, type=int)
        parser.add_argument('--num_seeds', default =10, type=int)
        parser.add_argument('--seed', default=[
            0,1,2,3,4,5,6,7,8,9
            # 0, 10 , 20, 30, 40, 50, 60, 70, 80, 90,
            # 1, 11, 21, 31, 41, 51, 61, 71, 81, 91,
            #                                    2, 12, 22, 32, 42, 52, 62, 72, 82, 92,
            #
            #                                    3, 13, 23, 33, 43, 53, 63, 73, 83, 93,
            #                                    4, 14, 24, 34, 44, 54, 64, 74, 84, 94,
            #                                    5, 15, 25, 35, 45, 55, 65, 75, 85, 95,
            #                                    6,16,26,36,46,56,66,76,86,96,
            #                                    7, 17, 27,37,47,57,67,77,87,97,
            #                                    8,18,28,38,48,58,68,78,88,98,
            #                                    9,19,29,39,49,59,69,79,89,99,
                                               ]
        , type=int, nargs='+')
        parser.add_argument('--out_path', default = '/Users/jendawk/logs/ERAW_OCT24_mmethane/', type=str)
        parser.add_argument('--change_params', default=[], type=str, nargs='+')
        parser.add_argument('--datasets', default=[
            # '../datasets/SEMISYN/processed/and_otus_1000',
            # '../datasets/SEMISYN/processed/otus_1000',

            # #
            # '../datasets/SEMISYN/processed/both_and_1000',
            # '../datasets/SEMISYN/processed/both_and_36',
            # '../datasets/SEMISYN/processed/both_and_48',
            # '../datasets/SEMISYN/processed/both_and_64',
            # '../datasets/SEMISYN/processed/both_and_128',
            # '../datasets/SEMISYN/processed/both_and_300',
            #
            # '../datasets/SEMISYN/processed/otus_50',
            # '../datasets/SEMISYN/processed/otus_100',
            # '../datasets/SEMISYN/processed/otus_200',

            # '../datasets/SEMISYN/processed/otus_48',
            # '../datasets/SEMISYN/processed/otus_64',
            # '../datasets/SEMISYN/processed/otus_128',
            # '../datasets/SEMISYN/processed/otus_300',
            # '../datasets/SEMISYN/processed/otus_36',
            # '../datasets/SEMISYN/processed/otus_1000',
            # # # #
            # '../datasets/SEMISYN/processed/metabs_1000',
            # '../datasets/SEMISYN/processed/metabs_36',
            # '../datasets/SEMISYN/processed/metabs_48',
            # '../datasets/SEMISYN/processed/metabs_64',
            # '../datasets/SEMISYN/processed/metabs_128',
            # '../datasets/SEMISYN/processed/metabs_300',
            # # # #

            # '../datasets/SEMISYN/processed/and_metabs_1000',
            # '../datasets/SEMISYN/processed/and_metabs_48',
            # '../datasets/SEMISYN/processed/and_metabs_36',
            # '../datasets/SEMISYN/processed/and_metabs_64',
            # '../datasets/SEMISYN/processed/and_metabs_128',
            # '../datasets/SEMISYN/processed/and_metabs_300',
            # #
            # '../datasets/SEMISYN/processed/and_otus_1000',
            # '../datasets/SEMISYN/processed/and_otus_48',
            # '../datasets/SEMISYN/processed/and_otus_36',
            # '../datasets/SEMISYN/processed/and_otus_64',
            # '../datasets/SEMISYN/processed/and_otus_128',
            # '../datasets/SEMISYN/processed/and_otus_300',


            # '../datasets/CDI/processed/cdi_map4/mets.pkl',
            # ('../datasets/CDI/processed/cdi_map4/mets.pkl',
            #  '../datasets/CDI/processed/cdi_cts/seqs.pkl'),
            # '../datasets/ERAWIJANTARI/processed/erawijantari_cts_old/seqs.pkl'
            #
            # '../datasets/SEMISYN/processed/metabs_128',
            # '../datasets/SEMISYN/processed/and_metabs_128',
            # '../datasets/SEMISYN/processed/both_128',
            # '../datasets/SEMISYN/processed/otus_128',

            # '../datasets/SEMISYN/processed/both_metabs_24',
            # '../datasets/SEMISYN/processed/both_metabs_300',
            # '../datasets/SEMISYN/processed/both_metabs_48',
            # '../datasets/SEMISYN/processed/both_metabs_64',
            # '../datasets/SEMISYN/processed/both_metabs_1000',
            #
            # '../datasets/SEMISYN/processed/otus_24',
            # '../datasets/SEMISYN/processed/otus_300',
            # '../datasets/SEMISYN/processed/otus_48',
            # '../datasets/SEMISYN/processed/otus_64',
            # '../datasets/SEMISYN/processed/otus_1000',
            #
            # '../datasets/SEMISYN/processed/metabs_24',
            # '../datasets/SEMISYN/processed/metabs_300',
            # '../datasets/SEMISYN/processed/metabs_48',
            # '../datasets/SEMISYN/processed/metabs_64',
            # '../datasets/SEMISYN/processed/metabs_1000',
            #
            # '../datasets/SEMISYN/processed/and_metabs_24',
            # '../datasets/SEMISYN/processed/and_metabs_300',
            # '../datasets/SEMISYN/processed/and_metabs_48',
            # '../datasets/SEMISYN/processed/and_metabs_64',
            # '../datasets/SEMISYN/processed/and_metabs_1000',

            # '../datasets/SEMISYN/processed/both_128',
            # '../datasets/SEMISYN/processed/both_metabs_24',
            # '../datasets/SEMISYN/processed/both_metabs_300',
            # '../datasets/SEMISYN/processed/both_metabs_48',
            # '../datasets/SEMISYN/processed/both_metabs_64',
            # '../datasets/SEMISYN/processed/both_metabs_1000',
            #

            #
            # ('../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl',
            #
            #  '../datasets/ERAWIJANTARI/processed/erawijantari_cts/seqs.pkl'),
            # '../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl',
            # # # # # #
            '../datasets/ERAWIJANTARI/processed/erawijantari_cts/seqs.pkl',
            # # # #
            # # # # # # # # # # # # # #
            '../datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl',
            ('../datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl',
             '../datasets/FRANZOSA/processed/franzosa_cts/seqs.pkl'),
            '../datasets/FRANZOSA/processed/franzosa_cts/seqs.pkl',
            # #
            # # # # # # # # # # # # # # # # # # #
            '../datasets/IBMDB/processed/ibmdb_pubchem/mets.pkl',
            ('../datasets/IBMDB/processed/ibmdb_pubchem/mets.pkl',
             '../datasets/IBMDB/processed/ibmdb_cts/seqs.pkl'),
            '../datasets/IBMDB/processed/ibmdb_cts/seqs.pkl',
            # #
            # # # # # #
            '../datasets/HE/processed/he_pubchem/2_mets.pkl',
            ('../datasets/HE/processed/he_pubchem/2_mets.pkl',
             '../datasets/HE/processed/he_cts/2_seqs.pkl'),
            # #
            # #
            # # # '../datasets/HE/processed/he_cts/2_seqs.pkl'
            # # # # # # # # #
            # # # # # # #
             '../datasets/WANG/processed/wang_pubchem/mets.pkl',
            ('../datasets/WANG/processed/wang_pubchem/mets.pkl',
             '../datasets/WANG/processed/wang_cts/seqs.pkl'),
            '../datasets/WANG/processed/wang_cts/seqs.pkl',
            # # # #
            # # # # #
            '../datasets/CDI/processed/cdi_pubchem/mets.pkl',
            ('../datasets/CDI/processed/cdi_pubchem/mets.pkl',
             '../datasets/CDI/processed/cdi_cts/seqs.pkl'),


            # '../datasets/ERAWIJANTARI/processed/erawijantari_map4/mets.pkl',
            # '../datasets/FRANZOSA/processed/franzosa_map4/mets.pkl',
            # '../datasets/HE/processed/he_map4/2_mets.pkl',
            # '../datasets/IBMDB/processed/ibmdb_map4/mets.pkl',
            # '../datasets/WANG/processed/wang_map4/mets.pkl',
            # '../datasets/CDI/processed/cdi_map4/mets.pkl',
            #
            # '../datasets/ERAWIJANTARI/processed/erawijantari_infomax/mets.pkl',
            # '../datasets/FRANZOSA/processed/franzosa_infomax/mets.pkl',
            # '../datasets/HE/processed/he_infomax/2_mets.pkl',
            # '../datasets/IBMDB/processed/ibmdb_infomax/mets.pkl',
            # '../datasets/WANG/processed/wang_infomax/mets.pkl',
            # '../datasets/CDI/processed/cdi_infomax/mets.pkl',
            #
            # '../datasets/ERAWIJANTARI/processed/erawijantari_morgan/mets.pkl',
            # '../datasets/FRANZOSA/processed/franzosa_morgan/mets.pkl',
            # '../datasets/HE/processed/he_morgan/2_mets.pkl',
            # '../datasets/IBMDB/processed/ibmdb_morgan/mets.pkl',
            # '../datasets/WANG/processed/wang_morgan/mets.pkl',
            # '../datasets/CDI/processed/cdi_morgan/mets.pkl',
            #
            # '../datasets/ERAWIJANTARI/processed/erawijantari_mqn/mets.pkl',
            # '../datasets/FRANZOSA/processed/franzosa_mqn/mets.pkl',
            # '../datasets/HE/processed/he_mqn/2_mets.pkl',
            # '../datasets/IBMDB/processed/ibmdb_mqn/mets.pkl',
            # '../datasets/WANG/processed/wang_mqn/mets.pkl',
            # '../datasets/CDI/processed/cdi_mqn/mets.pkl',

                            ], type=str, nargs='+')
        args = parser.parse_args()

        params_to_change = []

        if args.seed is None:
            seeds=np.arange(args.num_seeds)
        else:
            seeds = args.seed
        tasks=[]
        # SUHAS PARAMETERS
        param_dict = {
            "optimizer":["NAdam"],
            "bc_loss_type":"sum",
            "filter_data": 1,
            'otus_n_d':[10],
            "use_ray":1,
    "plot_all_seeds": 1,
    "epochs": [3000],
    "parallel": 6,
            # 'annealing_limit':[(0,0.75),(0, 0.7)],
            # ('metabs_lr_alpha','otus_lr_alpha','lr_beta'):[(0.001,0.001,0.001)],
            ('otus_init_clusters','metabs_init_clusters'):[
                ('kmeans','kmeans'),

                                                           ],
            ('metabs_use_pca','otus_use_pca'):[(0,0)],
            'metabs_lr_alpha':[0.001],
            'patience':[100],
            'metabs_lr_thresh':[0.0005],
            "metabs_min_k_bc":[0.5],
            "otus_lr_thresh":[0.0001],
            'schedule_lr':[0],
            # ('otus_adj_kappa_loss','metabs_adj_kappa_loss'):[(0,0),(1,1)],
            ('otus_adj_detector_loss','metabs_adj_detector_loss','otus_adj_kappa_loss','metabs_adj_kappa_loss'):
                [(0,0,0,0)],
            'adj_rule_loss':[0],
            'kmeans_noise':[1],
            "early_stopping":[1],
            # 'otus_lr_thresh': [0.0001],
            ('otus_adj_n_d','metabs_adj_n_d'):[(0,0)],
            'init_w_LR': [0],'init_mult':[0.1],
            'eta_min':[0.00001],
        'alpha_init_zeros':[1]
            # 'otus_kappa_mult':[1e3,10,1]

  }

        # num_det_dict = {'he_':{'metabs_n_d':10, 'otus_n_d':30},
        #                 'ibmdb': {'metabs_n_d': 30, 'otus_n_d':10},
        #                 'cdi':{'metabs_n_d'}}

       #                 /2_mets_2_seqs':25, 'he_pubchem/2_mets':20,
       # 'ibmdb_pubchem/mets':24, 'ibmdb_pubchem/mets_seqs':25, 'cdi_cts/seqs':21,
       # 'franzosa_pubchem/mets':23, 'franzosa_pubchem/mets_seqs':26,
       # 'franzosa_ra/seqs':23, 'he_cts/2_seqs':26, 'ibmdb_ra/seqs':21,
       # 'wang_pubchem/mets':22, 'wang_pubchem/mets_seqs':25,
       # 'erawijantari_pubchem/mets_seqs':27, 'erawijantari_pubchem/mets':24,
       # 'wang_ra/seqs':22,
       # 'erawijantari_ra/seqs':23,
       # 'cdi_pubchem/mets_seqs':28, 'cdi_pubchem/mets':26}

        total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__') and not isinstance(v, str)])
        total_cases = total_iters*args.num_seeds*len(args.datasets)
        # import pdb; pdb.set_trace()
        # timer=BackgroundTimer()
        # timer.my_init(total_cases, args.num_seeds)
        # timer.start()

        print(total_iters*args.num_seeds*len(args.datasets))
        list_keys = []
        list_vals = []
        cases_finished = 0
        added=False
        variable_n = False
        for key, value in param_dict.items():
            if ('n_d' in key or 'n_r' in key) and value==999:
                variable_n=True
                continue
            if (isinstance(value, list) and len(value)>1):
                if isinstance(key,str) or isinstance(key, int) or isinstance(key, float):
                    params_to_change.extend([key])
                else:
                    params_to_change.extend(key)
            if isinstance(key, list) or isinstance(key, tuple):
                if isinstance(key, tuple):
                    keyy = list(key)
                for k in keyy:
                    if k in args.change_params or (isinstance(value, list) and len(value)>1):
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
                data_str=''
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
                nsubj = int(d_tmp.split('_')[-1])
                if variable_n:
                    rfunc = lambda x: int(np.round((-9.32865658e-06)*(x**2)+ (1.64572593e-02)*x + 4.93365962e+00))
                    dfunc = lambda x: int(np.round((-3.61672845e-05)*(x**2)+ (6.46201083e-02)*x + 1.74270484e+00))
                    data_str += f' --n_r {rfunc(nsubj)}'
                    if 'both' in d_tmp or 'metabs' in d_tmp:
                        data_str += f' --metabs_n_d {dfunc(nsubj)}'
                    if 'both' in d_tmp or 'otus' in d_tmp:
                        data_str += f' --otus_n_d {dfunc(nsubj)}'
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

                    res_data_name = '/'.join(d_tmp[1].split('.pkl')[0].split('/')[-2:]) + '_' + \
                                d_tmp[0].split('.pkl')[0].split('/')[-1]
                    data_str += ' --dtype metabs otus'

                else:
                    dataset = d_tmp
                    if 'mets' in dataset:
                        data_str = ' --data_met ' + dataset + ' --dtype metabs'
                        if 'seqs' not in dataset:
                            otus=False
                    elif 'seqs' in dataset:
                        data_str = ' --data_otu ' + dataset + ' --dtype otus'
                        if 'mets' not in dataset:
                            metabs=False
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
                for il,l in enumerate(list_keys):
                    if il-1 > len(p):
                        break

                    if isinstance(l, tuple) and isinstance(p[i], tuple):
                        if otus is False and all(['otus' in l[ii] for ii in range(len(l))]):
                            i += 1
                            continue
                        elif metabs is False and all(['metabs' in l[ii] for ii in range(len(l))]):
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
                            i+=1
                            continue
                        lvar = l
                        pi_var = p[i]

                        try:
                            my_str = my_str + ' --' + l + ' ' + str(p[i])
                        except:
                            import pdb; pdb.set_trace()

                        if l in params_to_change:
                            data_name = data_name + '_' + l + str(p[i])
                    i += 1

                data_name = data_name.replace('.','d')
                if len(data_name)>251:
                    data_name = data_name[:250]
                if args.out_path is not None:
                    cmd=my_str + ' --data_name ' + data_name + ' --out_path ' + args.out_path + ' --plot_all_seeds ' + str(args.plot_all_seeds)
                else:
                    cmd=my_str + ' --data_name ' + data_name + ' --plot_all_seeds ' + str(args.plot_all_seeds)


                if 'CDI' in cmd:
                    cmd = cmd.replace('kfold','loo')
                # print(cmd)
                # import pdb; pdb.set_trace()
                cmds.append(cmd)

                # if 'SEMISYN' in args.case or 'SEMISYN' in data_name:

                tasks.append(asyncio.ensure_future(run_for_seeds(data_name, cmd, seeds, params_to_change, args)))
                while sum([x.poll() is None for x in pid_list]) >= args.max_load:
                    print(f'RUNNING CASES {cases_finished}-{cases_finished+len(seeds)} OUT OF {total_cases}')
                    time.sleep(1)
                # timer.jobs_completed += len(seeds)
                cases_finished += len(seeds)

            # timer.join()
            results = asyncio.gather(*tasks)
            await results
            if 'SEMISYN' not in d_tmp:
                print('Getting results to compare for cases:\n' + '\n'.join(cmds))
                try:
                    try:
                        final = pd.concat([pd.concat(r, axis=0) for r in results.result()], axis=0)
                    except:
                        final = pd.concat([r for r in results.result()], axis=0)
                    i = 1
                    if res_data_name[-1] == '/':
                        res_data_name = res_data_name[:-1]
                    case_res = args.case
                    while os.path.isfile(f'{args.out_path}/{res_data_name}_results_to_compare_{case_res}.csv'):
                        case_res = case_res + '_' + str(i)
                        i += 1
                    final.to_csv(f'{args.out_path}/{res_data_name}_results_to_compare_{case_res}.csv')
                except:
                    continue

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()