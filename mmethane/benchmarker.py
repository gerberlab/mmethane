import os
import sys

import pandas as pd
import numpy as np
import pickle as pkl
import argparse
sys.path.append(os.path.abspath(".."))
os.environ['QT_QPA_PLATFORM']='offscreen'
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('agg')
import sklearn

from utilities.util import cv_loo_splits, cv_kfold_splits, split_and_preprocess_dataset, merge_datasets

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from helper_plots import plot_input_data

class benchmarker():
    def __init__(self, dataset_dict, y, args, perturbed_mets = None, seed=0, path='.'):
        self.dataset_dict=dataset_dict
        self.y = y
        self.args = args
        self.num_subjects = self.y.shape[0]
        self.perturbed_mets = perturbed_mets
        self.seed = seed
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'

    def get_cv_splits(self):
        # Get train-test data splits for cross-val
        if self.args.cv_type == 'None':
            self.train_splits = [np.arange(self.num_subjects)]
            self.test_splits = self.train_splits
        elif self.args.cv_type == 'loo':
            self.train_splits, self.test_splits = cv_loo_splits(np.zeros(self.num_subjects), self.y)
        else:
            self.train_splits, self.test_splits = cv_kfold_splits(np.zeros(self.num_subjects), self.y,
                num_splits=self.args.kfolds, seed=self.seed)


    def train_ensemble_models(self, ensemble_model='RF'):
        grid_dict = {
            'bootstrap':[True],
            'n_estimators':[50,100],
            'max_depth':[None],
            'max_features':[None,'sqrt'],
            'min_samples_split':[2,9],
            'min_samples_leaf':[1,5],
            'learning_rate':[1e-5,5e-5,0.0001,5e-4,0.001,5e-3,0.01,0.05,0.1,0.5,1,5,10]
        }
        self.get_cv_splits()
        scores = {'F1': [], 'AUC': [], 'Case Acc':[],'Ctrl Acc':[]}
        coef_dfs = []
        pred_dfs = []
        best_params = []
        for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
            train_dict, test_dict = split_and_preprocess_dataset(self.dataset_dict, train_ids, test_ids, preprocess=True,
                                                                 logdir=self.path)

            plot_input_data(train_dict, self.path)
            x_tr = []
            for k, v in train_dict.items():
                x_tr.append(v['X'])
            X_tr = pd.concat(x_tr,axis=1)

            x_ts = []
            for k, v in test_dict.items():
                x_ts.append(v['X'])
            X_ts = pd.concat(x_ts,axis=1)

            if ensemble_model.lower()=='rf':
                rf = RandomForestClassifier()
            elif ensemble_model.lower()=='adaboost':
                rf = AdaBoostClassifier()
            elif ensemble_model.lower()=='gradboost':
                rf = GradientBoostingClassifier()

            input_dict = rf._parameter_constraints
            rm_keys = list(set(list(grid_dict.keys())) - set(list(input_dict.keys())))
            for key in rm_keys:
                grid_dict.pop(key)
            rf_random = sklearn.model_selection.GridSearchCV(estimator=rf, param_grid=grid_dict, scoring=self.args.scorer,
                                                             cv=int(np.min([self.args.inner_folds,self.y.iloc[train_ids].sum()])),
                                                           )


            rf_random.fit(X_tr, self.y.iloc[train_ids])
            best_model = rf_random.best_estimator_
            f1 = best_model.score(X_ts, self.y.iloc[test_ids])
            coef_df = pd.DataFrame(best_model.feature_importances_.squeeze(), index = X_tr.columns.values, columns = ['Fold ' + str(i)])
            pred_probs = best_model.predict_proba(X_ts)
            try:
                auc = sklearn.metrics.roc_auc_score(self.y.iloc[test_ids].values, pred_probs[:, 1] > 0.5)
            except:
                auc=0
            coef_dfs.append(coef_df)
            scores['F1'].append(f1)
            scores['AUC'].append(auc)
            # a, f = by_hand_calc(self.y.iloc[test_ids].values, pred_probs[:, 1], 0.5)
            if args.cv_type!='loo':
                ctrls, case = self.y.iloc[test_ids].values==0, self.y.iloc[test_ids].values!=0
                case_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[case], (pred_probs[:, 1][case] > 0.5).astype(int)), 3)
                ctrl_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[ctrls], (pred_probs[:, 1][ctrls] > 0.5).astype(int)), 3)
            else:
                case_acc,ctrl_acc = 0,0
            scores['Case Acc'].append(case_acc)
            scores['Ctrl Acc'].append(ctrl_acc)
            # scores['AUC by hand'].append(a)
            # scores['F1 by hand'].append(f)
            best_params.append(pd.DataFrame(rf_random.best_params_, index = ['Fold {0}'.format(i)]))
            pred_dfs.append(pd.DataFrame({'True': self.y.iloc[test_ids].values, 'Pred probs': pred_probs[:, 1],
                                          'Outer Fold': [i]*len(test_ids)}, index = test_ids))

            if args.cv_type!='loo':
                pd.DataFrame(rf_random.cv_results_).to_csv(self.path + '/cv_results_fold_{0}.csv'.format(i))


        pd.concat(best_params).to_csv(self.path +'/best_params.csv'.format(self.args.run_name))
        pd.DataFrame(scores).to_csv(self.path +'/scores.csv'.format(self.args.run_name))
        coef_df_full = pd.concat(coef_dfs, axis=1)
        if self.perturbed_mets is not None:
            pert_df = coef_df_full.index.isin(self.perturbed_mets)
            coef_df_full = pd.concat({'Perturbed': coef_df_full.loc[self.perturbed_mets],
                                      'Un-perturbed': coef_df_full.loc[~pert_df]}, names = ['Metabolite perturbed'])
        coef_df_full.to_csv(self.path+'/coefs.csv'.format(self.args.run_name))
        pd.concat(pred_dfs).to_csv(self.path +'/preds.csv'.format(self.args.run_name))


    def train_l1_model(self):
        self.get_cv_splits()
        scores = {'F1': [], 'AUC': [], 'Case Acc':[],'Ctrl Acc':[]}
        coef_dfs = []
        pred_dfs = []
        l1_lambda = []
        lambda_min = 0.001
        path_len = 100
        l_max=100
        if args.otu_tr == 'standardize':
            print('OTUS standardized')
        elif args.otu_tr=='clr':
            print('OTUS clr')
        elif args.otu_tr=='sqrt':
            print('OTUS sqrt')
        else:
            print('No OTU transform')
        for i, (train_ids, test_ids) in enumerate(zip(self.train_splits, self.test_splits)):
            train_dict, test_dict = split_and_preprocess_dataset(self.dataset_dict, train_ids, test_ids,
                                                                 preprocess=True, standardize_otus=args.otu_tr=='standardize',
                                                                 standardize_from_training_data=True, clr_transform_otus=args.otu_tr=='clr',
                                                                 sqrt_transform=args.otu_tr=='sqrt',
                                                                 logdir=self.path)
            plot_input_data(train_dict, self.path)

            x_tr = []
            for k, v in train_dict.items():
                x_tr.append(v['X'])
            X_tr = pd.concat(x_tr, axis=1)

            x_ts = []
            for k, v in test_dict.items():
                x_ts.append(v['X'])
            X_ts = pd.concat(x_ts, axis=1)

            if (len(np.unique(self.y.iloc[train_ids]))==1 or len(np.unique(self.y.iloc[test_ids]))==1) and args.cv_type!='loo':
                continue
            bval = True
            lam = 0
            while (bval):
                lam += 0.1
                model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
                try:
                    model2.fit(X_tr, self.y.iloc[train_ids])
                except:
                    print('error')
                if np.sum(np.abs(model2.coef_)) < 1e-8:
                    l_max = lam + 1
                    bval = False
            print(l_max)
            l_path = np.logspace(np.log10(l_max * lambda_min), np.log10(l_max), path_len)
            # l_path = np.logspace(np.log10(lambda_min), np.log10(l_max), path_len)
            Cs = [1/l for l in l_path]
            if X_tr.shape[0]<150:
                solver='liblinear'
            else:
                solver='saga'
            model = sklearn.linear_model.LogisticRegressionCV(cv = int(np.min([self.args.inner_folds,
                                                                           self.y.iloc[train_ids].sum()])),
                                                              penalty = 'l1', scoring = self.args.scorer,
                                                              Cs = Cs, solver = solver, class_weight='balanced',
                                                              random_state=self.seed)
            model.fit(X_tr, self.y.iloc[train_ids])
            with open(self.path + '/reg_params.txt','w') as f:
                f.write(str(1/model.C_))
            score = model.score(X_ts, self.y.iloc[test_ids])
            coef_df = pd.DataFrame(model.coef_.squeeze(), index = X_tr.columns.values, columns = ['Fold ' + str(i)])
            pred_probs = model.predict_proba(X_ts)

            f1_score = sklearn.metrics.f1_score(self.y.iloc[test_ids].values, pred_probs[:, 1] > 0.5, average='weighted')
            try:
                auc = sklearn.metrics.roc_auc_score(self.y.iloc[test_ids].values, pred_probs[:, 1]>0.5)
            except:
                auc=0
            coef_dfs.append(coef_df)
            scores['F1'].append(f1_score)
            scores['AUC'].append(auc)
            if args.cv_type!='loo':
                ctrls, case = self.y.iloc[test_ids].values==0, self.y.iloc[test_ids].values!=0
                case_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[case], (pred_probs[:, 1][case] > 0.5).astype(int)), 3)
                ctrl_acc = np.round(accuracy_score(self.y.iloc[test_ids].values[ctrls], (pred_probs[:, 1][ctrls] > 0.5).astype(int)), 3)
                # a, f = by_hand_calc(self.y.iloc[test_ids].values, pred_probs[:, 1], 0.5)
            else:
                case_acc = 0
                ctrl_acc = 0
            scores['Case Acc'].append(case_acc)
            scores['Ctrl Acc'].append(ctrl_acc)

            l1_lambda.append(model.C_)
            pred_dfs.append(pd.DataFrame({'True': self.y.iloc[test_ids].values, 'Pred probs': pred_probs[:, 1],
                                          'Outer Fold': [i]*len(test_ids)}, index =self.y.index.values[test_ids]))
            with open(self.path + '/test_dataset.pkl','wb') as f:
                pkl.dump(X_ts, f)
            with open(self.path + '/train_dataset.pkl','wb') as f:
                pkl.dump(X_tr, f)
            with open(self.path + '/y.pkl','wb') as f:
                pkl.dump(self.y, f)


        pd.DataFrame(scores).to_csv(self.path +'/scores.csv')
        coef_df_full = pd.concat(coef_dfs, axis=1)
        if self.perturbed_mets is not None:
            pert_df = coef_df_full.index.isin(self.perturbed_mets)
            coef_df_full = pd.concat({'Perturbed': coef_df_full.loc[self.perturbed_mets],
                                      'Un-perturbed': coef_df_full.loc[~pert_df]}, names = ['Metabolite perturbed'])
        coef_df_full.to_csv(self.path +'/coefs.csv')
        pd.concat(pred_dfs).to_csv(self.path +'/preds.csv')
        print(l1_lambda)

def process_benchmark_coefs(path, seed_vec = None):
    seed_res = {}
    row_names=None
    if seed_vec is not None:
        seed_vec_names = [f'seed_{s}' for s in seed_vec]
    else:
        seed_vec_names = None
    for seed_folder in os.listdir(path):
        if '.' in seed_folder:
            continue
        if seed_vec_names is not None and seed_folder not in seed_vec_names:
            continue
        try:
            res_df = pd.read_csv(path + '/' + seed_folder + '/coefs.csv', index_col = 0)
        except:
            continue
        # if row_names is None:
        row_names = res_df.index.values
        seed_res[seed_folder] = res_df.loc[row_names]
    all_coefs = pd.concat(seed_res.values(), axis=1, join='inner')
    all_coefs_mean = all_coefs.mean(axis=1)
    all_coefs_median = all_coefs.median(axis=1)
    all_coefs_std = all_coefs.std(axis=1)
    all_coefs_5p = all_coefs.quantile(0.05, axis=1)
    all_coefs_95p = all_coefs.quantile(0.95, axis=1)
    all_coefs_25p = all_coefs.quantile(0.25, axis=1)
    all_coefs_75p = all_coefs.quantile(0.75, axis=1)
    sort_ix = np.flip(np.argsort(np.abs(all_coefs_median)))

    res = pd.concat([all_coefs_median, all_coefs_5p, all_coefs_95p, all_coefs_25p, all_coefs_75p,all_coefs_mean, all_coefs_std], keys=['median', '5%', '95%', '25%', '75%','mean','stdev'], axis=1)

    key_func = lambda x: x.abs()
    res_sorted = res.sort_values(by=['median','mean'], ascending=False, key = key_func)
    if seed_vec is not None:
        res_sorted.to_csv(path + '/' + f'coef_res_{"".join([str(s) for s in seed_vec])}.csv')
    else:
        res_sorted.to_csv(path + '/' + f'coef_res.csv')

def process_benchmark_results(path):
    seed_res = {}
    for seed_folder in os.listdir(path):
        if '.' in seed_folder:
            continue
        try:
            res_df = pd.read_csv(path + '/' + seed_folder + '/preds.csv', index_col = 0)
        except:
            continue
        cv_f1 = np.round(f1_score(res_df['True'], (res_df['Pred probs'] > 0.5).astype(int)), 3)
        cv_auc = np.round(roc_auc_score(res_df['True'], res_df['Pred probs']), 3)
        cv_f1_weighted=np.round(f1_score(res_df['True'], (res_df['Pred probs'] > 0.5).astype(int), average='weighted'), 3)
        ctrls, case = res_df['True'].values==0, res_df['True'].values!=0

        seed_res[seed_folder] = {'F1': cv_f1, 'F1_weighted':cv_f1_weighted, 'AUC': cv_auc}

    seed_res['Mean'] = {'F1': np.round(np.mean([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                        'F1_weighted':np.round(np.mean([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.mean([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                        }
    seed_res['St dev'] = {'F1': np.round(np.std([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                          'F1_weighted': np.round(np.std([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.std([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                          }
    seed_res['Median'] = {'F1': np.round(np.median([seed_res[s]['F1'] for s in seed_res.keys()]), 3),
                          'F1_weighted': np.round(np.median([seed_res[s]['F1_weighted'] for s in seed_res.keys()]), 3),
                        'AUC': np.round(np.median([seed_res[s]['AUC'] for s in seed_res.keys()]), 3),
                          }

    seed_res['25%'] = {'F1': np.round(np.percentile([seed_res[s]['F1'] for s in seed_res.keys()], 25), 3),
                       'F1_weighted': np.round(np.percentile([seed_res[s]['F1_weighted'] for s in seed_res.keys()], 25), 3),
                        'AUC': np.round(np.percentile([seed_res[s]['AUC'] for s in seed_res.keys()], 25), 3),
                          }

    seed_res['75%'] = {'F1': np.round(np.percentile([seed_res[s]['F1'] for s in seed_res.keys()], 75), 3),
                       'F1_weighted': np.round(np.percentile([seed_res[s]['F1_weighted'] for s in seed_res.keys()], 75), 3),
                        'AUC': np.round(np.percentile([seed_res[s]['AUC'] for s in seed_res.keys()], 75), 3),
                          }
    pd.DataFrame(seed_res).T.to_csv(path + '/' + f'multiseed_results.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    parser.add_argument('--run_name', metavar='DIR',
                        help='run_name',
                        default='TEST')
    parser.add_argument('--met_data', metavar='DIR',
                        help='path to dataset',
                        default='/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/FRANZOSA/processed/franzosa_pubchem/mets.pkl')
    parser.add_argument('--otu_data', metavar='DIR',
                        help='path to dataset',
                        default='/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/FRANZOSA/processed/franzosa_ra/seqs.pkl')
    parser.add_argument('--seed', type=int, default=[0,1,2,3],
                        help='Set random seed for reproducibility', nargs = '+')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'None'],
                        help='Choose cross val type')
    parser.add_argument("--scorer",type=str, default='f1',choices=['f1', 'roc_auc', 'accuracy'])
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--case', type=str, default="grid")
    parser.add_argument('--model', type=str.lower, default='LR', choices=['rf','lr','gradboost','adaboost'])
    parser.add_argument('--dtype', type=str, default=['metabs'], nargs='+')
    parser.add_argument('--full', type=int, default=0)
    parser.add_argument('-log','--log_transform_otus',action='store_true')
    parser.add_argument('--out_path',type=str, default='logs/')
    parser.add_argument('--otu_tr', type=str,default='none')
    parser.add_argument('--no_filter', action='store_true')

    args,_ = parser.parse_known_args()

    print('')
    print('START')
    for k, v in args.__dict__.items():
        print(k, v)
    # args.run_name = '_' + args.met_data.split('/')[-2]
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.isdir(args.out_path + '/' + args.run_name ):
        os.mkdir(args.out_path + '/' + args.run_name)
    # if not os.path.isdir(args.out_path + '/' + args.model + '/'+ args.run_name):
    #     os.mkdir(args.out_path + '/' + args.model + '/' +args.run_name)

    if isinstance(args.seed, int):
        seed_vec = [args.seed]
    else:
        seed_vec = args.seed.copy()
    ds_str = ''
    for seed in seed_vec:
        seed_path = args.out_path + '/' + args.run_name + '/seed_{0}'.format(seed)

        if not os.path.isdir(seed_path):
            os.mkdir(seed_path)

        dataset_dict = {}
        if 'metabs' in args.dtype or 'both' in args.dtype:
            dataset_dict['metabs'] = pd.read_pickle(args.met_data)
            if args.no_filter:
                dataset_dict['metabs']['preprocessing']={}
            if 'distances' not in dataset_dict['metabs'].keys():
                dataset_dict['metabs']['distances'] = dataset_dict['metabs']['tree_distance']
            if not isinstance(dataset_dict['metabs']['distances'], pd.DataFrame) and \
                    dataset_dict['metabs']['distances'].shape[0] == dataset_dict['metabs']['X'].shape[1]:
                dataset_dict['metabs']['distances'] = pd.DataFrame(dataset_dict['metabs']['distances'],
                                                                       index=dataset_dict['metabs']['X'].columns.values,
                                                                       columns=dataset_dict['metabs'][
                                                                           'X'].columns.values)
            if args.full == 0:
                mets = dataset_dict['metabs']['distances'].columns.values
                # pert = pd.read_pickle(args.met_data.replace('mets.pkl','perturbed.pkl'))
                # mets = pert['metabs'][0]
                # mets.extend(pert['metabs'][1])
                print(mets)
                dataset_dict['metabs']['X'] = dataset_dict['metabs']['X'][mets]
            print(f'{dataset_dict["metabs"]["X"].shape[1]} metabolites in data')


        if 'otus' in args.dtype or 'both' in args.dtype:
            dataset_dict['otus'] = pd.read_pickle(args.otu_data)
            if args.no_filter:
                dataset_dict['otus']['preprocessing']={}
            if args.full == 0:
                otus = dataset_dict['otus']['distances'].columns.values
                dataset_dict['otus']['X'] = dataset_dict['otus']['X'][otus]
            if args.log_transform_otus:
                X = dataset_dict['otus']['X'].values
                dataset_dict['otus']['X'] = pd.DataFrame(np.log(X + 1e-6), columns=dataset_dict['otus']['X'].columns.values,
                                                         index=dataset_dict['otus']['X'].index.values)
            print(f'{dataset_dict["otus"]["X"].shape[1]} otus in data')

        dataset_dict, y = merge_datasets(dataset_dict)
        if 'metabs' in dataset_dict.keys():
            y = pd.Series(y, index=dataset_dict['metabs']['y'].index.values)
        else:
            y = pd.Series(y, index=dataset_dict['otus']['X'].index.values)

        # labels=y
        print(args.run_name)
        model = benchmarker(dataset_dict, y, args, perturbed_mets=None, seed = seed, path=seed_path)

        if args.model.lower() == 'lr':
            model.train_l1_model()
        else:
            model.train_ensemble_models(ensemble_model=args.model)
    # print('END')
    # print('\n\n')
    if isinstance(args.seed, list) and len(args.seed)>1:
        process_benchmark_results(args.out_path + '/' + args.run_name + '/')
        process_benchmark_coefs(args.out_path + '/' + args.run_name + '/')
