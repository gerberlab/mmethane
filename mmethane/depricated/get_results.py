import os
import sys
sys.path.append(os.path.abspath(".."))
# from lightning_trainer import *

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score
import pickle as pkl
import argparse
from utilities.data_utils import *


def by_hand_calc(y_true, y_prob, threshold):

    y_pred = np.where(y_prob >= threshold, 1, 0)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    return (1+tpr - fpr)/2, 2*(precision*recall)/(precision + recall)

def eval_rules(file, path, path_to_dataset, path_to_addn_taxonomy=None,
               addn_taxonomy_col_name=None, path_to_inchis=None, plot_metabolites_in_group=False):
    new_mets=[]
    if 'week_1' in path_to_dataset or 'w1' in path_to_dataset:
        week = 'week_1'
    elif 'week_2' in path_to_dataset or 'w2' in path_to_dataset:
        week = 'week_2'
    else:
        week = ''
    dpath = '/'.join(path_to_dataset.split('/')[:-1])
    if 'seqs' in path_to_dataset:
        path_to_otu_dataset = path_to_dataset
        path_to_dataset=''
        paths=None
    # elif 'cdi' in path_to_dataset or 'week' in path:
    #     paths = paths_dict['cdi']
    #     week = path.split('week_')[-1][0]
    #     path_to_dataset = f"../datasets/cdi/processed/week_{week}/mets.pkl"
    #     path_to_otu_dataset = path_to_dataset.replace('mets.pkl','seqs.pkl')
    elif 'seqs.pkl' in os.listdir(dpath):
        path_to_otu_dataset = dpath + '/seqs.pkl'
        paths=None
    # elif 'franzosa' in path_to_dataset or 'fransoza' in path_to_dataset:
    #     paths = paths_dict['franzosa']
    # elif 'preterm' in path_to_dataset or 'vaginal_microbiome' in path_to_dataset:
    #     paths = paths_dict['preterm']
    else:
        # if 'yachida' in path_to_dataset:
        #     tmp = path_to_dataset.split('_')[:-1]
        # ttmp = '/'.join(path_to_dataset.split('/')[:-1])
        # tmp = ttmp.split()
        if 'xdl' in path_to_dataset or '2_' in path_to_dataset:
            tmp = path_to_dataset.split('_')[:-2]
        else:
            tmp = path_to_dataset.split('_')[:-1]
        if len(tmp)>1:
            tmp = '_'.join(tmp)
        else:
            tmp = tmp[0]
        if '_pp' in path_to_dataset:
            if os.path.isfile('_'.join(tmp.split('_')[:-1]) + '_cts/seqs.pkl'):
                path_to_otu_dataset = '_'.join(tmp.split('_')[:-1]) + '_cts/seqs.pkl'
            else:

                path_to_otu_dataset = '_'.join(tmp.split('_')[:-1]) + '_cts/seqs.pkl'
        else:
            if '_ra/' in path:
                add_on='_ra/'
            else:
                add_on='_cts/'
            if '2_' in path:
                add_on += '2_'
            if os.path.isfile(tmp + add_on+ 'seqs.pkl'):
                path_to_otu_dataset = tmp +add_on+ 'seqs.pkl'
            else:
                path_to_otu_dataset = tmp +add_on+ 'seqs.pkl'
        paths = None
    if paths is not None:
        if path_to_addn_taxonomy is None:
            path_to_addn_taxonomy = paths['addn_taxonomy']
            addn_taxonomy_col_name = paths['col_name']
        if path_to_inchis is None:
            path_to_inchis = paths['path_to_inchis']

    dataset_dict={}
    try:
        dataset_dict['metabs'] = pd.read_pickle(path_to_dataset)
    except:
        dataset_dict['metabs']={}
    try:
        dataset_dict['otus'] = pd.read_pickle(path_to_otu_dataset)
    except:
        dataset_dict['otus']={}
#     dataset_dict = pd.read_pickle(path_to_dataset)
    # with open(path_to_dataset, 'rb') as f:
    #     dataset_dict = pkl.load(f)


    # taxonomy = dataset_dict['metabs']['taxonomy'].loc['subclass']
    # taxonomy =taxonomy[~taxonomy.index.duplicated(keep='first')]
    df_cols = ['subclass']
    if path_to_addn_taxonomy is not None:
        addn_taxa = read_ctsv(path_to_addn_taxonomy, index_col=0)
        if addn_taxonomy_col_name is not None:
            addn_taxa = addn_taxa[addn_taxonomy_col_name]
            df_cols.extend(addn_taxonomy_col_name)
        else:
            df_cols.extend(addn_taxa.columns.values)
    else:
        addn_taxa = None

    if file == 'rules.csv':
        rules = pd.read_csv(path + '/rules.csv', index_col=0, header=0)
        out_path = path
    elif os.path.isdir(path + file) and 'rules.csv' in os.listdir(path + file):
        rules = pd.read_csv(path + file + '/rules.csv', index_col=0, header=0)
        out_path = path + file + '/'
    else:
        return False, None, None, []
    rule_dict = {}
    # num_rules = rules.shape[0]
    num_detectors=0
    all_feats = []
    rules_ls = []
    for i in np.arange(rules.shape[0]):
        rule_type=None
        rule = rules.index.values[i]
        detector = rules.iloc[i, 0]
        mets=[]
        if 'metabs' in rules.columns.values:
            mets = rules['metabs'].iloc[i]
            rule_type = 'metabs'
        elif 'otus' in rules.columns.values:
            mets = rules['otus'].iloc[i]
            rule_type = 'otus'
        else:
            mets = rules['features'].iloc[i]

            if 'ASV' in mets or 'OTU' in mets or path_to_dataset=='':
                rule_type='otus'
            else:
                rule_type='metabs'
        if rule_type=='metabs':
            if mets != "[]" or mets != np.nan:
                try:
                    mets = mets[1:-2]
                    mets = re.split(r"(\', \'|\", \"|\', \"|\", \')", mets)
                except:
                    mets = []
            else:
                mets = []
        else:
            pattern = 'ASV\s\d+\s\D\w+\s\w+\D'
            if isinstance(mets,list):
                mets = mets[0]
            elif len(mets)>20:
                mets = list(set(re.findall(pattern, mets)))


        if 'weights' in rules.columns.values:
            wts = rules['weights'].iloc[i]
            wts = wts[1:-2]
            wts = re.split(r"(\', \'|\", \"|\', \"|\", \')", wts)
            try:
                wts = [float(w) for w in wts[0].split(', ') if w!='']
            except:
                print(wts)
        else:
            wts = None
            # wts = [int(w) for w in wts]
        if rule_type is None or len(mets)==0:
            continue
        elif len(mets) > 0 and mets[0] != '':
            try:
                dist_mat = dataset_dict[rule_type]['distances']
            except:
                continue
                # import pdb; pdb.set_trace()
            # tree = dataset_dict[rule_type]['variable_tree']
            num_detectors+= 1
            rules_ls.append(rule)
            new_mets = []
            for m in mets:
                new = re.sub(r"(\A\'|\A\"|\'\Z|\"\Z|\\)", '', m, count=10)
                new_mets.append(new)
            if all([len(m)<=1 for m in new_mets]):
                new_mets=[''.join(new_mets)]
            if wts==None:
                wts = [np.nan]*len(new_mets)
            
            if rule_type == 'metabs':
                new_mets = [s.replace('*', '') for s in new_mets if re.match("\S*\w+", s)]

            # if 'joined' in case_path:
                mets_w_id = list(set(new_mets).intersection(set(dist_mat.index.values)))
                mets_wo_id = list(set(new_mets) - set(mets_w_id))
                new_mets_simple = new_mets
            else:
                new_mets = [s.split('(')[0] for s in new_mets if re.match("\S*\w+", s)]
                mets_w_id=new_mets
                mets_wo_id=[]

            new_mets = [m.strip() for m in new_mets]
            all_feats.extend(new_mets)
            nm = len(mets_w_id)
            # rd_path = out_path + '/rule_{0}_detector_{1}'.format(rule.split(' ')[1],
            #                                                      detector.split(' ')[1])

            if 'Detect Selector' not in rules.columns.values:
                rules['Detect Selector'] = [np.nan]*rules.shape[0]
            elif any([(isinstance(rules['Detect Selector'].iloc[i], str) and '[' in rules['Detect Selector'].iloc[i]) for i in np.arange(rules.shape[0])]):
                rules['Detect Selector'] = [np.nan] * rules.shape[0]

            if rule_type=='metabs':
                if isinstance(dataset_dict['metabs']['taxonomy'], pd.Series):
                    dataset_dict['metabs']['taxonomy'] = pd.concat([dataset_dict['metabs']['taxonomy'],dataset_dict['metabs']['taxonomy']], axis=1).T
                    try:
                        dataset_dict['metabs']['taxonomy'].index = ['class','subclass']
                    except:
                        dataset_dict['metabs']['taxonomy'] = dataset_dict['metabs']['taxonomy'].T
                        dataset_dict['metabs']['taxonomy'].index = ['class', 'subclass']

                taxa_mets = list(set(dataset_dict['metabs']['taxonomy'].columns.values).intersection(set(new_mets)))
                if len(taxa_mets) == 0:
                    tmp = [n.split(': ')[-1] for n in new_mets]
                    taxa_mets = list(set(dataset_dict['metabs']['taxonomy'].columns.values).intersection(set(tmp)))
                met_df = dataset_dict['metabs']['taxonomy'][taxa_mets].loc[['class','subclass']]
                met_df = met_df[~met_df.index.duplicated(keep='first')]
                for im,met in enumerate(new_mets):
                    if met in met_df.columns.values:
                        try:
                            rule_dict[(rule, detector, met)] = {'Class/Family': met_df[met].loc['class'],
                                                                'Subclass/genus-species': met_df[met].loc['subclass'],
                                                                'Detector Threshold': rules['Detector Threshold'].iloc[i],
                                                                'Detector Radius': rules['Detector Radius'].iloc[i],
                                                                'Rule Log Odds': rules['Rule Log Odds'].iloc[i],
                                                                'Feature Selector': wts[im],
                                                                'Detector Selector': np.round(rules['Detect Selector'].iloc[i],
                                                                                              3),
                                                                }
                        except:
                            rule_dict[(rule, detector, met)] = {'Class/Family': met_df[met].loc['class'],
                                                                'Subclass/genus-species': met_df[met].loc['subclass'],
                                                                'Detector Threshold': rules['Detector Threshold'].iloc[i],
                                                                # 'Detector Radius': rules['Detector Radius'].iloc[i],
                                                                'Rule Log Odds': rules['Rule Log Odds'].iloc[i],
                                                                # 'Feature Selector': wts[im],
                                                                # 'Detector Selector': np.round(rules['Detect Selector'].iloc[i],
                                                                #                               3),
                                                                }

                    else:
                        rule_dict[(rule, detector, met)] = {'Class/Family':'NA',
                                                            'Subclass/genus-species': 'NA',
                                                            'Detector Threshold': rules['Detector Threshold'].iloc[i],
                                                            # 'Detector Radius': rules['Detector Radius'].iloc[i],
                                                            'Rule Log Odds': rules['Rule Log Odds'].iloc[i],
                                                            # 'Feature Selector': wts[im],
                                                            # 'Detector Selector': np.round(rules['Detect Selector'].iloc[i],
                                                            #                               3),
                                                            }
                    if 'Rule Selector' in rules.columns.values:
                        rule_dict[(rule, detector, met)].update(
                            {'Rule Selector': np.round(rules['Rule Selector'].iloc[i], 3),
                             'Rule Log Odds': np.round(rules['Rule Log Odds'].iloc[i], 3)})
                has_id = [1 if m in mets_w_id else 0 for m in met_df.index.values]
                met_df['Has ID'] = has_id
                met_df.to_csv(out_path + '/' + 'categories.csv')
            else:
                if 'taxonomy' in dataset_dict[rule_type].keys():
                    if isinstance(dataset_dict[rule_type]['taxonomy'], pd.Series):
                        dataset_dict[rule_type]['taxonomy'] = pd.concat([dataset_dict[rule_type]['taxonomy'],dataset_dict[rule_type]['taxonomy']], axis=1).T
                        try:
                            dataset_dict[rule_type]['taxonomy'].index = ['Family','Species']
                        except:
                            dataset_dict[rule_type]['taxonomy'] = dataset_dict[rule_type]['taxonomy'].T
                            dataset_dict[rule_type]['taxonomy'].index = ['class', 'subclass']
                    m_map = {m.strip():m.split('(')[0].strip() for m in mets_w_id}
                    met_dict = {m.split('(')[0].strip():m.split('(')[-1].split(' ')[-1].replace(')','').strip() for m in mets_w_id}
                    # fam = dataset_dict[rule_type]['taxonomy'][met].loc['Family']
                    # spec=dataset_dict[rule_type]['taxonomy'][met].loc['Species']
                # else:
                    # fam='NA'
                    # spec='NA'
    #             met_df = pd.Series(met_dict)


                mets_w_id = list(set(mets_w_id))
                mets_w_id = [m.strip() for m in mets_w_id]
                for im,met in enumerate(mets_w_id):
                    try:
                        rule_dict[(rule, detector,met)] = {'Class/Family': dataset_dict[rule_type]['taxonomy'][met].loc['Family'],
                                                                  'Subclass/genus-species': dataset_dict[rule_type]['taxonomy'][met].loc['Species'],
                                                           'Detector Threshold': rules['Detector Threshold'].iloc[i],
                                                           'Detector Radius': rules['Detector Radius'].iloc[i],
                                                           'Rule Log Odds': rules['Rule Log Odds'].iloc[i],
                                                                  'Feature Selector': wts[im],
                                                                  'Detector Selector': np.round(rules['Detect Selector'].iloc[i],3)}
                        if 'Rule Selector' in rules.columns.values:
                            rule_dict[(rule, detector, met)].update({'Rule Selector': np.round(rules['Rule Selector'].iloc[i],3),
                                                                  'Rule Log Odds': np.round(rules['Rule Log Odds'].iloc[i],3)})
                    except:
                        rule_dict[(rule, detector,met)] = {'Class/Family': 'NA',
                                                                  'Subclass/genus-species': 'NA',
                                                           'Detector Threshold': rules['Detector Threshold'].iloc[i],
                                                           # 'Detector Radius': rules['Detector Radius'].iloc[i],
                                                           'Rule Log Odds': rules['Rule Log Odds'].iloc[i],
                                                           }
                                                                  # 'Feature Selector': wts[im],
                                                                  # 'Detector Selector': np.round(rules['Detect Selector'].iloc[i],3)}
                        if 'Rule Selector' in rules.columns.values:
                            rule_dict[(rule, detector, met)].update({'Rule Selector': np.round(rules['Rule Selector'].iloc[i],3),
                                                                  'Rule Log Odds': np.round(rules['Rule Log Odds'].iloc[i],3)})
                    # except:
                        # rule_dict[(rule, detector, m_map[met])] = {
                        #     'Class/Family': m_map[met],
                        #     'Subclass/genus-species': m_map[met],
                        #     'Feature Selector': wts[im],
                        #     'Detector Selector': np.round(rules['Detect Selector'].iloc[i], 3),
                        # }
                        # if 'Rule Selector' in rules.columns.values:
                        #     rule_dict[(rule, detector, m_map[met])].update({'Rule Selector': np.round(rules['Rule Selector'].iloc[i],3),
                        #                                           'Rule Log Odds': np.round(rules['Rule Log Odds'].iloc[i],3)})



            # met_df['Has ID'] = has_id
            # met_df.to_csv(rd_path + '/' + 'categories.csv')

    return pd.DataFrame(rule_dict).T, len(set(rules_ls)), num_detectors,all_feats
    # empty_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([],
    #                                             names=('Rule','Detector','Metabolite')), columns = df_cols)
    # df_ls = []
    # for k, v in rule_dict.items():
    #     df_ls.append(
    #         pd.DataFrame([v.values], index=pd.MultiIndex.from_tuples([k], names=['Rule','Detector','Metabolite']), columns=v.index.values))
    # if len(df_ls)>0:
    #     pd.concat(df_ls).to_csv(out_path + 'rules_cats.csv')
    #     return pd.concat(df_ls), num_rules
    # else:
    #     return empty_df, num_rules

def get_results(case_path, case_type, change_vars, path_to_dataset, args):
    args_comp = {}
    compare_dict = {}
    full_df = []

    print(f'Getting results for case {case_path}')
    if '.DS_' in case_path or case_type not in case_path:
        print(f'No results for case {case_path}')
        return [], ''
    seed_res = []
    seeds = []

    eval_best_per_seed={}
    eval_last_per_seed = {}
    metabs={}
    # last_metabs=[]
    seqs={}
    detect_select = {}
    log_odds = {}
    rule_select={}
    feat_select={}
    # last_seqs=[]
    for seed in os.listdir(case_path):
        if '.DS' in seed or '.' in seed:
            continue
        # print(seed)
        seed_path = case_path + '/' + seed + '/'

        try:
            with open(seed_path + '/commandline_args_eval.txt', 'r') as f:
                args_dict = json.load(f)
        except:
            continue
        param_ls = []
        for val in change_vars:
            if 'args_dict' in args_dict.keys():
                ad = args_dict['args_dict']
            else:
                ad = args_dict
            if val in ad.keys():
                tmp = ad[val]
                if isinstance(tmp, list) and len(tmp)==1:
                    tmp = tmp[0]
                elif isinstance(tmp, list) and len(tmp)>1:
                    tmp = [str(t) for t in tmp]
                    tmp = '_'.join(tmp)
            else:
                tmp = 'NA'
            param_ls.append(tmp)
        if case_path not in args_comp.keys():
            args_copy = copy.deepcopy(args_dict)
            args_comp[case_path] = args_copy

        all_preds = {}


        if 'running_loss_dict.pkl' in os.listdir(f'{seed_path}/EVAL/'):
            try:
                with open(f'{seed_path}/EVAL/running_loss_dict.pkl','rb') as f:
                    loss_dict = pkl.load(f)
            except:
                continue
            try:
                ending_ce_loss = loss_dict['train_loss'][-1].detach().item()
                ending_total_loss = 0
                for k,v in loss_dict.items():
                    val = np.array([vi.detach().numpy().item() for vi in v])
                    if all(abs(val[1:]-val[:-1])<0.01) and abs(val[-1]-val[0])<0.01*np.max(np.abs(val)):
                        print(f"{k} doesn't change, not included in loss calcuation")
                        continue
                    else:
                        ending_total_loss += val[-1]
            except:
                ending_ce_loss = 99
                ending_total_loss = 99

        else:
            ending_ce_loss = 99
            ending_total_loss = 99

        num_rules = []
        num_detectors = []
        fold_path = seed_path + 'EVAL/'

        if not os.path.isdir(fold_path):
            continue
        fname='last'
        if 'rules.csv' in os.listdir(fold_path):
        # for file in os.listdir(seed_path + '/' + fold + '/'):
        # #     if fold == 'EVAL':
        #         if os.path.isdir(seed_path + '/' + fold + '/' + file) and 'rules.csv' in os.listdir(seed_path + '/' + fold + '/' + file):
            if 'REPLACE' in path_to_dataset:
                path_to_dataset = path_to_dataset.replace('REPLACE',seed.split('_')[-1][-1])
            # try:
            rule_df, r_tmp, n_detectors, feats = eval_rules('rules.csv', fold_path, path_to_dataset,
                                plot_metabolites_in_group=False)
            # except:
            #     import pdb; pdb.set_trace()
                    # if file != 'last':
                    #     fname = 'best'
                    # else:
                    #     fname = 'last'


            if fname not in metabs.keys():
                metabs[fname] = []
            metabs[fname].append(feats)

            ats = ['Feature Selector','Rule Selector','Detector Selector','Rule Log Odds']
            for feat in feats:
                if feat not in rule_select.keys():
                    for at in ats:
                        if at in rule_df.columns.values:
                            feat_select[feat] = [rule_df[at].loc[(slice(None), slice(None),feat)].values.mean()]

                else:
                    for at in ats:
                        if at in rule_df.columns.values:
                            feat_select[feat].append(rule_df[at].loc[(slice(None), slice(None),feat)].values.mean())

            if r_tmp is None:
                r_tmp=0
            if n_detectors is None:
                n_detectors=0
            num_rules.append(r_tmp)
            num_detectors.append(n_detectors)
            if isinstance(rule_df, pd.DataFrame) and not rule_df.empty:
                eval_last_per_seed[seed] = rule_df
                    # else:
                    #     num_rules[fname] = {seed: 0}
            # except:
            #     tmp = pd.read_csv(os.path.join(fold_path, 'rules.csv'), header=0)
            #     num_detectors.append(tmp.shape[0])
            #     continue

        res_fname = [f for f in os.listdir(seed_path) if 'pred_results' in f]
        if len(num_rules)>0:
            num_rules = np.mean(num_rules)
        else:
            num_rules = 0
        if len(num_detectors)>0:
            num_detectors = np.mean(num_detectors)
        else:
            num_detectors = 0
        resbool=False
        if len(res_fname) != 0:
            try:
                for res in res_fname:
                    if '_pred' not in res:
                        res_name = ''
                    else:
                        res_name = res.split('_pred')[0]

                    total_results = pd.read_csv(seed_path + '/' + res, index_col=0)
                    all_preds[res_name] = total_results
                resbool=True
            except:
                pass

        if resbool==False:
            res_vec = []
            for res_name in ['last']:
                for fold in os.listdir(seed_path):
                    if 'fold' not in fold:
                        continue
                    print(fold)
                    fold_path = seed_path + '/' + fold
                    if res_name == 'best':
                        try:
                            res = [f for f in os.listdir(fold_path) if 'epoch=' in f][0]
                        except:
                            continue
                    else:
                        res = res_name
                    try:
                        res_fname = [f for f in os.listdir(fold_path + '/' + res) if 'pred_results' in f]
                        tmp = pd.read_csv(fold_path + '/' + res + '/' + res_fname[0], index_col=0)
                    except:
                        continue
                    res_vec.append(tmp)
                try:
                    total_results = pd.concat(res_vec, axis=0)
                    all_preds[res_name] = total_results
                    total_results.to_csv(seed_path + '/' + res_name + '_pred.csv')
                except:
                    continue

        ii = 0
        for res_name in all_preds.keys():
            index = pd.MultiIndex.from_tuples([tuple([seed] + param_ls)], names=['seed'] + change_vars )
            results = all_preds[res_name]
            if len(np.unique(results['true'].values)) == 1:
                continue
            else:
                if ii == 0:
                    seeds.append(seed)
            tn, fp, fn, tp = confusion_matrix(results['true'], (results['preds'] > 0.5).astype(int)).ravel()

            cv_f1 = f1_score(results['true'], (results['preds'] > 0.5).astype(int))
            cv_f1_weighted = f1_score(results['true'], (results['preds'] > 0.5).astype(int), average='weighted')
            cv_auc = roc_auc_score(results['true'], results['preds'])
            cv_auc_weighted = roc_auc_score(results['true'], results['preds'], average='weighted')
            auc_by_hand, f1_by_hand = by_hand_calc(results['true'], results['preds'], 0.5)
            auc_binary = roc_auc_score(results['true'], (results['preds'] > 0.5).astype(int), average='weighted')

            case, ctrls = np.where(results['true'].values==1)[0], np.where(results['true'].values==0)[0]
            acc_case = accuracy_score(results['true'].values[case], results['preds'].values[case]>0.5)
            acc_ctrl = accuracy_score(results['true'].values[ctrls], results['preds'].values[ctrls]>0.5)

            f1_by_cutoff=[]
            cutoffs = np.arange(0.01,1,0.01)
            for cutoff in cutoffs:
                f1_by_cutoff.append(f1_score(results['true'], (results['preds'] > cutoff).astype(int), average='weighted'))

            f1_by_cutoff = np.array(f1_by_cutoff)
            best_cutoff = cutoffs[np.argmax(f1_by_cutoff)]
            best_f1 = f1_by_cutoff[np.argmax(f1_by_cutoff)]

            all_results = pd.DataFrame({'F1': np.round(cv_f1,3),
                                        'F1_weighted':np.round(cv_f1_weighted,3),
                'AUC':np.round(cv_auc,3),
                                        'AUC_weighted': np.round(cv_auc_weighted,3),
                                        'CE Loss':ending_ce_loss,
                                        'Total Loss':ending_total_loss,
                                        'Case Acc': np.round(acc_case,3), 'Ctrl Acc':np.round(acc_ctrl,3) ,
                                        'AUC binary': np.round(auc_binary, 3),

                                        # 'Best cutoff':best_cutoff,
                                        '# Rules':num_rules,
                                        '# Detectors':num_detectors,
                                        'True -': tn, 'False +': fp, 'False -': fn, 'True +': tp},
                                       index=index)
            print(all_results)
            # all_results.index = index
            seed_res.append(all_results)
            ii += 1

        # seeds.append(seed)

    # get metabs/seqs common across rules
    metabs_in_rules={'best':{},'last':{}}
    # seqs_in_rules={'best':{},'last':{}}
    for bl in metabs.keys():
        met_tmp=sum(metabs[bl],[])
        if len(met_tmp)>0:
        #     if isinstance(met_tmp[0], list):
        #         print(met_tmp)
        #         met_tmp = met_tmp[0]
        # metabs_in_rules[bl]['Total #'] = len(set(met_tmp))
            met_to_perc = {m:np.sum([m in m_ls for m_ls in metabs[bl]])/len(metabs[bl]) for m in list(set(met_tmp))}
            # met_to_perc['Total #'] = len(set(met_tmp))
            metabs_in_rules[bl] = met_to_perc
    df_rules = pd.DataFrame(metabs_in_rules)
    if 'last' in df_rules.columns.values:
        try:
            df_rules = df_rules.sort_values(by='last', ascending=False).drop('best', axis=1).rename(
                columns={'last': '% present in'})
        except:
            df_rules = df_rules.sort_values(by='last', ascending=False).rename(
                columns={'last': '% present in'})
    else:
        try:
            df_rules = df_rules.sort_values(by='best', ascending=False).rename(
                columns={'best': '% present in'})
        except:
            df_rules = df_rules
    r_ls=[]
    d_ls=[]
    lo_ls = []
    f_ls = []
    for mf in df_rules.index.values:
        if mf=='Total #':
            continue
        if feat_select.__len__()!=0:
            f_ls.append(f'{np.round(np.mean(feat_select[mf]),3)}+-{np.round(np.std(feat_select[mf]),3)}')
        else:
            f_ls.append(f'NA')
        if rule_select.__len__() != 0:
            r_ls.append(f'{np.round(np.mean(rule_select[mf]),3)}+-{np.round(np.std(rule_select[mf]),3)}')
        else:
            r_ls.append(f'NA')
        if detect_select.__len__() != 0:
            d_ls.append(f'{np.round(np.mean(detect_select[mf]),3)}+-{np.round(np.std(detect_select[mf]),3)}')
        else:
            d_ls.append('NA')
        if log_odds.__len__() !=0:
            lo_ls.append(f'{np.round(np.mean(log_odds[mf]),3)}+-{np.round(np.std(log_odds[mf]),3)}')
        else:
            lo_ls.append('NA')

    df_rules['Feature Selector'] = f_ls
    df_rules['Detector Selector'] = d_ls
    df_rules[f'Rule Selector'] = r_ls
    df_rules['Log Odds'] = lo_ls
    df_rules.to_csv(case_path + '/feats_evaluation_over_seeds.csv')
    # for bl in seqs.keys():
    #     seq_tmp=sum(seqs[bl],[])
    #     if len(seq_tmp)>0:
    #
    #         seq_to_perc = {m:np.sum([m in m_ls for m_ls in seqs[bl]])/len(set(seq_tmp)) for m in list(set(seq_tmp))}
    #         seq_to_perc['Total #'] = len(set(seq_tmp))
    #         seqs_in_rules[bl] = seq_to_perc
    #     # for perc in [1,0.9,0.8,0.7,0.6,0.5]:
    #
    #
    # pd.DataFrame(seqs_in_rules).to_csv(case_path+ '/seqs_evaluation_over_seeds.csv')
    # metabs_across_rules={'Total # of metabolites': }

    # try:
    #     eval_best_df = pd.concat(eval_best_per_seed.values(), axis=0, keys=eval_best_per_seed.keys())
    #     eval_best_df.to_csv(case_path + '/eval_best_per_seed.csv')
    # except:
    #     pass

    try:
        eval_last_df = pd.concat(eval_last_per_seed.values(), axis=0, keys=eval_last_per_seed.keys())
        # for i, ix in enumerate(eval_last_df.index.values):
        #     pattern = 'ASV\s\d+\s\D\w+\s\w+\D'
        #     try:
        #         if 'ASV' in ix[-1]:
        #             try:
        #                 asv = ix[-1].strip()
        #                 taxa = taxonomy[asv]
        #             except:
        #                 asv = re.findall(pattern, ix[-1])
        #                 if len(asv) > 0:
        #                     asv = asv[0]
        #
        #             if eval_last_df['Class/Family'].iloc[i].strip() == asv:
        #                 eval_last_df['Class/Family'].iloc[i] = taxonomy[asv].loc['Family']
        #             if eval_last_df['Subclass/genus-species'].iloc[i].strip() == asv:
        #                 eval_last_df['Subclass/genus-species'].iloc[i] = taxonomy[asv].loc['Species']
    
        # eval_last_df = pd.concat(eval_last_per_seed.values(), axis=0, keys=eval_last_per_seed.keys())
        eval_last_df.to_csv(case_path + '/eval_last_per_seed.csv')
    except:
        pass
    if len(seed_res)>0:
        df = pd.concat(seed_res)
        # df.index = [dd[0] for dd in df.index.values]
        # df = df.sort_index()
        # try:
        #     df_best = df.loc['best']
        # except:
        #     df_best = df

        # df_best.loc['Mean'] = df_best.mean(axis=0)
        # df_best.loc['StDev'] = df_best.std(axis=0)
        # df_best.loc['Median'] = df_best.median(axis=0)
        # try:
        #     df_best.loc['25% Quantile'] = df_best.quantile(0.25, axis=0)
        #     df_best.loc['75% Quantile'] = df_best.quantile(0.75, axis=0)
        # except:
        #     df_best.loc['25% Quantile'] = df_best.quantile(0.25)
        #     df_best.loc['75% Quantile'] = df_best.quantile(0.75)
        # df_best.to_csv(case_path + '/' + 'results_best.csv')

        try:
            df_last = df.loc['last']
        except:
            df_last = df
        df_last.loc['Mean'] = df_last.mean(axis=0)
        df_last.loc['StDev'] = df_last.std(axis=0)
        df_last.loc['Median'] = df_last.median(axis=0)
        try:
            df_last.loc['25% Quantile'] = df_last.quantile(0.25,axis=0)
            df_last.loc['75% Quantile'] = df_last.quantile(0.75, axis=0)
        except:
            df_last.loc['25% Quantile'] = df_last.quantile(0.25)
            df_last.loc['75% Quantile'] = df_last.quantile(0.75)
        results = df_last.iloc[:-5,:]
        lowest_loss = results.index.values[results['Total Loss'] == results['Total Loss'].min()][0][0]
        seed = lowest_loss.split("'")[0]
        res_paths = [os.path.join(case_path, seed)]
        res_lists = [os.listdir(res_paths[0])]
        with open(os.path.join(case_path, f'lowest_loss_{seed}.txt'), 'w') as f:
            f.writelines(f'{seed} has lowest loss: {results["Total Loss"].min()}')
        df_last.to_csv(case_path + '/' + 'results_last.csv')

        for res_name in all_preds.keys():
            # df_sm = df.loc[res_name]
            df_sm = df
            if len(df_sm.shape)>1:
                df_sm_avg = df_sm.median(axis=0)
                df_sm_p1 = df_sm.quantile(0.25, axis=0)
                df_sm_p2 = df_sm.quantile(0.75, axis=0)
            else:
                df_sm_avg = df_sm
                df_sm_p1 = pd.Series([0]*df_sm.shape[0], index=df_sm.index.values)
                df_sm_p2 = pd.Series([0]*df_sm.shape[0], index=df_sm.index.values)
            if res_name + 'F1' not in compare_dict.keys():
                compare_dict[res_name + 'F1'] = {}
                compare_dict[res_name + 'F1_weighted'] = {}
                compare_dict[res_name + 'AUC'] = {}
            compare_dict[res_name+ 'F1'] = str(np.round(df_sm_avg['F1'], 4))
            compare_dict[res_name+ 'F1_weighted'] = str(np.round(df_sm_avg['F1_weighted'], 4))
            compare_dict[res_name+ 'AUC'] = str(np.round(df_sm_avg['AUC'], 4))

            if len(change_vars) == 0:
                change_vars = [0]
                param_ls = [0]
            compare_df = pd.DataFrame(
                {'F1': '{0} [{1},{2}]'.format(np.round(df_sm_avg['F1'], 4), np.round(df_sm_p1['F1'], 4), np.round(df_sm_p2['F1'], 4)),
                 'F1_weighted': '{0} [{1},{2}]'.format(np.round(df_sm_avg['F1_weighted'], 4), np.round(df_sm_p1['F1_weighted'], 4),
                                              np.round(df_sm_p2['F1_weighted'], 4)),
                 'AUC': '{0} [{1},{2}]'.format(np.round(df_sm_avg['AUC'], 4), np.round(df_sm_p1['AUC'], 4), np.round(df_sm_p2['AUC'], 4)),
                 'CE Loss': '{0} [{1},{2}]'.format(np.round(df_sm_avg['CE Loss'], 4), np.round(df_sm_p1['CE Loss'], 4),
                                               np.round(df_sm_p2['CE Loss'], 4)),
                 'Total Loss': '{0} [{1},{2}]'.format(np.round(df_sm_avg['Total Loss'], 4), np.round(df_sm_p1['Total Loss'], 4),
                                                   np.round(df_sm_p2['Total Loss'], 4)),
                    'Case Acc': '{0} [{1},{2}]'.format(np.round(df_sm_avg['Case Acc'], 4), np.round(df_sm_p1['Case Acc'], 4), np.round(df_sm_p2['Case Acc'], 4)),
                 'Ctrl Acc': '{0} [{1},{2}]'.format(np.round(df_sm_avg['Ctrl Acc'], 4), np.round(df_sm_p1['Ctrl Acc'], 4), np.round(df_sm_p2['Ctrl Acc'], 4)),
                 # 'Best cutoff':'{0} [{1},{2}]'.format(np.round(df_sm_avg['Best cutoff'], 4), np.round(df_sm_p1['Best cutoff'], 4), np.round(df_sm_p2['Best cutoff'], 4)),
                 '# Detectors':np.round(df_sm_avg['# Detectors'],4),
                 '# Rules': np.round(df_sm_avg['# Rules'], 4),
                 '# Seeds': len(seeds)},
                index=pd.MultiIndex.from_tuples([tuple(param_ls)], names=change_vars))
            full_df.append(compare_df)

        rename_str = '_'.join([k+'='+v.replace('.','d') for k,v in compare_dict.items()])
        return full_df, rename_str
    else:
        return pd.DataFrame([]), ''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default='/Users/jendawk/logs/', type=str)
    parser.add_argument('--changed_vars', default=[], type=str, nargs='+')
    parser.add_argument('--command_list', default=[], type=str, nargs='+')
    parser.add_argument('--case_dirs', default=[]
                        , type=str, nargs='+')
    parser.add_argument('--super_super_folder', default='', type=str)
    parser.add_argument('--super_folder', default='/Users/jendawk/logs/semisyn-sep17/full_nn_oct21/', type=str)
    parser.add_argument('--path_to_dataset', default='', type=str)
    parser.add_argument('--plot_metabolite_structures', default=0, type=int)
    parser.add_argument('--case', default='', type=str)
    parser.add_argument('--semisyn', default=1, type=int)
    parser.add_argument('--super_case', default='', type=str)
    parser.add_argument('--super_ignore', default=['F1'], type=str, nargs='+')
    parser.add_argument('-r','--redo_renamed', default=0, type=int)
    args = parser.parse_args()

    # res, res_df = get_results('/Users/jendawk/logs/mditre-logs/both_and_12_SEMISYN/',
    #                           'SEMISYN',[],'/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/SEMISYN/processed/both_and_12_9/mets.pkl', args)

    print(args.case_dirs)
    super_final_ls = []
    if len(args.super_super_folder)!=0 and len(args.command_list)==0 and \
            ((args.case_dirs is None or args.case_dirs=='') or len(args.case_dirs)==0) \
            and len(args.super_folder)==0:
        args.case_dirs = [[args.super_super_folder + '/' + subpath + '/' + p for p in os.listdir(args.super_super_folder + '/' + subpath) if
                               os.path.isdir(args.super_super_folder + '/' + subpath + '/' + p)] for subpath in
             os.listdir(args.super_super_folder) if os.path.isdir(args.super_super_folder + '/' + subpath)]
        print('')
        for case_super_paths in args.case_dirs:

            if isinstance(args.super_ignore, list):
                tmp = case_super_paths
                for si in args.super_ignore:
                    tmp = [a for a in tmp if args.super_case in a and si not in a]
            else:
                tmp = [a for a in case_super_paths if args.super_case in a and args.super_ignore not in a]
            if len(tmp)==0:
                continue
            case_paths = []
            for c in tmp:
                if c[-1]=='/':
                    case_paths.append(c[:-1])
                else:
                    case_paths.append(c)

            args.logs_dir = '/'.join(case_paths[0].split('/')[:-1]) + '/'
            case_split=case_paths[0].split('/')[-1].split('mets')
            if len(case_split)>1:
                case_add_on = case_split[0] + 'mets'
            else:
                case_add_on='mets'
            paths_to_dataset = [
                f'../datasets/{c.split("/")[-2].split("_")[0].upper()}/processed/{c.split("/")[-2]}/{case_add_on}.pkl' for c in
                case_paths]
            fin_results = []
            for c in np.arange(len(case_paths)):
                if (('best=' in case_paths[c]) or ('last=' in case_paths[c])) and args.redo_renamed != 1:
                    continue
                res, rename_str = get_results(case_paths[c], args.case, args.changed_vars, paths_to_dataset[c], args)
                if res is None:
                    continue
                fin_results.extend(res)
                if rename_str == '' or 'last' in case_paths[c] or 'best' in case_paths[c]:
                    continue
                else:
                    if not os.path.isdir(case_paths[c] + '_' + rename_str):
                        os.rename(case_paths[c], case_paths[c] + '_' + rename_str)

            try:
                final = pd.concat(fin_results, axis=0)
            except:
                continue

            i = 1
            tmp=0
            while os.path.isfile(f'{args.logs_dir}/results_{args.super_case}_{tmp}.csv'):
                tmp = args.case + '_' + str(i)
                i += 1
            final.to_csv(f'{args.logs_dir}/results_{args.super_case}_{tmp}.csv')
        
            super_final = final.reset_index()
            super_final.index = [case_super_paths[0].replace('//','/').split(args.super_super_folder)[-1].split('/')[0]]*final.shape[0]
            super_final_ls.append(super_final)
        
        sf = pd.concat(super_final_ls)
        sf.to_csv(os.path.join(args.super_super_folder,f'{args.super_case}.csv'))



    else:
        if len(args.command_list)>0:
            paths_to_dataset = [c.split('--data_met ')[-1].split(' ')[0] for c in args.command_list]
            case_paths = [args.log_dir + '/' + c.split('--data_name ')[-1].split(' ')[0] for c in args.command_list]
        elif args.case_dirs is not None and len(args.case_dirs)>0:
            case_paths = []
            for c in args.case_dirs:
                if c[-1]=='/':
                    case_paths.append(c[:-1])
                else:
                    case_paths.append(c)

            args.case_dirs=case_paths
            try:
                paths_to_dataset = [
                    f'../datasets/{c.split("/")[-2].split("_")[0].upper()}/processed/{c.split("/")[-2]}/mets.pkl' for c in
                    case_paths]
            except:
                paths_to_datasets=[args.path_to_dataset]
        else:
            if args.super_folder[-1] == '/':
                args.super_folder = args.super_folder[:-1]
            if isinstance(args.super_ignore, list):
                tmp = [args.super_folder + '/' + d for d in os.listdir(args.super_folder) if '.' not in d]
                for si in args.super_ignore:
                    tmp = [a for a in tmp if args.case in a and si not in a]
            else:
                tmp = [args.super_folder + '/' + d for d in os.listdir(args.super_folder) if args.case in d and '.' not in d and args.super_ignore not in d]
            case_paths = tmp
            if 'cdi' in args.super_folder:
                paths_to_dataset = [args.path_to_dataset] * len(case_paths)
            else:
                if 'SEMISYN' in args.super_folder or 'semisyn' in args.super_folder or args.semisyn:
                    if 'SEMISYN' not in case_paths[0].split('/')[-1]:
                        try:
                            cpath = ['_'.join([c.split('__')[0],c.split('__')[1].split('_')[0]]) for c in case_paths]
                        except:
                            cpath = ['_'.join([c.split('_Adam_')[0], c.split('_Adam_')[-1].split('_')[0]]) for c in case_paths]
                        # cpath = [c.split('__')[0] for c in case_paths]
                        paths_to_dataset = [
                            f'../datasets/SEMISYN/processed/{c.split("/")[-1].replace("SEMISYN", "REPLACE")}/mets.pkl'
                            for c in cpath]

                    else:
                        cpath = case_paths
                        paths_to_dataset = [f'../datasets/{args.super_folder.split("/")[-1].split("_")[0].split("-")[0].upper()}/processed/{c.split("/")[-1].replace("SEMISYN","REPLACE")}/mets.pkl' for c in cpath]

                else:
                    paths_to_dataset = [f'../datasets/{args.super_folder.split("/")[-1].split("_")[0].split("-")[0].upper()}/processed/{args.super_folder.split("/")[-1]}/mets.pkl']*len(case_paths)
            args.logs_dir = args.super_folder
            args.case = args.super_case


        if len(case_paths) > 1:
            fin_results = []
            for c in np.arange(len(case_paths)):
                if (('best=' in case_paths[c]) or ('last=' in case_paths[c])) and args.redo_renamed!=1:
                    continue
                res, rename_str = get_results(case_paths[c], args.case, args.changed_vars, paths_to_dataset[c], args)
                if res is None:
                    continue
                fin_results.extend(res)
                if rename_str=='' or 'last' in case_paths[c] or 'best' in case_paths[c]:
                    continue
                else:
                    if not os.path.isdir(case_paths[c] + '_' + rename_str):
                        os.rename(case_paths[c], case_paths[c] + '_' + rename_str)

            final = pd.concat(fin_results, axis=0)
            if args.case=='':
                args.case='res'
            i=1
            while os.path.isfile(f'{args.logs_dir}/{args.case}.csv'):
                args.case = args.case + '_' + str(i)
                i += 1
            final.to_csv(f'{args.logs_dir}/{args.case}.csv')
        else:
            res, rename_str = get_results(case_paths[0], args.case, args.changed_vars, paths_to_dataset[0], args)
            i=1
            while os.path.isfile(f'{args.logs_dir}/{args.case}.csv'):
                args.case = args.case + '_' + str(i)
                i += 1
            if isinstance(res, list):
                pd.concat(res, axis=0).to_csv(case_paths[0] + f'/{args.case}.csv')
            if not os.path.isdir(case_paths[0] + '_' + rename_str):
                os.rename(case_paths[0], case_paths[0] + '_' + rename_str)