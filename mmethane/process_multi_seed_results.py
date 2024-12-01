import os
import sys

sys.path.append(os.path.abspath(".."))
# from lightning_trainer import *

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score
import pickle as pkl
import argparse
from utilities.data_utils import *

def get_results(case_path):

    print(f'Getting results for case {case_path}')
    if not os.path.isdir(case_path):
        warnings.warn(f'{case_path} does not exist. Cannot compute results')
        return
    seed_res = []
    for seed in os.listdir(case_path):
        if '.DS' in seed or '.' in seed:
            continue
        seed_path = case_path + '/' + seed + '/'

        fold_path = seed_path + 'EVAL/'
        if not os.path.isdir(fold_path):
            warnings.warn(
                f"Path {seed_path}/EVAL/ not found. Cannot compute seed with lowest loss on evaluation fold.")
        if 'running_loss_dict.pkl' in os.listdir(f'{seed_path}/EVAL/'):
            with open(f'{seed_path}/EVAL/running_loss_dict.pkl', 'rb') as f:
                loss_dict = pkl.load(f)
            ending_ce_loss = loss_dict['train_loss'][-1].detach().item()
            ending_total_loss = 0
            for k, v in loss_dict.items():
                val = np.array([vi.detach().numpy().item() for vi in v])
                if all(abs(val[1:] - val[:-1]) < 0.01) and abs(val[-1] - val[0]) < 0.01 * np.max(np.abs(val)):
                    continue
                else:
                    ending_total_loss += val[-1]
        else:
            warnings.warn(f"File 'running_loss_dict.pkl' not found in {seed_path}/EVAL/. Cannot compute seed with lowest loss.")
            ending_ce_loss = np.nan
            ending_total_loss = np.nan
        res_fname = [f for f in os.listdir(seed_path) if 'pred_results' in f]
        results = pd.read_csv(seed_path + '/' + res_fname[0], index_col=0)

        cv_f1 = f1_score(results['true'], (results['preds'] > 0.5).astype(int))
        cv_f1_weighted = f1_score(results['true'], (results['preds'] > 0.5).astype(int), average='weighted')
        cv_auc = roc_auc_score(results['true'], results['preds'])

        all_results = pd.DataFrame({'F1': np.round(cv_f1, 3),
                                    'F1_weighted': np.round(cv_f1_weighted, 3),
                                    'AUC': np.round(cv_auc, 3),
                                    'Cross-entropy Loss': ending_ce_loss,
                                    'Total Loss': ending_total_loss}, index=[seed])
        seed_res.append(all_results)

    assert(len(seed_res)>1)
    df_last = pd.concat(seed_res)
    df_last.loc['Mean'] = df_last.mean(axis=0)
    df_last.loc['StDev'] = df_last.std(axis=0)
    df_last.loc['Median'] = df_last.median(axis=0)
    df_last.loc['25% Quantile'] = df_last.quantile(0.25, axis=0)
    df_last.loc['75% Quantile'] = df_last.quantile(0.75, axis=0)
    results = df_last.iloc[:-5, :]
    lowest_loss_seed = results.index.values[results['Total Loss'] == results['Total Loss'].min()][0].split('_')[-1]
    with open(os.path.join(case_path, f'lowest_loss_{lowest_loss_seed}.txt'), 'w') as f:
        f.writelines(f'{lowest_loss_seed} has lowest loss: {results["Total Loss"].min()}')
    print(f'Seed {lowest_loss_seed} has lowest loss')
    df_last.to_csv(case_path + '/' + 'multiseed_results.csv')
    print(f'Multi-seed results written to {case_path}/multiseed_results.csv')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/Users/jendawk/github-repos/mmethane/mmethane/logs/run_franzosa_test_0/', type=str)
    args = parser.parse_args()

    get_results(args.path)