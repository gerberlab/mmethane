import os
import pandas as pd
import re

results_path = '/Users/jendawk/logs/mditre-logs/'
for root, dirs, files in os.walk(results_path):
    if 'week_1' in root or 'he_' in root or 'ibmdb' in root or 'preterm' in root:
        continue
    if 'eval_last_per_seed.csv' in files and 'eval_annotated.csv' not in files:
        print('/'.join(root.split('/')[-2:]))
        dataset_name = '_'.join(root.split(results_path)[-1].split('/')[0].split('_')[:-1])
        dataset_path = f'/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/{dataset_name.split("_")[0].upper()}/processed/{dataset_name}_cts/seqs_xdl.pkl'
        dataset = pd.read_pickle(dataset_path)
        taxonomy = dataset['taxonomy']

        eval = pd.read_csv(os.path.join(root, 'eval_last_per_seed.csv'), index_col=[0, 1, 2, 3])
        # Replace feat with
        for i, ix in enumerate(eval.index.values):
            pattern = 'ASV\s\d+\s\D\w+\s\w+\D'
            try:
                if 'ASV' in ix[-1]:
                    try:
                        asv = ix[-1].strip()
                        taxa = taxonomy[asv]
                    except:
                        asv = re.findall(pattern, ix[-1])
                        if len(asv) > 0:
                            asv = asv[0]

                    if eval['Class/Family'].iloc[i].strip() == asv:
                        eval['Class/Family'].iloc[i] = taxonomy[asv].loc['Family']
                    if eval['Subclass/genus-species'].iloc[i].strip() == asv:
                        eval['Subclass/genus-species'].iloc[i] = taxonomy[asv].loc['Species']
            except:
                continue
        # print(eval.head())
        eval.to_csv(os.path.join(root, 'eval_annotated.csv'))