import configparser
import pandas as pd
import numpy as np
from utilities.data_utils import read_ctsv
import os
import argparse

def get_week_x_step_ahead(data, targets, week):
    ixs = data.index.values
    pts = [x.split('-')[0] for x in ixs]
    tmpts = [x.split('-')[1] for x in ixs]
    week_one = np.where(np.array(tmpts) == str(week))[0]
    pt_keys = np.array(pts)[week_one]

    rm_ix = []
    targets_out = {}
    event_time = {}
    for pt in np.unique(pts):
        targets_out[pt] = 'Non-recurrer'
        ix_pt = np.where(pt == np.array(pts))[0]
        tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.', '').isnumeric()]
        event_time[pt] = tm_floats[-1]
        if targets[pt] == 'Recurrer':
            ix_pt = np.where(pt == np.array(pts))[0]
            tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.','').isnumeric()]
            if week not in tm_floats:
                continue
            if max(tm_floats) == week:
                rm_ix.append(pt)
                continue
            # tm_floats.sort()
            # tmpt_step_before = tm_floats[-2]
            # if tmpt_step_before == week:
            targets_out[pt] = 'Recurrer'

    pt_keys = np.array(list(set(pt_keys) - set(rm_ix)))
    pt_keys_1 = np.array([pt + '-' + str(week) for pt in pt_keys])
    data_w1 = data.loc[pt_keys_1]
    targs = pd.Series(targets_out, name='Outcome')[pt_keys]
    data_w1.index = [xx.split('-')[0] for xx in data_w1.index.values]
    return data_w1,targs,pd.Series(event_time)[pt_keys]

def get_new_paths(base_path, week):
    metab_new_path = f'{base_path}/week_{week}/metabolomics.csv'
    sample_meta_new_path=f'{base_path}/week_{week}/sample_meta_data.csv'
    metab_meta_new_path=f'{base_path}/week_{week}/metab_meta_data.csv'
    sequences_new_path=f'{base_path}/week_{week}/sequence_data.csv'
    replicates_path = f'{base_path}/week_{week}/replicates.csv'
    return metab_new_path, sample_meta_new_path, metab_meta_new_path, sequences_new_path, replicates_path

def load_cdiff_data(config_file_path, outcome_var='Outcome'):

    config_file = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config_file.read(config_file_path)

    if 'out_path' in config_file['description'] and config_file['description']['out_path'] is not None:
        base_path = config_file['description']['out_path']
    else:
        base_path=os.getcwd()

    week = int(config_file['data']['week'])
    metab_file = config_file['metabolite_data']['data']
    xl = pd.ExcelFile(metab_file)
    cdiff_raw = xl.parse('OrigScale', header=None, index_col=None)
    ixs = np.where(cdiff_raw == 'MASS EXTRACTED')
    ix_row, ix_col = ixs[0].item(), ixs[1].item()
    act_data = cdiff_raw.iloc[ix_row + 2:, ix_col + 1:]
    feature_header = cdiff_raw.iloc[ix_row + 2:, :ix_col + 1]
    pt_header = cdiff_raw.iloc[:ix_row + 1, ix_col + 1:]
    pt_names = list(cdiff_raw.iloc[:ix_row + 1, ix_col])
    feat_names = list(cdiff_raw.iloc[ix_row + 1, :ix_col + 1])
    feat_names[-1] = 'HMDB'

    col_mat_mets = feature_header
    col_mat_mets.columns = feat_names
    col_mat_mets.index = np.arange(col_mat_mets.shape[0])
    #
    col_mat_pts = pt_header.T
    col_mat_pts.columns = pt_names

    targets_dict = pd.Series(col_mat_pts['PATIENT STATUS (BWH)'].values,
                                  index=col_mat_pts['CLIENT SAMPLE ID'].values).to_dict()

    cdiff_dat = pd.DataFrame(np.array(act_data), columns=col_mat_pts['CLIENT SAMPLE ID'].values,
                                  index=col_mat_mets['BIOCHEMICAL'].values).fillna(0).T

    print(f'CDI metabolite data has {cdiff_dat.shape[1]} metabolites and {cdiff_dat.shape[0]} samples')
    targets_by_pt = pd.Series({key.split('-')[0]: value for key, value in targets_dict.items() if
                          key.split('-')[1].isnumeric()}).replace('Cleared','Non-recurrer').replace('Recur','Recurrer')
    # tmpts = [c.split('-')[1] for c in col_mat_pts]
    # col_mat_pts.index = [c.split('-')[0] for c in col_mat_mets]
    # col_mat_pts['Timepoint'] = tmpts
    col_mat_pts=col_mat_pts.set_index('CLIENT SAMPLE ID')
    replicate_ids = [c for c in col_mat_pts.index.values if not c.split('-')[1].replace('.','').isnumeric()]

    replicates = cdiff_dat.loc[replicate_ids]
    non_replicates = cdiff_dat.drop(replicate_ids)
    weekXdata, weekXtargets, eventTimes = get_week_x_step_ahead(non_replicates, targets_by_pt.to_dict(), week=week)

    col_mat_ix_keep = [c for c in col_mat_pts.index.values if c.split('-')[1]==str(week)]
    col_mat_pts = col_mat_pts.loc[col_mat_ix_keep]
    col_mat_pts.index = [c.split('-')[0] for c in col_mat_pts.index.values]
    sample_meta = col_mat_pts.loc[weekXdata.index.values]
    sample_meta['RecurrenceWeek'] = eventTimes
    sample_meta[outcome_var] = weekXtargets

    # drop metabolites that are all 0 (i.e. were not zero in other timepoint)
    nonzero_ixs = weekXdata.sum(0)!=0
    weekXdata = weekXdata.loc[:,nonzero_ixs]
    print(f'Week {week} CDI metabolite data has {weekXdata.shape[1]} metabolites and {weekXdata.shape[0]} samples')

    # load sequence data and get only current week
    seq_file = config_file['sequence_data']['data']
    if seq_file.split('.')[1] == 'xlsx':
        xl = pd.ExcelFile(seq_file)
        seq_data = xl.parse(header=0, index_col=0).T
    else:
        seq_data = read_ctsv(seq_file, index_col=0, header=0).T

    print(f'CDI data sequence has {seq_data.shape[1]} ASVs and {seq_data.shape[0]} samples')

    seq_ixs = [w + '-' + str(week) for w in weekXdata.index.values if w + '-' + str(week) in seq_data.index.values]
    seq_data=seq_data.loc[seq_ixs]
    nonzero_ixs = seq_data.sum(0)!=0
    seq_data = seq_data.loc[:,nonzero_ixs]
    seq_data.index = [s.split('-')[0] for s in seq_data.index.values]
    print(f'Week {week} CDI data sequence has {seq_data.shape[1]} ASVs and {seq_data.shape[0]} samples')

    col_mat_mets = col_mat_mets.set_index('BIOCHEMICAL')
    # cdiff_data_dict = {'sampleMetadata': sample_meta, 'metabMetadata': col_mat_mets,
    #                         'metab_data': weekXdata, 'replicates': replicates,
    #                         'targets': weekXtargets,
    #                    'seq_data': seq_data}

    if base_path[0]!='.':
        base_path='./' + base_path
        out_base_path='./datasets/cdi/' + base_path
    else:
        out_base_path='./datasets/cdi/' + base_path.replace('./','')
    if not os.path.isdir(f'{base_path}/week_{week}/'):
        os.mkdir(f'{base_path}/week_{week}/')

    metab_new_path, sample_meta_new_path, metab_meta_new_path, sequences_new_path, replicates_path = get_new_paths(base_path, week)

    weekXdata.to_csv(metab_new_path)
    sample_meta.to_csv(sample_meta_new_path)
    col_mat_mets.to_csv(metab_meta_new_path)
    seq_data.to_csv(sequences_new_path)
    replicates.to_csv(replicates_path)

    metab_new_path, sample_meta_new_path, metab_meta_new_path, sequences_new_path, replicates_path = get_new_paths(out_base_path, week)

    config_file.set('metabolite_data', 'data', metab_new_path)
    config_file.set('data', 'subject_data', sample_meta_new_path)
    if config_file['metabolite_data']['meta_data']=='':
        config_file.set('metabolite_data', 'meta_data', metab_meta_new_path)
    config_file.set('sequence_data', 'data', sequences_new_path)
    config_file.set('metabolite_data','replicates',replicates_path)
    config_file.set('description', 'out_path', out_base_path.replace('/tmp/','/processed/') +f'/week_{week}/')
    config_file.set('description', 'in_path', out_base_path.replace('/tmp/', '/raw/'))
    config_file.set('data', 'outcome_variable', outcome_var)
    config_file.set('data', 'outcome_positive_value', 'Recurrer')

    cfg_name = args.config_file.split('/')[-1]
    with open(base_path + '/'+cfg_name, 'w') as f:
        config_file.write(f)

    if not os.path.isdir(base_path.replace('/tmp/','/processed/') +f'/week_{week}/'):
        os.mkdir(base_path.replace('/tmp/','/processed/') +f'/week_{week}/')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='raw/cdi_w2.cfg')
    args = parser.parse_args()

    load_cdiff_data(args.config_file)
