import pickle, warnings, json, itertools, re, copy
try:
    from rdkit import Chem
except:
    pass
try:
    import tmap as tm
    from map4.map4 import MAP4Calculator

    dim = 1024
    MAP4 = MAP4Calculator(dimensions=dim)
    ENC = tm.Minhash(dim)
except:
    # print('debug')
    pass

try:
    import configparser as ConfigParser
except:
    from six.moves import configparser as ConfigParser
import time
from joblib.parallel import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import pandas as pd
try:
    import requests
except:
    pass
try:
    import pubchempy as pcp
except:
    pass

try:
    from rdkit.Chem import Draw,MolFromInchi,AllChem, rdMolDescriptors
    from rdkit import Chem, DataStructs
except:
    pass

import configparser
import os
try:
    import ete4
except:
    import ete3 as ete4

import shutil
import re
import operator as op
try:
    from functools import reduce
except:
    pass
import psutil
from scipy.spatial.distance import cityblock, squareform, pdist

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except:
    pass

# from collections import defaultdict
try:
    import dgl
    from dgl.nn.pytorch.glob import AvgPooling
    from dgllife.model import load_pretrained
    from dgllife.model.model_zoo import *
    from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
except:
    pass
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
except:
    pass

global TIME
TIME = time.time()
# from scipy.spatial.distance import pdist, squareform

def collapse_metabolites(data, meta_df):
    mets = data.columns.values
    m_rename = {m: '__'.join([m.split('_')[0], m.split('__')[-1].split(': ')[-1]]) for m in mets}

    data_renamed = data.rename(columns=m_rename)
    kept = [m for m in data_renamed.columns.values if m.split('_')[-1]!='NA']
    data_renamed = data_renamed[kept]
    data_collapsed = data_renamed.T.groupby(data_renamed.columns).sum().T

    print(f'Dataset has {data_renamed.shape[1]} named metabolites')

    meta_df = meta_df.rename(index=m_rename)
    meta_df = meta_df.loc[~meta_df.index.duplicated(),:]
    # new_fran = copy.deepcopy(dataset)
    # new_fran['X'] = data_collapsed
    # new_fran['distances'] = new_fran['distances'].rename(columns=m_rename).rename(index=m_rename)
    # new_fran['distances'] = new_fran['distances'].loc[
    #     ~new_fran['distances'].index.duplicated(), ~new_fran['distances'].columns.duplicated()]
    #
    # new_fran['taxonomy'] = new_fran['taxonomy'].rename(columns=m_rename)
    # new_fran['taxonomy'] = new_fran['taxonomy'].loc[:, ~new_fran['taxonomy'].columns.duplicated()]
    # new_fran['variable_names'] = np.unique(new_fran['X'].columns.values)
    print(f'Collapsed dataset has {data_collapsed.shape[1]} named metabolites')
    return data_collapsed, meta_df


def collapse_dataset(dataset):
    mets = dataset['X'].columns.values
    m_rename = {m: '__'.join([m.split('_')[0], m.split('__')[-1].split(': ')[-1]]) for m in mets}

    data_renamed = dataset['X'].rename(columns=m_rename)
    kept = [m for m in data_renamed.columns.values if m.split('_')[-1] != 'NA']
    data_renamed = data_renamed[kept]
    data_collapsed = data_renamed.T.groupby(data_renamed.columns).sum().T

    print(f'Dataset has {data_renamed.shape[1]} named metabolites')

    # meta_df = meta_df.rename(index=m_rename)
    # meta_df = meta_df.loc[~meta_df.index.duplicated(), :]
    new_fran = copy.deepcopy(dataset)
    new_fran['X'] = data_collapsed
    new_fran['distances'] = new_fran['distances'].rename(columns=m_rename).rename(index=m_rename)
    new_fran['distances'] = new_fran['distances'].loc[
        ~new_fran['distances'].index.duplicated(), ~new_fran['distances'].columns.duplicated()]

    new_fran['taxonomy'] = new_fran['taxonomy'].rename(columns=m_rename)
    new_fran['taxonomy'] = new_fran['taxonomy'].loc[:, ~new_fran['taxonomy'].columns.duplicated()]
    new_fran['variable_names'] = np.unique(new_fran['X'].columns.values)
    print(f'Collapsed dataset has {data_collapsed.shape[1]} named metabolites')
    return new_fran



def collate(gs):
    return dgl.batch(gs)

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def read_ctsv(filename, index_col=0, header=0):
    if '.tsv' in filename:
        df = pd.read_csv(filename, index_col=index_col, header=header, sep='\t')
    elif '.csv' in filename:
        df = pd.read_csv(filename, index_col=index_col, header=header)
    else:
        raise ImportError(f'{filename} is neither a .csv or .tsv')
    return df

def make_16s_tree(seq_config, sequences, sequence_data, save_path):
    print(os.getcwd())
    # os.chdir('utilities/')
    phylo_inputs='../phylo_placement/input/phylo_placement_inputs.fa'
    if not os.path.isdir('../phylo_placement/input/'):
        os.mkdir('../phylo_placement/input/')
    phylo_outputs='../phylo_placement/placement_outputs/'
    if not os.path.isdir(phylo_outputs):
        os.mkdir(phylo_outputs)
    if sequences is not None or seq_config.split('.')[-1]=='csv' or \
            seq_config.split('.')[-1]=='tsv':
        if sequences is None:
            sequences = read_ctsv(seq_config)
        if sequences.shape[0] < sequence_data.shape[1]:
            missing_seqs = set(sequence_data.columns.values) - set(sequences.index.values)
            print('WARNING: the following ASVs are missing assigned sequences: ')
            print(missing_seqs)
            print('')
        seqs = list(set(sequence_data.columns.values).intersection(set(sequences.index.values)))
        sequence_data = sequence_data[seqs]
        sequences =sequences.loc[seqs]
        # fname=phylo_inputs
        # outfname = self.save_path+ '/tmp/'
        lines=[]
        for asv in sequences.index.values:
            lines.append('>'+asv + '\n')
            lines.append(sequences.loc[asv] + '\n')
        with open(phylo_inputs,'w') as f:
            f.writelines(lines)
    elif seq_config.split('.')[-1]=='txt' or seq_config.split('.')[-1]=='fa':
        with open(seq_config,'r') as f:
            lines = f.readlines()
        with open(phylo_inputs,'w') as f:
            f.writelines(lines)
    else:
        raise ValueError('Please provide sequences for 16s data, either in the form of column names in input \n'
                         'sequence counts, csv/tsv file with labels as index and sequences as first column, or .fa/.txt\n'
                         'file.')
    # save_tmp = self.save_path.replace('(','\(').replace(')','\)') + '/tmp/'
    print(os.getcwd())
    os.system(f"python ../phylo_placement/scripts/place_seqs.py --v4-region-start 868 --v4-region-end 1161 "
              f"--refpkg ../phylo_placement/refpkg/RDP-11-5_TS_Processed.refpkg/ "
              f"--query-reads '{phylo_inputs}' --output-folder '{phylo_outputs}'")
    # os.chdir('..')
    filename = os.path.basename('../' + phylo_outputs+'newick_tree_query_reads.nhx')
    dest = os.path.join(save_path+ '/tmp/', filename)
    shutil.move(phylo_outputs+'newick_tree_query_reads.nhx', dest)
    # os.chdir('..')
    print(os.getcwd())
    if 'newick_tree_query_reads.nhx' in os.listdir(save_path+ '/tmp/'):
        sequence_tree=ete4.Tree(save_path+ '/tmp/' + 'newick_tree_query_reads.nhx')
        sequence_tree.write(features=['name'], outfile=save_path + '/sequence_tree.nhx', format=0)
    else:
        raise FileNotFoundError("No newick tree in output folder. Check to make sure phylo placement ran.")

    return sequence_tree

def collapse_by_genus(sequence_data):
    gen_names = [f.split('g__')[0] + 'g__' + re.sub(r'[0-9]', '', f.split('g__')[1].split('_')[0].split('-')[0]) for f
                 in
                 sequence_data.columns.values]
    mdata_collapsed = sequence_data.copy()
    mdata_collapsed.columns = gen_names
    sequence_data = mdata_collapsed.groupby(lambda x: x, axis=1).sum()
    return sequence_data

def get_asv_labels(taxa_strings):
    return {taxa_strings[i]: f'ASV {i}' for i in range(len(taxa_strings))}


def get_taxa_df_from_strings(taxa_strings, tax_to_label_dict, ref_dict=None):
    ix_labels=['Kingdom','Phylum','Class','Order','Family','Genus']
    if '|' in taxa_strings[0]:
        delim='|'
    else:
        delim=';'
    if len(taxa_strings[0].split(delim))==7:
        ix_labels.append('Species')
    elif ref_dict is not None and len(list(ref_dict.values())[0].split(delim))==7:
        ix_labels.append('Species')
    taxa_dict={}
    ct = 0
    for tax in taxa_strings:
        tls = tax.split(delim)
        if len(tls)>1:
            val = {ix_labels[i]:tls[i].split('__')[-1] for i in range(len(ix_labels))}
        elif ref_dict is not None:
            if tax not in ref_dict.keys():
                continue
            tax = ref_dict[tax]
            tls = tax.split(delim)
            val = {ix_labels[i]: tls[i].split('__')[-1] for i in range(len(ix_labels))}
        else:
            print(tax)
            raise ValueError('Need reference file if only partial taxonomy in data')
        if tax not in tax_to_label_dict.keys():
            lab = f'ASV {len(tax_to_label_dict.keys()) + ct}'
            ct += 1
            taxa_dict[lab] = val
        else:
            taxa_dict[tax_to_label_dict[tax]] = val
    return pd.DataFrame(taxa_dict)

def collapse_nodes(tree, nodes_to_collapse):
    for same_nodes in nodes_to_collapse:

        node_list = [n.name for n in tree.traverse() if n.name in same_nodes]
        if len(node_list) > 0:
            node_list = [tree.search_nodes(name=n)[0] for n in node_list]
            node_dists = [n.dist for n in node_list]
            while 1:
                try:
                    parent = tree.get_common_ancestor(node_list)
                except:
                    amx = np.argmax(node_dists)
                    node_list.pop(amx)
                break
            if len(node_list) > 0:
                for node in node_list:
                    node.detach()
                parent.add_child(name=same_nodes[0], dist=np.mean(node_dists))

def collapse_nodes_short(tree, nodes_to_collapse):
    for same_nodes in nodes_to_collapse:

        node_list = [n for n in tree.traverse() if n.name==same_nodes]
        if len(node_list) > 0:
            node_dists = [n.dist for n in node_list]
            while 1:
                try:
                    parent = tree.get_common_ancestor(node_list)
                except:
                    amx = np.argmax(node_dists)
                    node_list.pop(amx)
                break
        if len(node_list) > 0:
            for node in node_list:
                node.detach()
            parent.add_child(name=same_nodes, dist=np.mean(node_dists))

def make_wgs_tree(taxa_strings_list, reference_tree='./utilities/mpa_vJan21.nwk.txt', reference_mapper='./utilities/mpa_vJan21_CHOCOPhlAnSGB_202103_SGB2GTDB.tsv'):
    is_genera_bool = all([len(t.split(';')) == 6 for t in taxa_strings_list]) or all([len(t.split('|')) == 6 for t in taxa_strings_list])
    if ';' in taxa_strings_list[0]:
        delim = ';'
    else:
        delim = '|'
    if reference_mapper is None:
        keep_nodes = taxa_strings_list
        tree = ete4.Tree(reference_tree)
        try:
            tree.prune(keep_nodes, preserve_branch_length=True)
        except:
            names_used = []
            nodes_to_collapse = []
            for n in tree.traverse():
                if n.is_leaf():
                    if is_genera_bool:
                        n.name = delim.join(n.name.split(delim)[1:-1])
                    else:
                        n.name = delim.join(n.name.split(delim)[1:])
                    if n.name in names_used:
                        nodes_to_collapse.append(n.name)
                    else:
                        names_used.append(n.name)

            collapse_nodes_short(tree, np.unique(nodes_to_collapse))
            ktemp = list(set(keep_nodes).intersection(set([n.name for n in tree.traverse() if n.is_leaf()])))
            if len(ktemp)==0:
                print('ERROR: NO NODES IN TREE! MAKE SURE TREE IS RIGHT!')
            elif len(ktemp)<len(keep_nodes):
                print(f'{len(keep_nodes)-len(ktemp)} nodes not in tree and will be removed from dataset')
            tree.prune(ktemp, preserve_branch_length=True)
        return tree, None
    else:
        print(f'{len(taxa_strings_list)} taxa to map to tree')
        taxa_labels = read_ctsv(reference_mapper, header=None, index_col=None)
        taxa_labels.iloc[:,1] = taxa_labels.iloc[:,1].str.replace('|',';')
        tree = ete4.Tree(reference_tree,quoted_node_names=True, format=1)

        # is_genera_bool = all([len(t.split(';'))==6 for t in taxa_strings_list])
        if is_genera_bool:
            print('Creating genera-level WGS tree as data provided looks to be at the genus level')
            taxa_labels.iloc[:,1]= taxa_labels.iloc[:,1].apply(lambda x:';'.join(x.split(';')[:-1]))
        taxa_labels = taxa_labels.set_index(1)

        new_taxa_df = {}
        for k in taxa_labels.index.values:
            if ',' in k:
                new_taxa_df[k.split(',')[0]] = taxa_labels.loc[k]
                new_taxa_df[k.split(',')[1]] = taxa_labels.loc[k]
            else:
                new_taxa_df[k] = taxa_labels.loc[k]

        taxa_labels = pd.Series(new_taxa_df)
        partial_taxa_bool=all([len(t.split(';'))<6 for t in taxa_strings_list])
        if partial_taxa_bool:
            print('Only partial taxonomy provided')
            ref_dict={}
            for k in taxa_labels.index.values:
                if ',' in k:
                    ks = k.split(',')
                    for kk in ks:
                        ref_dict[kk.split(';')[-1].split('s__')[-1].replace(' ','_')]=k
                else:
                    ref_dict[k.split(';')[-1].split('s__')[-1].replace(' ','_')] = k
            # ref_dict = {k.split(';')[-1].split('s__')[-1].replace(' ','_'):k for k in taxa_labels.index.values}
            # ref_dict = {}
            # for key in taxa_labels.squeeze().keys():
            #     ref_dict[key.split(';')[-1].split('s__')[-1].replace(' ', '_')] = key
            tmp = copy.copy(taxa_strings_list)
            taxa_strings_list=[]
            for t in tmp:
                if t in ref_dict.keys():
                    taxa_strings_list.append(ref_dict[t])
                else:
                    taxa_strings_list.append(t)
        else:
            ref_dict=None

        node_to_species_taxa = {}
        nodes_to_collapse=[]
        not_in_ct = 0
        unmatched = []
        for t in taxa_strings_list:
            if t in taxa_labels.index.values:
                out = taxa_labels.loc[t].values.squeeze()
                if len(out.shape) > 0:
                    nodes_to_collapse.append([o.replace('SGB', '').replace('_group', '') for o in out])
                    for o in out:
                        node_to_species_taxa[o.replace('SGB', '').replace('_group', '')] = t
                else:
                    node_to_species_taxa[out.item().replace('SGB', '').replace('_group', '')] = t
            else:
                not_in_ct += 1
                unmatched.append(t)

        for u in unmatched:
            ref_keys=[]
            for r in taxa_labels.index.values:
                if partial_taxa_bool and u.split('_')[0] in r.split('g__')[-1].split(';')[0]:
                    if isinstance(taxa_labels[r], pd.DataFrame):
                        ref_keys.extend([v.item() for v in taxa_labels[r].values])
                    else:
                        ref_keys.append(taxa_labels[r].item())
                elif u.split('g__')[-1].split(';')[0] in r.split('g__')[-1].split(';')[0]:
                    if isinstance(taxa_labels[r], pd.DataFrame):
                        ref_keys.extend([v.item() for v in taxa_labels[r].values])
                    else:
                        ref_keys.append(taxa_labels[r].item())
            #
            # if partial_taxa_bool:
            #     ref_keys = [taxa_labels[r].item() for r in taxa_labels.index.values if u.split('_')[0] in r.split('g__')[-1].split(';')[0]]
            # else:
            #     ref_keys = [taxa_labels[r].item() for r in taxa_labels.index.values if u.split('g__')[-1].split(';')[0] in r.split('g__')[-1].split(';')[0]]
            nodes_to_collapse.append([o.replace('SGB', '').replace('_group', '') for o in ref_keys])
            # keep_nodes.extend([o.replace('SGB', '').replace('_group', '') for o in ref_keys])
            for n in ref_keys:
                node_to_species_taxa[n.replace('SGB', '').replace('_group', '')] = u


        if not_in_ct>0:
            print(f'WARNING: {not_in_ct} species in data are not mapped in reference mapper.\nCheck to see that reference mapper is the correct one for this dataset')

        keep_nodes = [t for t in tree.traverse() if t.is_leaf and t.name in list(node_to_species_taxa.keys())]
        # if len(node_to_species_taxa.keys())>len(keep_nodes):
        #     print(f'WARNING: {len(keep_nodes)-len(node_to_species_taxa.keys())} species in data are not mapped to node in tree.\nCheck to see that reference tree is the correct one for this dataset')
        tree.prune(keep_nodes, preserve_branch_length=True)

        if len(nodes_to_collapse)>0:
            print(f'WARNING:{len(nodes_to_collapse)} different species in data are mapped to duplicate nodes')
            collapse_nodes(tree, nodes_to_collapse)

        sp_in_tree = []
        taxa_to_node = {}
        taxa_to_cad = {}
        for t in tree.traverse():
            if t.is_leaf():
                if t.name == '':
                    t.delete()
                    continue
                sp_in_tree.append(node_to_species_taxa[t.name])
                parent_taxa = ';'.join(node_to_species_taxa[t.name].split(';')[:-1])
                t.name = node_to_species_taxa[t.name]
                parent = t.up
                while parent is not None:
                    if parent_taxa not in taxa_to_cad.keys():
                        distances = [c.dist for c in parent.children]
                        taxa_to_cad[parent_taxa] = np.mean(distances)
                        taxa_to_node[parent_taxa] = parent
                    parent = parent.up
                    parent_taxa = ';'.join(parent_taxa.split(';')[:-1])

        sp_missing = list(set(taxa_strings_list) - set(sp_in_tree))
        if len(sp_missing)>0:
            print(
                f'WARNING: {len(sp_missing)} species in data are not mapped to node in tree.\nCheck to see that reference tree is the correct one for this dataset')

            for sp in sp_missing:
                sp_sm = ';'.join(sp.split(';')[:-1])
                while sp_sm not in taxa_to_cad.keys():
                    sp_sm = ';'.join(sp_sm.split(';')[:-1])
                sp_dist = taxa_to_cad[sp_sm]
                sp_parent = taxa_to_node[sp_sm]
                sp_parent.add_child(name=sp, dist=sp_dist)

        # assert(set([t.name for t in tree.traverse() if t.is_leaf()]) == set(taxa_strings_list))
        # families = np.unique([n.split(';')[-2] for n in taxa_strings_list])
        # colors = ete4.random_color(num=len(families))
        # cdict = {f:c for f,c in zip(families,colors)}
        # for n in tree.traverse():
        #     if n.is_leaf():
        #         n.add_face(ete4.TextFace(n.name.split(';')[-1], fgcolor=cdict[n.name.split(';')[-2]]), column=0, position='branch-top')
        #     else:
        #         n.name = n.name.split(';')[-1]
        # ts = ete4.TreeStyle()
        # ts.show_leaf_name = True
        # disp_tree.render(save_path+'/tree_for_WGS.pdf', tree_style=ts)
        # plt.close()
        # tree.write(features=['name'], outfile=save_path + '/sequence_tree.nhx', format=0)
        return tree, ref_dict

def clean_labels(taxa_to_label):
    new_dict={}
    for asv_name, taxa_label in taxa_to_label.items():
        taxa_label = taxa_label.split(' ')[0]
        taxa_label = taxa_label.replace('-', '').replace(' ', '')
        nums = re.findall(r'\d+', taxa_label)
        for n in nums:
            taxa_label = taxa_label.replace(n, '')
        taxa_ls = taxa_label.split(';')

        while len(taxa_ls[-1]) <= 3 or taxa_ls[3:].isupper():
            taxa_ls.pop()

        if len(taxa_ls) == 7:
            clean_lab = taxa_ls[-2].replace('g__', '') + ', ' + taxa_ls[-1].replace('s__', '')
        else:
            clean_lab = taxa_ls[-1]
        new_dict[asv_name] = clean_lab
    return new_dict


def relabel_tree(tree, label_to_clean_label):
    ct = 0
    for t in tree.traverse():
        if t.is_leaf():
            if t.name not in label_to_clean_label.keys():
                t.name = f'ASV {len(label_to_clean_label.items())+ct}'
                ct += 1
            else:
                t.name = label_to_clean_label[t.name]
    return tree


# def make_wgs_tree(sequence_data, save_path):
#     print('\nMaking WGS tree')
#     names_to_nodes = {}
#     for name in sequence_data.columns:
#         names_to_nodes[name] = ete4.TreeNode(name=name, dist=1.0)
#     nodes = copy.copy(np.unique(sequence_data.columns.values)).tolist()
#     i=0
#     while (len(nodes))>0:
#         k=nodes.pop()
#         v=names_to_nodes[k]
#         taxonomy = k.split(';')
#         parent = ';'.join(taxonomy[:-1])
#         if len(taxonomy)==1:
#             parent = taxonomy[0]
#             print(len(nodes))
#         if parent in names_to_nodes.keys():
#             parent_node=names_to_nodes[parent]
#         else:
#             parent_node = ete4.Tree(name=parent, dist=1.0)
#             names_to_nodes[parent] = parent_node
#         if k not in [n.name for n in parent_node.children]:
#             parent_node.add_child(v)
#             if len(taxonomy)>2:
#                 nodes.append(parent)
#
#     sequence_tree = names_to_nodes['d__Bacteria']
#     disp_tree = sequence_tree.copy()
#
#     families = np.unique([n.split(';')[-2] for n in sequence_data.columns.values])
#     colors = ete4.random_color(num=len(families))
#     cdict = {f:c for f,c in zip(families,colors)}
#     for n in disp_tree.traverse():
#         if n.is_leaf():
#             n.add_face(ete4.TextFace(n.name.split(';')[-1], fgcolor=cdict[n.name.split(';')[-2]]), column=0, position='branch-top')
#         else:
#             n.name = n.name.split(';')[-1]
#     ts = ete4.TreeStyle()
#     ts.show_leaf_name = True
#     disp_tree.render(save_path+'/tree_for_WGS.pdf', tree_style=ts)
#     plt.close()
#     sequence_tree.write(features=['name'], outfile=save_path + '/sequence_tree.nhx', format=0)
#     return sequence_tree

def get_sequence_distance_matrix(sequence_data, sequence_tree):
    print('\nObtaining sequence data distances')
    dist = {}
    tot = ncr(sequence_data.shape[1], 2)
    perc = 0.01
    ct=0

    for seq1,seq2 in itertools.combinations(sequence_data.columns.values,2):
        try:
            d = sequence_tree.get_distance(seq1, seq2)
        except:
            continue
        if seq1 not in dist.keys():
            dist[seq1] = {}
            dist[seq1][seq1]=0
        if seq2 not in dist.keys():
            dist[seq2] = {}
            dist[seq2][seq2]=0
        # d = sequence_tree.get_distance(seq1, seq2)
        dist[seq1][seq2] = d
        dist[seq2][seq1] = d
        ct += 1
        if ct == int(np.round(perc * tot)):
            print(int(perc * 100), '% complete')
            perc += 0.1

    sequence_dist_df = pd.DataFrame(dist)
    return sequence_dist_df
    # sequence_dist_df.to_csv(self.save_path+'/seq_dist.csv')

def get_epsilon(data):
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = np.max(vals)
    epsilon = 0.1*np.min(np.abs(vals))
    return epsilon

def transform_func(data_train, data_test=None, standardize_from_training_data=True, log_transform=True):
    # data_trans = (data - np.mean(data.values, 0) + 1e-10
    #                         ) / (np.std(data.values, 0) + 1e-10)
    # eps = get_epsilon(data_train)
    # data_train = np.log(data_train + 1)
    if log_transform:
        eps = get_epsilon(data_train)
        data_train = np.log(data_train + eps)

    if isinstance(data_train, pd.DataFrame):
        mean = np.mean(data_train.values, 0) + 1e-10
        std = np.std(data_train.values, 0) + 1e-10
    else:
        mean = np.mean(data_train, 0) + 1e-10
        std = np.std(data_train, 0) + 1e-10
    data_train = (data_train - mean) / std
    if data_test is not None and standardize_from_training_data:
        if log_transform:
            data_test = np.log(data_test + eps)
        # data_test = np.log(data_test + 1)
        data_test = (data_test - mean) / std
        return data_train, data_test
    elif data_test is not None:
        if log_transform:
            data_test = np.log(data_test + eps)
        mean = np.mean(data_test,0) + 1e-10
        std = np.std(data_test,0) + 1e-10
        data_test = (data_test - mean)/std
        return data_train, data_test
    else:
        return data_train

def filter_by_presence_func(data_train, data_test=None, perc_present_in=10, limit_of_detection=0):
    ppi = int(perc_present_in)
    if isinstance(data_train, pd.DataFrame):
        tmp = copy.deepcopy(data_train.values)
    else:
        tmp = copy.deepcopy(data_train)
    mets = np.zeros(tmp.shape)
    mets[np.abs(tmp)>limit_of_detection] = 1
    mets_counts = np.sum(mets,0)
    mets_keep = np.where(mets_counts >= np.round(ppi*mets.shape[0]/100))[0]
    if data_test is not None:
        if isinstance(data_train, pd.DataFrame):
            return data_train.iloc[:, mets_keep], data_test.iloc[:, mets_keep]
        else:
            return data_train[:, mets_keep], data_test[:, mets_keep]
    else:
        if isinstance(data_train, pd.DataFrame):
            return data_train.iloc[:, mets_keep]
        else:
            return data_train[:, mets_keep]

def filter_by_cov_func(data_train, data_test=None, cov_percentile=50):
    cp=int(cov_percentile)
    if isinstance(data_train, pd.DataFrame):
        variances = np.std(data_train.values, 0) / np.abs(np.mean(data_train.values, 0))
    else:
        variances = np.std(data_train, 0) / np.abs(np.mean(data_train, 0))
    ixs_keep = np.where(variances > np.percentile(variances, cp))[0].tolist()
    if data_test is not None:
        if isinstance(data_train, pd.DataFrame):
            return data_train.iloc[:, ixs_keep], data_test.iloc[:, ixs_keep]
        else:
            return data_train[:, ixs_keep], data_test[:, ixs_keep]
    else:
        if isinstance(data_train, pd.DataFrame):
            return data_train.iloc[:, ixs_keep]
        else:
            return data_train[:, ixs_keep]


def get_inchikeys(idType_id, met=None, i=0, out_id='InChIKey'):
    if i%100==0 and i!=0:
        print(f'{i} metabolites left')
    for idType, id in idType_id:
        if idType.lower()!=out_id.lower():
            try:
                string = f"https://cts.fiehnlab.ucdavis.edu/rest/convert/{idType}/{out_id}/{id}"
                r = requests.get(string)
                out = r.json()[0]['results'][0]
            except:
                out = None
            if not isinstance(out, str):
                continue
            else:
                return met, out
        else:
            out = id
            return met,out
    return met, out


def get_met_keys(metab_df, key_type='InChIKey'):
    metab_df.columns = [c.lower() for c in metab_df.columns.values]
    mets_to_query = list(zip(np.arange(len(metab_df.index.values)), metab_df.index.values))
    id_vec = []
    num_wo_id=0
    while len(mets_to_query) > 0:
        i, metabolite = mets_to_query.pop(0)
        id_types=[]
        if 'inchikey' in metab_df.columns.values:
            inchi_id = metab_df.loc[metabolite]['inchikey']
            if isinstance(inchi_id, str):
                idType = 'InChIKey'
                id = inchi_id
                id_types.append((idType, id))
        if 'kegg' in metab_df.columns.values:
            kegg_id = metab_df.loc[metabolite]['kegg']
            if isinstance(kegg_id, str):
                idType = 'KEGG'
                id = kegg_id
                id_types.append((idType, id))
        if 'hmdb' in metab_df.columns.values:
            hmdb_id = metab_df.loc[metabolite]['hmdb']
            if isinstance(hmdb_id, str):
                idType = 'Human Metabolome Database'
                tmp = hmdb_id.split('B')
                try:
                    id = tmp[0] + 'B00' + tmp[1]
                    id_types.append((idType, id))
                except:
                    pass
        if len(id_types)==0:
            num_wo_id+=1
            # print(f'No ID for {metabolite}, {i}')
            continue
        id_vec.append((id_types, metabolite))
    if num_wo_id>0:
        print(f'Warning: {num_wo_id} metabolites have no KEGG, HMDB, or INCHIKEY ID')
    if 'inchi' not in key_type.lower():
        out_id = 'InChIKey'

    else:
        out_id = key_type
    # import pdb; pdb.set_trace()
    # inchis_out = []
    # for i,(idt_id,met) in enumerate(id_vec):
    #     inchis_out.append(get_inchikeys(idt_id,met=met,i=len(id_vec)-i,out_id=out_id))
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    inchis_out = Parallel(n_jobs=8)(delayed(get_inchikeys)(idt_id,met=met,i=len(id_vec)-i,out_id=out_id) for i,(idt_id,met) in enumerate(id_vec))
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        print('Killing process with pid {}'.format(subproc))
        psutil.Process(subproc).terminate()

    res_dict = {m: {out_id.lower(): id} for m, id in inchis_out}
    res_df = pd.DataFrame(res_dict).T
    try:
        metab_df = metab_df.join(res_df, how='left')
    except:
        pass
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    print(f"{sum(~res_df[out_id.lower()].isna())} metabolites ID'd with {out_id}")
    if out_id!=key_type:
        res_df=pd.DataFrame(get_pubchem_something(res_df, out_id.lower(), out_type=key_type))
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        print('Killing process with pid {}'.format(subproc))
        psutil.Process(subproc).terminate()

    print(f"{sum(~res_df[key_type.lower()].isna())} metabolites ID'd with {key_type}")
    try:
        return metab_df.join(res_df,how='left')
    except:
        return metab_df

def add_family_names_to_tree(dataset, features):
    in_feats = features
    taxonomy = dataset['taxonomy']
    query_parent_dict = {}
    # tree.prune(in_feats, preserve_branch_length=True)
    # query_parent_dict['COMPOUNDS'] = {}
    weights_dict = {}
    taxonomy=taxonomy.T
    it = 0
    for taxa in in_feats:
        classification = taxonomy.loc[taxa].dropna()
        it += 1
        for l in np.arange(1, len(classification)):
            if classification.iloc[l - 1].upper() not in query_parent_dict.keys():
                query_parent_dict[classification.iloc[l - 1].upper()] = [
                    classification.iloc[l].upper()]
            else:
                if classification.iloc[l].upper() not in query_parent_dict[
                    classification.iloc[l - 1].upper()]:
                    query_parent_dict[classification.iloc[l - 1].upper()].append(
                        classification.iloc[l].upper())
        if 'COMPOUNDS' not in query_parent_dict.keys():
            query_parent_dict['COMPOUNDS'] = [classification.iloc[0].upper()]
        else:
            if classification.iloc[0].upper() not in query_parent_dict['COMPOUNDS']:
                query_parent_dict['COMPOUNDS'].append(classification.iloc[0].upper())
        if classification.iloc[-1].upper() not in query_parent_dict.keys():
            query_parent_dict[classification.iloc[-1].upper()] = [taxa]
        else:
            query_parent_dict[classification.iloc[-1].upper()].append(taxa)

    # root = query_parent_dict[None][0]
    root = 'COMPOUNDS'
    query_root = ete4.TreeNode(name=root)
    parents, added = [query_root], set([root])
    while parents:
        nxt = parents.pop()
        child_nodes = {child: ete4.TreeNode(name=child) for child in query_parent_dict[nxt.name]}
        for child in query_parent_dict[nxt.name]:
            nxt.add_child(child_nodes[child], name=child, dist=1)
            if child not in added:
                if child in query_parent_dict.keys():
                    parents.append(child_nodes[child])
                added.add(child)
    query_root.write(format=1, outfile="new_tree.nw")

def get_map4_fingerprint(df):
    fv=[]
    indices=[]
    dfn=df.fillna(np.nan)
    df=dfn.fillna('')
    for k in df.index.values:

        ii=df[k]
        if ii is None or len(ii)==0:
            fv.append(None)
            indices.append(None)
        else:
            # try:
            fa=MAP4.calculate(Chem.MolFromSmiles(ii))
            # fa = Chem.MolFromSmiles(ii)
            # except:
            #     continue
            fv.append(fa)
            indices.append(k)
            # import pdb; pdb.set_trace()
    return pd.Series(fv, index=indices, name='fingerprints')


def get_infomax_fingerprint(inchi_df):
    graphs = []
    mets=[]
    for met, inchi in inchi_df.items():
        if not isinstance(inchi, str):
            continue
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            continue
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=True)
        graphs.append(g)
        mets.append(met)

    model = load_pretrained('gin_supervised_infomax');  # contextpred infomax edgepred masking
    model.to('cpu')
    model.eval()

    data_loader = DataLoader(graphs, batch_size=256, collate_fn=collate, shuffle=False)

    readout = AvgPooling()

    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        bg = bg.to('cpu')
        nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
                  bg.ndata.pop('chirality_type').to('cpu')]
        efeats = [bg.edata.pop('bond_type').to('cpu'),
                  bg.edata.pop('bond_direction_type').to('cpu')]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    fingerprint_df = pd.DataFrame(mol_emb, index=mets)
    # dist_mat_df = pd.DataFrame(squareform(pdist(mol_emb)), index=mets, columns=mets)
    return fingerprint_df

def get_rdkit_fingerprint(inchi_df, ftype):
    new_f=[]
    indices=[]
    ftype=ftype.lower()
    for k in inchi_df.index.values:
        i=inchi_df[k]
        if i is not None and isinstance(i, str):
            try:
                mol=MolFromInchi(i)
            except:
                print(f"error when converting {i} to molecule in rdkit")
                continue
            if mol is None:
                continue
            if ftype=='rdkit':
                fa = Chem.RDKFingerprint(mol)
            if ftype=='morgan':
                fa = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            if ftype=='mqn':
                fa = rdMolDescriptors.MQNs_(mol)
            else:
                ValueError('Please specify either "morgan", "rdkit", or "mqn" as fingerprint type option!')
            new_f.append(fa)
            indices.append(k)
        else:
            new_f.append(None)
            indices.append(k)
    return pd.Series(new_f, index=indices, name='fingerprints')

def call_pubchem(inchi_key, met, key_type, ii = None):
    if ii is not None:
        if ii%100==0:
            print(f'{ii} metabolites IDd with pubchem')
    if inchi_key is not None and isinstance(inchi_key, str):
        try:
            # string=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{key_type.lower()}/{inchi_key}/record/JSON"
            # r = requests.get(string)
            # out = r.json()[0]['results'][0]
            compound = pcp.get_compounds(inchi_key, namespace=key_type.lower(), listkey_count = 5)
            TIME = time.time()
            # fp = compound[0].cactvs_fingerprint
            return {met:compound[0]}
        except:
            return {met:None}
    else:
        return {met:None}


def get_pubchem_something(met_df, key_type, out_type='fingerprints'):
    # inchi_ls = inchi_df.values
    # print(inchi_ls)
    # print(key_type)
    met_df_ = copy.deepcopy(met_df)
    pouts = []
    TIME=time.time()
    for ii, (met, i) in enumerate(met_df_[key_type].items()):
        pouts.append(call_pubchem(i, met, key_type, ii))
        if ii%5==0 and ii>0:
            while time.time()-TIME<1:
                time.sleep(0.1)
            TIME=time.time()

    # current_process = psutil.Process()
    # subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    # pouts = Parallel(n_jobs=10)(delayed(call_pubchem)(i, met, key_type, ii)
    #                                             for ii, (met,i) in enumerate(met_df_[key_type].items()))
    # subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    # for subproc in subproc_after - subproc_before:
    #     print('Killing process with pid {}'.format(subproc))
    #     psutil.Process(subproc).terminate()

    f_save={}
    p_tries=10
    last_counter=0
    for i in range(p_tries):
        met_redo={}
        none_counter=0
        for ftemp in pouts:
            (met,fp),=ftemp.items()
            if fp is None:
                # try:
                met_redo[met]=met_df.loc[met].item()
                none_counter+=1
                # except:
                #     continue
            else:
                if 'smiles' in out_type.lower():
                    f_save[met]=fp.isomeric_smiles
                if 'fingerprints' in out_type.lower():
                    f_save[met]=fp.cactvs_fingerprint
        print(f'{len(met_redo.keys())}/{met_df.shape[0]} metabolites failed pubchem call {i + 1}')
        time.sleep(5)
        if last_counter==none_counter:
            break
        else:
            last_counter=none_counter
        # n_jobs = (10-i)
        # pouts = Parallel(n_jobs=n_jobs)(delayed(call_pubchem)(i, met, key_type, ii) for ii, (met,i) in enumerate(met_redo.items()))
        # pouts = [call_pubchem(i, met, key_type, ii) for ii, (met, i) in enumerate(met_redo.items())]
        pouts = []
        TIME = time.time()
        for ii, (met, i) in enumerate(met_redo.items()):
            pouts.append(call_pubchem(i, met, key_type, ii))
            if ii % 5 == 0 and ii > 0:
                while time.time() - TIME < 1:
                    time.sleep(0.1)
                TIME = time.time()

    for ftemp in pouts:
        (met, fp), = ftemp.items()
        if fp is not None:
            if 'smiles' in out_type.lower():
                f_save[met] = fp.isomeric_smiles
            if 'fingerprints' in out_type.lower():
                f_save[met] = fp.cactvs_fingerprint
        else:
            f_save[met] = fp

    fseries=pd.Series(f_save,name=out_type.lower())[met_df.index.values]
    return fseries

def tanimoto(a, b):
    both = len(np.where((a + b) > 1)[0])
    return both / ((np.sum(a) + np.sum(b)) - both)

def get_dist_mat(fingerprint_dict, ftype='rdkit'):
    if ftype=='pubchem':
        ndict = {}
        for k, v in fingerprint_dict.items():
            if v is None or v is np.nan:
                continue
            ndict[k] = np.fromstring(v, 'u1') - ord('0')
        fdf = pd.DataFrame(ndict).T
        dist_df = pd.DataFrame(squareform(pdist(fdf.values, metric='jaccard')), index=fdf.index.values, columns=fdf.index.values)
    else:
        dist_dict = {}
        for m1, m2 in itertools.combinations(fingerprint_dict.keys(), 2):
            if fingerprint_dict[m1] is None or fingerprint_dict[m2] is None:
                continue
            if np.isnan(fingerprint_dict[m1]).any() or np.isnan(fingerprint_dict[m2]).any():
                continue
            # if np.isnan(fingerprint_dict[m1]) or np.isnan(fingerprint_dict[m2]):
            #     continue
            if m1 not in dist_dict.keys():
                dist_dict[m1] = {}
                dist_dict[m1][m1]=0
            if m2 not in dist_dict.keys():
                dist_dict[m2] = {}
                dist_dict[m2][m2]=0
            if ftype=='rdkit' or ftype=='morgan':
                sim=DataStructs.FingerprintSimilarity(fingerprint_dict[m1], fingerprint_dict[m2])
                dist=1-sim
            elif ftype=='mqn':
                dist=cityblock(fingerprint_dict[m1],fingerprint_dict[m2])
            elif ftype=='map4':
                dist = ENC.get_distance(fingerprint_dict[m1], fingerprint_dict[m2])
            else:
                sim = tanimoto(np.fromstring(fingerprint_dict[m1], 'u1') - ord('0'), np.fromstring(fingerprint_dict[m2], 'u1') - ord('0'))
                dist=1-sim

            dist_dict[m1][m2] = dist
            dist_dict[m2][m1] = dist
        dist_df = pd.DataFrame(dist_dict)
        dist_df = dist_df / dist_df.max().max()
    return dist_df

def get_metabolite_tree_from_classifications(metabolite_classifications):
    print('\nGetting metabolite tree')
    met_classes = metabolite_classifications.T
    query_parent_dict = {}
    it = 0
    for met in met_classes.index.values:
        classification = met_classes.loc[met].dropna()
        it += 1
        for l in np.arange(1, len(classification)):
            if classification.iloc[l - 1].upper() not in query_parent_dict.keys():
                query_parent_dict[classification.iloc[l - 1].upper()] = [
                    classification.iloc[l].upper()]
            else:
                if classification.iloc[l].upper() not in query_parent_dict[
                    classification.iloc[l - 1].upper()]:
                    query_parent_dict[classification.iloc[l - 1].upper()].append(
                        classification.iloc[l].upper())
        if 'COMPOUNDS' not in query_parent_dict.keys():
            query_parent_dict['COMPOUNDS'] = [classification.iloc[0].upper()]
        else:
            if classification.iloc[0].upper() not in query_parent_dict['COMPOUNDS']:
                query_parent_dict['COMPOUNDS'].append(classification.iloc[0].upper())
        if classification.iloc[-1].upper() not in query_parent_dict.keys():
            query_parent_dict[classification.iloc[-1].upper()] = [met]
        else:
            query_parent_dict[classification.iloc[-1].upper()].append(met)

    root = 'COMPOUNDS'
    metabolite_tree = ete4.TreeNode(name=root)
    parents, added = [metabolite_tree], set([root])
    while parents:
        nxt = parents.pop()
        child_nodes = {child: ete4.TreeNode(name=child) for child in query_parent_dict[nxt.name]}
        for child in query_parent_dict[nxt.name]:
            nxt.add_child(child_nodes[child], name=child, dist=1)
            if child not in added:
                if child in query_parent_dict.keys():
                    parents.append(child_nodes[child])
                added.add(child)

    for n in metabolite_tree.traverse():
        if not n.is_leaf():
            n.add_face(ete4.TextFace(n.name + '   '), column=0, position='branch-top')

    return metabolite_tree



