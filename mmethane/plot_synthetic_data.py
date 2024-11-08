# #!/Users/jendawk/miniconda3/envs/ete_env/bin python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
import sys
try:
    import ete4
except:
    pass
import subprocess
import argparse
import seaborn as sns
from matplotlib.colors import LogNorm
from utilities.data_utils import transform_func
import scipy.stats as st
def plot_syn_vs_real_hists(syn_data, mets_perturbed, y, orig_data,orig_y, fig_path='./semi_syn_figures/'):
    fig_empty, ax_empty = plt.subplots()

    _, bins, _ = ax_empty.hist(syn_data, bins = 20, density=True)
    ax_empty.axis('off')
    fig, ax = plt.subplots(figsize = (7, 8))
    ax.hist(syn_data.to_numpy().flatten(), bins=bins, label='Simulated Data', alpha = 0.5, density=True)
    ax.hist(orig_data.to_numpy().flatten(), bins=bins, label='Real Data', alpha=0.5, density=True)
    if 'otus' in fig_path:
        ax.set_ylim([0,1])
    ax.legend()

    # ax[1].hist(syn_data[sum(mets_perturbed, [])].iloc[y.values == 1].to_numpy().flatten(), bins=bins, label='Simulated Case Data', alpha = 0.5, density=True)
    # ax[1].hist(orig_data.iloc[orig_y.values == 1].to_numpy().flatten(), bins=bins, label='Real Case Data', alpha=0.5, density=True)
    # ax[1].legend()
    fig.savefig(fig_path)
    plt.close(fig)

def dot_plots(syn_data, mets_perturbed, labels, dtype = 'metabs', gd=None):
    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    fig, ax = plt.subplots(1, len(mets_perturbed))
    if len(mets_perturbed)==1:
        ax = [ax]
    for i,mets in enumerate(mets_perturbed):
        if 'metabs' in dtype:
            edata_agg = transform_func(syn_data[mets]).mean(1)
            # edata_agg = syn_data[sum(mets_perturbed, [])].mean(1)

            ylab = 'Metabolites'
            ylab2 = '\nStandardized'
        else:
            edata_agg = syn_data[mets].sum(1)
            ylab = 'Taxa'
            if (edata_agg<=1).all():
                ylab2='\nRelative Abundances'
            else:
                ylab2='\nCounts'

        labels = labels.loc[list(set(labels.index.values))]
        if gd is not None:
            if len(mets_perturbed)==len(gd.perturbed_per_feat):
                iter = i
            elif dtype == 'metabs' and len(mets_perturbed)==1:
                iter = 0
            elif dtype=='otus' and len(mets_perturbed)==1:
                iter=1
            pert_per_feat = gd.perturbed_per_feat[iter]
            ctrl_per_feat = gd.ctrl_per_feat[iter]
            colors = []
            lkeys = []
            for l in labels.index.values:
                if l in pert_per_feat:
                    colors.append('r')
                    lkeys.append("perturbed")
                else:
                    colors.append('k')
                    lkeys.append("un-perturbed")

            ax[i].scatter(labels.values + np.random.rand(len(labels))*0.1, edata_agg.loc[labels.index.values],
                          c=colors,label=lkeys)
        else:
            ax[i].scatter(labels.values + np.random.rand(len(labels))*0.1, edata_agg.loc[labels.index.values])
        ax[i].set_xlim([-0.25,1.25])
        ax[i].set_xticks([0,1],['Case 0', 'Case 1'])
        ax[i].set_ylabel(f'Pertubated {ylab} Group Levels{ylab2}')
        u, p = st.mannwhitneyu(edata_agg.loc[labels == 0], edata_agg.loc[labels == 1])
        ax[i].set_title(f'p-value={np.round(p,3)}')
    if len(ax)==1:
        ax = ax[0]
    fig.tight_layout()
    return (fig, ax), p

def plot_data_hists(syn_data, mets_perturbed, labels, fig_path = './semi_syn_figures/'):
    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    fig_empty, ax_empty = plt.subplots()
    _, bins, _ = ax_empty.hist(syn_data[sum(mets_perturbed, [])], bins = 20)
    ax_empty.axis('off')
    fig, ax = plt.subplots(2, 1, figsize = (7, 15))
    ax[0].hist(syn_data[sum(mets_perturbed, [])].iloc[labels.values == 0].to_numpy().flatten(), bins=bins, label='Control', alpha = 0.5)
    for i,clade in enumerate(mets_perturbed):
        ax[0].hist(syn_data[clade].iloc[labels.values==1].to_numpy().flatten(), bins = bins, label = 'Perturbed Clade {0}'.format(i), alpha = 0.5)
    ax[0].set_title('control subjects versus perturbed subjects\nfor the perturbed features')
    ax[0].legend()

    ax[1].hist(syn_data.iloc[labels.values == 0].to_numpy().flatten(), label='Control', alpha = 0.5)
    ax[1].hist(syn_data.iloc[labels.values==1].to_numpy().flatten(), label = 'Perturbed', alpha = 0.5)
    ax[1].set_title('control subjects versus perturbed subjects\nfor all features')
    ax[1].legend()
    fig.savefig(fig_path + 'semi_syn_data.pdf')
    plt.close(fig_empty)
    plt.close(fig)

def plot_data_hists_new(syn_data, perturbed, labels, fig_path = ''):
    fe, ae = plt.subplots()
    fig, ax = plt.subplots(2,1, figsize=(6.4,4.8*2))

    edata_agg = syn_data[sum(perturbed, [])].sum(1)
    edata = syn_data[sum(perturbed, [])]
    _, bins, _ = ae.hist(edata, bins = 20)
    ae.axis('off')
    ax[0].hist(edata.loc[labels==0].values.flatten(), label='ctrls', alpha=0.5, bins=bins)
    ax[0].hist(edata.loc[labels==1].values.flatten(), label='cases', alpha=0.5, bins=bins)
    ax[0].legend()
    ax[0].set_title('values')

    _, bins, _ = ae.hist(edata_agg, bins=20)
    ae.axis('off')

    ax[1].hist(edata_agg.loc[labels==0].values.flatten(), label='ctrls', alpha=0.5, bins=bins)
    ax[1].hist(edata_agg.loc[labels==1].values.flatten(), label='cases', alpha=0.5, bins=bins)
    ax[1].legend()
    ax[1].set_title('aggregated')
    fig.savefig(fig_path + 'semisyn_data_sums.pdf')
    plt.close(fig)
    plt.close(fe)

def plot_data_hists_agg(syn_data, mets_perturbed, labels, fig_path = './semi_syn_figures/'):
    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    fig_empty, ax_empty = plt.subplots()

    fig, ax = plt.subplots(2, len(mets_perturbed), figsize = (7*len(mets_perturbed), 12))
    # if 'metabs' in fig_path:
    #     plot_data = syn_data[sum(mets_perturbed, [])].iloc[labels.values == 0].mean(1)
    #     edata = syn_data[sum(mets_perturbed, [])].mean(1)
    # else:
    #     plot_data = syn_data[sum(mets_perturbed, [])].iloc[labels.values == 0].sum(1)
    if 'metabs' in fig_path:
        edata_agg = syn_data[sum(mets_perturbed, [])].mean(1)
    else:
        edata_agg = syn_data[sum(mets_perturbed, [])].sum(1)
    edata = syn_data[sum(mets_perturbed, [])]
    _, bins, _ = ax_empty.hist(edata, bins = 20)
    _, bins_agg, _ = ax_empty.hist(edata_agg, bins=20)
    ax_empty.axis('off')

    for i,clade in enumerate(mets_perturbed):
        if 'metabs' in fig_path:
            case_subs_agg = syn_data[clade].iloc[labels.values == 1].mean(1)
            ctrl_subs_agg = syn_data[clade].iloc[labels.values == 0].mean(1)
            case_subs = syn_data[clade].iloc[labels.values == 1]
            ctrl_subs = syn_data[clade].iloc[labels.values == 0]
        else:
            case_subs_agg = syn_data[clade].iloc[labels.values == 1].sum(1)
            ctrl_subs_agg = syn_data[clade].iloc[labels.values == 0].sum(1)
            case_subs = syn_data[clade].iloc[labels.values == 1]
            ctrl_subs = syn_data[clade].iloc[labels.values == 0]
        if len(mets_perturbed)==1:
            ax[0].hist(case_subs_agg.to_numpy(), bins=bins_agg, label='cases', alpha=0.5)
            ax[0].hist(ctrl_subs_agg.to_numpy(), bins=bins_agg, label='ctrls', alpha=0.5)
            ax[0].set_title('Aggregated/Summed values of perturbed clade {0}'.format(i))
            ax[1].hist(case_subs.to_numpy().flatten(), bins=bins, label='cases', alpha=0.5)
            ax[1].hist(ctrl_subs.to_numpy().flatten(), bins=bins, label='ctrls', alpha=0.5)
            ax[1].set_title('Values of perturbed clade {0}'.format(i))
            ax[0].legend()
            ax[1].legend()
            # if 'metabs' not in fig_path:
                # ax[0].set_xscale('log')
                # ax[1].set_xscale('log')
                # ax[1].set_xlabel('log values')
        else:
            ax[0,i].hist(case_subs_agg.to_numpy(), bins = bins_agg, label = 'cases', alpha = 0.5)
            ax[0,i].hist(ctrl_subs_agg.to_numpy(), bins=bins_agg, label='ctrls', alpha=0.5)
            ax[0,i].set_title('Aggregated/Summed values of perturbed clade {0}'.format(i))
            ax[1,i].hist(case_subs.to_numpy().flatten(), bins = bins, label = 'cases', alpha = 0.5)
            ax[1,i].hist(ctrl_subs.to_numpy().flatten(), bins=bins, label='ctrls', alpha=0.5)
            ax[1,i].set_title('Values of perturbed clade {0}'.format(i))
            ax[0,i].legend()
            ax[1,i].legend()
            # if 'metabs' not in fig_path:
            #     # ax[0,i].set_xscale('log')
            #     # ax[1,i].set_xscale('log')
            #     ax[1,i].set_xlabel('log values')
    # ax.set_title('33 control subjects versus 33 perturbed subjects\nfor the perturbed metabolites')
    # ax.legend()

    if 'metabs' in fig_path:
        fig.savefig(fig_path + 'semi_syn_data_clade_w_avgs.pdf')
    else:
        fig.savefig(fig_path + 'semi_syn_data_clade_w_sums.pdf')
    plt.close(fig_empty)
    plt.close(fig)

def plot_dist_hists(syn_data, mets_perturbed, dmat, fig_path = './semi_syn_figures/'):
    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    # fig, ax = plt.subplots(2, 1, figsize = (7, 15))
    dmat_df = pd.DataFrame(dmat, index=syn_data.columns.values, columns=syn_data.columns.values)
    fig_empty, ax_empty = plt.subplots()
    try:
        _, bins, _ = ax_empty.hist(dmat.flatten(), bins=20)
    except:
        _, bins, _ = ax_empty.hist(dmat.values.flatten(), bins=20)
    # un_perturbed = list(set(syn_data.columns.values) - set(mets_perturbed))
    # ax[0].hist(dmat_df[mets_perturbed].loc[mets_perturbed].to_numpy().flatten(), bins=bins, alpha=0.5,
    #         label='between perturbed metabolites')
    # ax[0].set_title('Histogram of embedded distances')
    # ax[0].hist(dmat_df[mets_perturbed].loc[un_perturbed].to_numpy().flatten(), bins=bins, alpha=0.5,
    #         label='between perturbed and\nnon-perturbed metabolites')
    # ax[0].legend()
    # # fig.savefig('semi_syn_data_dists.pdf')
    # plt.close(fig_empty)
    # # plt.close(fig)

    fig, ax = plt.subplots(len(mets_perturbed), 1, figsize = (7, 7*len(mets_perturbed)))
    if len(mets_perturbed) == 1:
        clade = mets_perturbed[0]
        un_perturbed = list(set(syn_data.columns.values) - set(clade))
        ax.hist(dmat_df[clade[0]].loc[clade[1:]].to_numpy().flatten(), bins=bins, alpha=0.5,
                label='between starting metabolite\nand perturbed metabolites')
        ax.set_title('Histogram of embedded distances, Clade {0}'.format(0))
        ax.hist(dmat_df[clade[0]].loc[un_perturbed].to_numpy().flatten(), bins=bins, alpha=0.5,
                label='between starting metabolite and\nnon-perturbed metabolites')
        ax.legend()
        fig.savefig(fig_path + 'semi_syn_data_dists.pdf')
        plt.close(fig)
    else:
        for i, clade in enumerate(mets_perturbed):
            un_perturbed = list(set(syn_data.columns.values) - set(clade))
            ax[i].hist(dmat_df[clade[0]].loc[clade[1:]].to_numpy().flatten(), bins=bins, alpha=0.5,
                    label='between starting metabolite\nand perturbed metabolites')
            ax[i].set_title('Histogram of embedded distances, Clade {0}'.format(i))
            ax[i].hist(dmat_df[clade[0]].loc[un_perturbed].to_numpy().flatten(), bins=bins, alpha=0.5,
                    label='between starting metabolite and\nnon-perturbed metabolites')
            ax[i].legend()
            fig.savefig(fig_path + 'semi_syn_data_dists.pdf')
            plt.close(fig)
    plt.close(fig_empty)

def plot_heatmap(data_ls, labels, fig_path = './semi_syn_figures/'):
    keys = list(data_ls.keys())
    if len(data_ls.keys())>1:
        fig, ax = plt.subplots(1,2, figsize=(data_ls[keys[0]].shape[1]/2 + data_ls[keys[1]].shape[1]/2, data_ls[keys[0]].shape[0]/6 + data_ls[keys[1]].shape[0]/6))
        # print((data_ls[keys[0]].shape[1]/6 + data_ls[keys[1]].shape[1]/6, data_ls[keys[0]].shape[0]/6 + data_ls[keys[1]].shape[0]/6))
        print('')
    else:
        fig, ax = plt.subplots(figsize=(data_ls[keys[0]].shape[1]/4, data_ls[keys[0]].shape[0] / 6))
        ax = [ax]
    # labels = [int(y) for y in labels]
    ixs = np.argsort(labels)
    labs = labels.iloc[ixs]
    sns.set(font_scale=.3)
    for i,(dtype, data) in enumerate(data_ls.items()):
        if dtype=='metabs':
            cmap = sns.color_palette("vlag", as_cmap=True)
            sns.heatmap(data.iloc[ixs,:], ax=ax[i], yticklabels=[str(labs.index.values[ii]) + ','+str(int(labs.iloc[ii])) for ii in range(len(labs))],
                        xticklabels = data.columns.values, center=0, cmap=cmap, square=True)
        else:
            cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
            sns.heatmap(data.iloc[ixs,:], ax=ax[i], yticklabels=[str(labs.index.values[ii]) + ','+str(labs.iloc[ii]) for ii in range(len(labs))],
                        xticklabels = data.columns.values, cmap=cmap, square=True, norm=LogNorm())

        ax[i].set_xlabel(dtype)
        ax[i].set_ylabel('Subjects')

        fig.tight_layout()
        fig.savefig(fig_path + 'heatmap.pdf')
        plt.close(fig)

def recurse_parents(node, parents = []):
    if node.up.name != '':
        parents.append(node.up.name)
        parents = recurse_parents(node.up, parents)
        return parents
    else:
        return parents

def plot_metab_tree_2(mets_keep, newick_in_path='/inputs/newick_met.nhx',
                    fig_path='/semi_syn_figures/', figname = 'met_tree.pdf'):

    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file

    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    t = ete4.TreeNode(newick_in_path)

    if mets_keep is not None and len(mets_keep) > 0:

        colors = ete4.random_color(num = len(mets_keep))
        mets = [m.replace('(', '_').replace(')', '_').
                   replace(':','_').replace(',','_').replace('[','_').
                   replace(']','_').replace(';','_') for m in mets_keep]
        leaf_nodes = [n for n in t.traverse() if n.name in mets]
        parents = []
        for node in leaf_nodes:
            parents = recurse_parents(node, parents = parents)
            # print(parents)
        all_parents = np.unique(parents)
        colors2 = ete4.random_color(num=2)
        for n in t.traverse():
            # if n in leaf_nodes:
            #     print(n.name)
            if n.name in all_parents:
                n.add_face(ete4.TextFace(n.name, fgcolor=colors[1]), column = 0, position = 'branch-top')
                n.name = ''
            elif n in leaf_nodes:
                n.add_face(ete4.TextFace(n.name, fgcolor=colors2[0]), column = 0, position = 'branch-top')
            elif n not in leaf_nodes and n.name not in all_parents:
                n.name = ''

    ts = ete4.TreeStyle()
    ts.show_leaf_name = False
    t.render(fig_path + '/' + figname, tree_style = ts)
    plt.close()

# def plot_taxonomic_tree(otus_keep, newick_in_path,fig_path,figname):
#     t = ete4.TreeNode(newick_in_path)

def plot_metab_tree(mets_keep, newick_in_path='/inputs/newick_met.nhx',
                    fig_path='/semi_syn_figures/', taxonomy = None):

    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file

    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    t = ete4.Tree(newick_in_path)

    if mets_keep is not None and len(mets_keep) > 0:
        if not isinstance(mets_keep[0], list):
            mets_keep = [mets_keep]

        colors = ete4.random_color(num = len(mets_keep))
        for i, m_ls in enumerate(mets_keep):
            mets = [m.replace('(', '_').replace(')', '_').
                       replace(':','_').replace(',','_').replace('[','_').
                       replace(']','_').replace(';','_') for m in m_ls]

            for n in t.traverse():
                if n.is_leaf():
                    if taxonomy is not None:
                        tax = taxonomy[n.name].dropna().values
                        if len(tax)>6:
                            n.name = n.name + ', ' + ' '.join(tax[-2:])
                        else:
                            n.name = n.name + ', ' + tax[-1]

                    if n.name in mets:
                        n.add_face(ete4.TextFace(n.name, fgcolor=colors[i]), column = 0, position = 'branch-top')
                        n.name = ''

    ts = ete4.TreeStyle()
    ts.show_leaf_name = True
    t.render(fig_path, tree_style = ts)
    plt.close()

def make_metab_tree(in_mets = None, data_path ='./inputs/',
                         newick_out_path = './inputs/newick_met.nhx'):

    # Constructs the metabolomic tree given the input metabolites and the classy-fire classifications
    # inputs:
    # - newick_path: path to save the newick tree to
    # - out_path: path to save the tree plot
    # - data_path: path for the classy fire data
    # - in_mets: which metabolites to use in making the tree
    # - name: name of the tree plot file
    # - dist_type: How to construct the distance between branches of the tree; options:
    #           '' (empty string): set all branch distances = 1
    #           'clumps': set branch distance = 1 if within the same level 5 classification, or distance = 100 if not
    #           'stratified': increase branch distance exponentially with increasing classification levels

    met_classes = pd.read_csv(data_path + 'classy_fire_df.csv', index_col = 0, header = 0).T

    # query_child_dict = {}
    query_parent_dict = {}
    # query_parent_dict['COMPOUNDS'] = {}
    weights_dict = {}
    it = 0
    for met in met_classes.index.values:
        classification = met_classes.loc[met].dropna()
        if in_mets is not None:
            if met not in in_mets:
                continue
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

    for n in query_root.traverse():
        if not n.is_leaf():
            n.add_face(ete4.TextFace(n.name + '   '), column = 0, position = 'branch-top')

    query_root.write(features=['name'], outfile=newick_out_path, format=0)

if __name__ == "__main__":

    with open('./datasets/semi_synthetic/semi_syn_case_1.pkl','rb') as f:
        dataset = pkl.load(f)
        # plot_metab_tree(mets_perturbed, newick_in_path='./inputs/case_1_newick_met.nhx',
        #                 fig_path='./semi_syn_figures/case_1/')
    plot_metab_tree_2(dataset['met_ids'][0], newick_in_path='inputs/case_1_newick_met.nhx',
                      fig_path='./semi_syn_figures/case_1/test')