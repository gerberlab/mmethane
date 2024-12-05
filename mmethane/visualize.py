import os
import sys
import argparse
import itertools

sys.path.append(os.path.abspath(".."))
import seaborn as sns
import re
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
import warnings
import pandas as pd
from collections import Counter  # pip install pandas

import matplotlib # pip install matplotlib
from matplotlib.transforms import Bbox
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from plot_ete_tree import *


def axis_off(axes):
    for ax in axes:
        ax.axis('off')


def ticks_off(axes, left=False, labelleft=False, bottom=False, labelbottom=False):
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]
    for ax in axes:
        ax.yaxis.set_tick_params(left=left, labelleft=labelleft)
        ax.xaxis.set_tick_params(bottom=bottom, labelbottom=labelbottom)


def get_num_removed_from_percent_removed(percent_removed, num_0, num_1):
    if isinstance(percent_removed, list):
        n0 = int(num_0 * (percent_removed[0] / 100))
        n1 = int(num_1 * (percent_removed[1] / 100))
    elif isinstance(percent_removed, float) or isinstance(percent_removed, int):
        n0 = int(num_0 * (percent_removed / 100))
        n1 = int(num_1 * (percent_removed / 100))
    else:
        n0, n1 = 0, 0
    return n0, n1


def plot_activation_heatmap(activation, class_labels_per_subject, cmap, fig=None, ax=None, percent_removed=None,
                            class_labels=None, ylabel=None, max_ylabels=None, width=None):
    # class_labels_per_subject: Series or list of 1s and 0s corresponding to class labels
    # lylabel = max([len(yl) for yl in ylabel.split('\n')])
    # if max_ylabels is not None:
    #     if lylabel<max_ylabels:
    # ylabel = ' '*(max_ylabels-lylabel) + ylabel
    Y = class_labels_per_subject
    if isinstance(Y, pd.Series):
        Y = Y.values
    if isinstance(activation, pd.Series):
        activation = activation.value
    if len(activation.shape) == 1:
        activation = np.expand_dims(activation, 0)
    f, N, S = 10, 1, len(Y)
    h = N * f / 72
    w = ((S + 2) / N) * h
    if width is not None:
        w = width
    n0, n1 = get_num_removed_from_percent_removed(percent_removed, sum(Y == 0), sum(Y == 1))
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={'hspace': 0.05, 'wspace': 0.01,
                                                                  'width_ratios': [sum(Y == 0) - n0, sum(Y == 1) - n1,
                                                                                   1.5]})
    ticks_off(ax[0], labelleft=True)
    ticks_off(ax[1])
    l=sns.heatmap(activation[:, Y == 0], ax=ax[0], cbar=False, square=True, cmap=cmap, center=0.5, vmin=0, vmax=1,
                yticklabels=[ylabel])
    r=sns.heatmap(activation[:, Y == 1], ax=ax[1], cbar_ax=ax[2], cbar_kws={'ticks': [0, 1]}, square=True, cmap=cmap,
                center=0.5, vmin=0, vmax=1)
    if class_labels:
        ax[0].set_xlabel(class_labels[0])
        ax[1].set_xlabel(class_labels[1])
    l.set_yticklabels(l.get_yticklabels(), rotation=0, fontsize=10)
    # ax[0].tick_params(axis='y', labelsize=f, labelrotation=90)
    return fig, ax


def plot_data_heatmap(data_0: pd.DataFrame, data_1: pd.DataFrame, threshold, cmap, ylabels=True, class_labels=None,
                      agg=None,
                      percent_removed=None, norm=None, vmin=None, vmax=None, fig=None, ax=None, ylabel_mapper=None,
                      width=None):
    # percent_removed: Remove X% of subjects from each group to make visualization easier
    #   (recommended if the total number of subjects is greater than 100)
    #   If list, remove X[0]% from data_0 and X[1]% from data_1. If float, remove X% from both groups
    # class_labels: label for class 0 and class 1 as list [class 0 label, class 1 label]
    # ylabels: True/False, OR list of feature labels
    if ylabels is True:
        ylabels = data_0.index.values
    if ylabel_mapper is not None:
        ylabels = [ylabel_mapper[y] for y in ylabels]
    cmap.set_bad('w')
    if data_0.shape[0] > 1:
        assert (isinstance(agg, str))
        data_0.loc[' '] = [np.nan] * data_0.shape[1]
        data_1.loc[' '] = [np.nan] * data_1.shape[1]
        ylabels.append(' ')
        if agg.lower() == 'average':
            data_0.loc['Average'] = data_0.mean(0)
            data_1.loc['Average'] = data_1.mean(0)
            if isinstance(ylabels, list):
                ylabels.append('Average')
        elif agg.lower() == 'sum':
            data_0.loc['Average'] = data_0.sum(0)
            data_1.loc['Average'] = data_1.sum(0)
            if isinstance(ylabels, list):
                ylabels.append('Sum')

    n0, n1 = get_num_removed_from_percent_removed(percent_removed, data_0.shape[1], data_1.shape[1])
    data_0 = data_0.iloc[:, :data_0.shape[1] - n0]
    data_1 = data_0.iloc[:, :data_1.shape[1] - n1]
    max_ylabels = max([len(y) for y in ylabels])
    N = data_0.shape[0]
    S = data_0.shape[1] + data_1.shape[1]
    if ax is None:
        N = data_0.shape[0]
        if N>1:
            N+=2
        S = data_0.shape[1] + data_1.shape[1] + max_ylabels
        f = 10
        h = N * f / 72
        w = S*f
        fig, ax = plt.subplots(1, 3, figsize=(w, h),
                               gridspec_kw={
                                   'hspace': 0.001, 'wspace': 0.025,
                                   'width_ratios': [data_0.shape[1], data_1.shape[1], 1.5], }, )
    if vmin is None:
        vmin = min([data_0.min().min(), data_1.min().min()])
    if vmax is None:
        vmax = max([data_0.max().max(), data_1.max().max()])
    ticks_off(ax[0], labelleft=True)
    ticks_off(ax[1])
    # print(ylabels, f'd1 {data_0.shape}, d2 {data_1.shape}\n')
    l=sns.heatmap(data_0.values, norm=norm, ax=ax[0], cbar=False, square=True, cmap=cmap,
                center=threshold, yticklabels=ylabels, vmin=vmin, vmax=vmax)
    r=sns.heatmap(data_1.values, norm=norm, ax=ax[1], cbar_ax=ax[2], cbar_kws={'ticks': [threshold]},
                square=True, cmap=cmap,
                center=threshold, vmin=vmin, vmax=vmax)
    plt.yticks(rotation=90)
    if class_labels:
        ax[0].set_xlabel(class_labels[0])
        ax[1].set_xlabel(class_labels[1])

    bbox = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    fsize_y = (height / data_0.shape[0]) * 72
    # ax[0].tick_params(axis='y', labelsize=fsize_y)
    l.set_yticklabels(l.get_yticklabels(), rotation=0, fontsize=9)
    return fig, ax, max_ylabels


def plot_metabolite_structures(metabolite_names, ikeys):
    if isinstance(ikeys, str):
        ikeys = pd.read_csv(ikeys, index_col=0)
    mnames = [m.split('__')[-1].split(': ')[-1] for m in metabolite_names]
    m_rename = {m: m.split('__')[-1].split(': ')[-1] for m in ikeys.index.values}
    ikeys = ikeys.rename(index=m_rename)
    inchis = ikeys['inchikey'].loc[mnames].drop_duplicates().to_dict()
    ms, names, imgs, drawn = [], [], [], []
    for k, ikey in inchis.items():
        if ikey in drawn:
            continue
        try:
            out = pcp.get_compounds(ikey, namespace='inchikey')[0]
        except:
            warnings.warn(f"Warning: Can't identify {k} from inchikey {ikey}. No metabolite structure drawn")
            continue
        m = Chem.MolFromInchi(out.inchi)
        ms.append(m)
        names.append(k)
        imgs.append(Draw.MolToImage(m))
        drawn.append(ikey)
    fig_ls = []
    if len(imgs) > 0:
        for i in range(len(imgs)):
            fig, ax = plt.subplots(figsize=(3, 3))
            img_arr = np.asarray(imgs[i])
            ax.imshow(img_arr);
            ax.set_title(mnames[i])
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ticks_off(ax)
            fig_ls.append(fig)
    if 'hmdb' in ikeys.columns.values:
        links = []
        hmdbs = ikeys['hmdb'].drop_duplicates().to_dict()
        for m in mnames:
            if m in hmdbs.keys():
                link = f'https://hmdb.ca/metabolites/{hmdbs[m]}'
            else:
                link = ''
            links.append(link)
    else:
        links = ['' for m in mnames]
    return fig_ls, links


def get_info_from_path(path, class_label=1):
    dataset_tr = pd.read_pickle(path + '/train_dataset_used.pkl')
    dataset_ts = pd.read_pickle(path + '/test_dataset_used.pkl')
    rules = pd.read_csv(path + '/EVAL/rules.csv')
    rule_activators = pd.read_pickle(path + '/EVAL/plotting_data/rule_activators.pkl')
    detector_activators = pd.read_pickle(path + '/EVAL/plotting_data/detector_params.pkl')
    k0 = list(detector_activators.keys())[0]
    k1 = list(detector_activators[k0].keys())[0]
    Y = detector_activators[k0][k1]['y']
    ixs = Y.index.values

    if 'metabs' in dataset_tr.keys():
        if Y.shape[0] > dataset_tr['metabs']['X'].shape[0]:
            X_metabs = pd.concat([dataset_tr['metabs']['X'], dataset_ts['metabs']['X']]).loc[ixs]
        else:
            X_metabs = dataset_tr['metabs']['X'].loc[ixs]
    if 'otus' in dataset_tr.keys():
        if Y.shape[0] > dataset_tr['otus']['X'].shape[0]:
            X_otus = pd.concat([dataset_tr['otus']['X'], dataset_ts['otus']['X']]).loc[ixs]
        else:
            X_otus = dataset_tr['otus']['X'].loc[ixs]

    rule_locs = []
    det_params = {}
    rls = []
    rfin=[]
    rtmp = []
    last_detector = False
    rule_counter = Counter(rules.iloc[:, 0])
    ct = 0
    rule_ct = 0
    feats_dict = {}
    for ix in rules.index.values:
        rule = rules.loc[ix].iloc[0]

        rule_int = int(rule.split(' ')[-1])
        if len(rule_locs) == 0:
            rule_locs.append(rule_int)
        elif rule_int != rule_locs[-1]:
            rule_locs.append(rule_int)
        if rule_int not in det_params.keys():
            det_params[rule_int] = {}
        det = rules.loc[ix].iloc[1]
        det_int = int(det.split(' ')[-1])
        features = rules.loc[ix]['features']
        feats_ls = features.split('[')[-1].split(']')[0].split("'")[1::2]

        if 'ASV' in feats_ls[0]:
            feats_dict[(rule, det)] = {f.split(' (')[0]: f.split(')')[0].split('(')[-1] for f in feats_ls}
            feats_key = [f.split(' (')[0] for f in feats_ls]
            data_1 = X_otus[feats_key].loc[Y == 1]
            data_0 = X_otus[feats_key].loc[Y == 0]
            dtype = 'otus'
            feats_ls = [f.split(')')[0].split('(')[-1] for f in feats_ls]
        else:
            feats_dict[(rule, det)] = {f: f.split(': ')[-1].split('__')[-1] for f in feats_ls}
            feats_key = feats_ls
            data_1 = X_metabs[feats_key].loc[Y == 1]
            data_0 = X_metabs[feats_key].loc[Y == 0]
            dtype = 'metabs'

        thresh = rules.loc[ix]['Detector Threshold']
        log_odds = rules.loc[ix]['Rule Log Odds']

        ct += 1
        if rule not in rtmp:
            rule_ct += 1
            rls.append(f"<h1>Rule {rule_ct}</h1>")
            if dtype == 'otus':
                rls.append(
                    f"<h2>IF the aggregated relative abundances of the following taxa is greater than {thresh}</h2>")
            else:
                rls.append(
                    f"<h2>IF the average of the standardized levels of the following metabolites is greater than {thresh}</h2>")
        else:
            if dtype == 'otus':
                rls.append(
                    f"<h2>AND the aggregated relative abundances of the following taxa is greater than {thresh}</h2>")
            else:
                rls.append(
                    f"<h2>AND the average of the standardized levels of the following metabolites is greater than {thresh}</h2>")
        rtmp.append(rule)
        if ct == rule_counter[rule]:
            last_detector = True
            ct = 0
        else:
            last_detector = False

        descript = f"<h3>{', '.join(feats_ls)}</h3>"

        rls.append(descript)
        if last_detector:
            rls.append(f"<h2>THEN Log odds of {class_label} are {log_odds}</h2><br />")
            rfin.append(rls)
            rls=[]
        det_params[rule_int][det_int] = {'feature_names': feats_ls, 'threshold': thresh, 'log_odds': log_odds,
                                         'feature_key': feats_key, 'data_0': data_0.T, 'data_1': data_1.T,
                                         'activators': detector_activators[rule_int][det_int]['activators'],
                                         'dtype': dtype}
    num_detectors = len(rules.index.values)
    num_rules = len(rule_locs)

    return rule_activators, det_params, num_rules, num_detectors, rfin, Y, feats_dict


def embed_figure(fig, cl=''):
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    # Embed the result in the html output.
    # fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    # out = f'data:image/png;base64,{fig_data}'
    out = "<img src='data:image/png;base64,{0}' class='{1}'>".format(fig_data, cl)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default = 'logs//run_franzosa_1130_3/seed_5/')
    parser.add_argument("--outcome_positive_value", type=str, default = "Control")
    parser.add_argument("--outcome_negative_value", type=str, default = "CD,UC")
    args = parser.parse_args()
    path = args.path
    labels_map = {1:args.outcome_positive_value,0:args.outcome_negative_value}
    dataset_tr = pd.read_pickle(path + '/train_dataset_used.pkl')
    rule_activators, det_params, num_rules, num_detectors, written_rule, Y, feats_dict = get_info_from_path(
        path, class_label = args.outcome_positive_value)
    figures_html = []
    cmap_taxa = sns.color_palette('YlGnBu', as_cmap=True)
    cmap_mets = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cmap_act = sns.light_palette('tomato', as_cmap=True)
    rct = 0
    ml = []

    for ir, r in enumerate(det_params.keys()):
        ct = 0
        heights = []
        for d in det_params[r].keys():
            dtmp = det_params[r][d]
            h = dtmp['data_0'].shape[0]
            if h>1:
                h += 2
            heights.append(h)
            heights.append(1)

        heights.append(1)
        num_feats = [len(det_params[r][k]['feature_names']) for k in det_params[r].keys()]
        num_feats = [n+2 if n>1 else n for n in num_feats]
        num_activs = len(det_params[r].keys()) + 1
        num_bw = 3*(num_activs*2 - 2)
        N = sum(num_feats) + num_activs + num_bw

        max_label_len = max(itertools.chain.from_iterable(
            [[len(nm) for nm in det_params[r][k]['feature_names']] for k in det_params[r].keys()]))
        S = len(Y) + 0.5 + 1.5
        f=10
        h_pts = N*f
        w_pts = S*f
        h, w = h_pts/72, w_pts/72

        wspace = 0.5/(len(Y)/2)
        hspace = 3/np.mean(num_feats)
        if hspace<0.2:
            hspace=0.2
        # print(wspace, hspace)
        fig, ax = plt.subplots(len(det_params[r].keys())*2 + 1, 3,
                               width_ratios = [sum(Y==0), sum(Y==1), 1.5],
                               gridspec_kw={'wspace':wspace, 'hspace':hspace},
                               height_ratios = heights,
                               figsize=(w,h), layout='none')

        ix = 0
        im_left = []
        for d in det_params[r].keys():
            # if ct > 0:
            #     figures_html.append('''
            #                 </div>
            # </div>
            # </div>
            # ''')
            metabolite_names = det_params[r][d]['feature_names']
            dtmp = det_params[r][d]
            if det_params[r][d]['dtype'] == 'metabs':
                cmap = cmap_mets
                agg = 'average'
                ikeys = dataset_tr['metabs']['metabolite_IDs']
                metabolite_structures, links = plot_metabolite_structures(metabolite_names, ikeys)
                dpi = metabolite_structures[0].get_dpi()
                images_left = [embed_figure(m) for m in metabolite_structures]
                new_images_left = []
                for im, li in zip(images_left, links):
                    if li!='':
                        new_im = f'<p><a href="{li}"> {im} </a></p>'
                        new_images_left.append(new_im)
                    else:
                        new_im = im
                images_left = new_images_left
            else:
                cmap = cmap_taxa
                agg = 'sum'
                # ht_pts = len(det_params[r][d]['feature_key'])*18
                # ht = ht_pts/72
                # figtree, axtree = plt.subplots(figsize=(4, ht))
                figtree, axtree, taxa_ids = plot_phylo_tree(dataset_tr['otus'], det_params[r][d]['feature_key'],
                                     out_path=f'{path}/rule_{r}_detector_{d}_tree.nw', fontsize=9, width = 4)
                dtmp['data_0'] = dtmp['data_0'].loc[taxa_ids]
                dtmp['data_1'] = dtmp['data_1'].loc[taxa_ids]
                images_left = [embed_figure(figtree, 'tree')]
                dpi = figtree.get_dpi()


            fig, _, max_labels = plot_data_heatmap(dtmp['data_0'], dtmp['data_1'], dtmp['threshold'], cmap=cmap,
                                                        fig = fig, ax = ax[ix,:],
                                                        class_labels=[args.outcome_negative_value, args.outcome_positive_value],
                                                        agg=agg, percent_removed=[0, 0], norm=None,
                                                        ylabel_mapper=feats_dict[(f'Rule {r}', f'Detector {d}')], width=None)
            ix += 1
            fig, _ = plot_activation_heatmap(dtmp['activators'], Y, cmap_act,
                                                     ylabel=f'Detector {ct} Activation',
                                                     class_labels=[args.outcome_negative_value, args.outcome_positive_value],
                                                     max_ylabels=max_labels, width=None,
                                                     fig=fig, ax = ax[ix,:])
            ix += 1
            # images_right = [embed_figure(fig_data), embed_figure(fig_det_act)]
            im_left.extend(images_left)
            # temp = f'''
            # <div class="row">
            # <div class="image-group">
            # {"".join(images_left)}
            # <div class="column">
            # {"".join(images_right)}
            # '''
            # figures_html.append(temp)
            # ct += 1
            ml.append(max_labels)
            ct += 1
        fig_all, _ = plot_activation_heatmap(rule_activators[r], Y, cmap_act, ylabel=f"Rule {rct} Activation",
                                                 class_labels=[args.outcome_negative_value, args.outcome_positive_value],
                                                 max_ylabels = max(ml), fig=fig, ax = ax[ix,:])



        ax_pos_0 = ax[0,0].get_position()
        ax_pos_1 = ax[0,1].get_position()
        for a in ax[:,0]:
            new_box = Bbox([[ax_pos_0.x0, a.get_position().y0],[ax_pos_0.x1, a.get_position().y1]])
            a.set_position(new_box,
                           which='active')
        for ix,a in enumerate(ax[:,1]):
            ax_ref = ax[ix, 0].get_position()
            new_box = Bbox([[ax_pos_1.x0, ax_ref.y0], [ax_pos_1.x1, ax_ref.y1]])
            a.set_position(new_box, which='active')
        for ix,a in enumerate(ax[:,2]):
            ax_ref = ax[ix,0].get_position()
            new_box = Bbox([[a.get_position().x0, ax_ref.y0-0.01], [a.get_position().x1, ax_ref.y1+0.01]])
            a.set_position(new_box,
                           which='active')
        temp = f'''
        <div class="image-group">
        <div class="container-left">
        {"".join(im_left)}
        </div>
        <div class="column">
        <div class="container">
        {"".join(embed_figure(fig_all, 'bordered'))}
        </div>
        </div>
        </div>
        '''

        figures_html.append(temp)
        # figures_html.append('''
        # </div>
        # </div>
        # ''')
        rct += 1

    css_template = '''
    body {
    font-family: arial, helvetica, sans-serif;
    font-size: 12px;
    color: black;
    background-color: white;
    margin: 20px;
    padding: 0;
}

p {
    line-height: 21px;
}

h1 {
    color: #000;
    background-color: white;
    font-size: 3em;
    margin: 0;
    margin-bottom: 7px;
    padding: 4px;
    font-style: normal;
    text-align: center;
    letter-spacing: 0em;
    border-bottom-style: solid;
    border-bottom-width: 0.5em;
    border-bottom-color: #c00;
}
h2 {
    color: #000;
    background-color: white;
    font-size: 2em;
    margin: 0;
    margin-bottom: 7px;
    padding: 4px;
    font-style: normal;
    text-align: left;
    letter-spacing: 0em;
}
h3 {
    color: #000;
    background-color: white;
    font-size: 2em;
    margin: 0;
    margin-bottom: 7px;
    padding: 4px;
    font-style: italic;
    text-align: center;
    letter-spacing: 0em;
}
img {
    display: flex;  
    max-width: none
    height: auto; /* Preserve image aspect ratio */
}
img.bordered {
    border-style: solid;
    border-width: 2px;
    border-color: #ccc;
    overflow-x: scroll;
    # overflow-y: scroll;
}

.img.tree{
    max-width: none
    height: auto; /* Preserve image aspect ratio */
    overflow-x: scroll;
    overflow-y: scroll;
}

li {
    color: #000000;
}
.row {
    display: flex;
    justify-content: left;
    margin-bottom: 20px;
    flex-direction:row
    flex-wrap:nowrap
    overflow-x:scroll
}

.container {
    flex-direction: row;
    flex-wrap: nowrap; /* Prevent wrapping of columns */
    # overflow-x: scroll; /* Horizontal scrolling for the columns container */
}
.container-left {
    width: fit-content;
    justify-content: center;
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap; /* Prevent wrapping of columns */
    # overflow-x: auto; /* Horizontal scrolling for the columns container */
}
.column {
    justify-content: center;
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap; /* Prevent wrapping of columns */
    # overflow-x: scroll; /* Horizontal scrolling for the columns container */
}

.image-group {
    display: flex;
    # overflow-x: scroll;
    flex-wrap: nowrap
}

    '''

    html_template = '''<!DOCTYPE html>
<html lang="en">
  <head>
  <link rel="stylesheet" href="style.css">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Visualization</title>
    <link rel="stylesheet" href="./style.css">
    <link rel="icon" href="./favicon.ico" type="image/x-icon">
  </head>
  <body>
    <main>
        {0}
    </main>
    <script src="index.js"></script>
  </body>
</html>"'''

    fin_out = []
    for rix in range(len(written_rule)):
        fin_out.append("<h1>" + "<br />".join(written_rule[rix]) + figures_html[rix] + "</h1>")
    html_template = html_template.format("".join(fin_out))

    with open(args.path + "/style.css", 'w') as f:
        f.write(css_template)

    with open(args.path + "/visualization.html", 'w') as f:
        f.write(html_template)
