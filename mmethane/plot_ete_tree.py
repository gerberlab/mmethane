
# Original code from: https://gist.github.com/fransua/da703c3d2ba121903c0de5e976838b71
# Minor edits to make it run in Python 3,
# and added label_func argument to control the displayed leaf names.

from itertools import chain

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd
import ete3
import re


def round_sig(x, sig=2):
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def to_coord(x, y, xmin, xmax, ymin, ymax, plt_xmin, plt_ymin, plt_width, plt_height):
    x = (x - xmin) / (xmax - xmin) * plt_width + plt_xmin
    y = (y - ymin) / (ymax - ymin) * plt_height + plt_ymin
    return x, y


def plot_tree(tree, align_names=False, name_offset=None, max_dist=None, font_size=9, axe=None, label_func=None, color_func=None,
              **kwargs):
    """
    Plots a ete3.Tree object using matploltib.

    :param tree: ete Tree object
    :param False align_names: if True names will be aligned vertically
    :param None max_dist: if defined any branch longer than the given value will be 
       reduced by this same value.
    :param None name_offset: offset relative to tips to write leaf_names. In bL scale
    :param 12 font_size: to write text
    :param None axe: a matploltib.Axe object on which the tree will be painted.
    :param label_func: a function to display leaf names, e.g., lambda x: my_names_dict.get(x, x)
    :param kwargs: for tree edge drawing (matplotlib LineCollection) 
    :param 1 ms: marker size for tree nodes (relative to number of nodes)

    :returns: a dictionary of node objects with their coordinates
    """

    if color_func is None:
        color_func = lambda x, **kwargs: x

    if axe is None:
        axe = plt.subplot(111)

    def __draw_edge_nm(c, x):
        h = node_pos[c] # node position of child
        hlinec.append(((x, h), (x + c.dist, h)))
        hlines.append(cstyle)
        return (x + c.dist, h)

    def __draw_edge_md(c, x):
        h = node_pos[c]
        if c in cut_edge:
            offset = max_x / 600.
            hlinec.append(((x, h), (x + c.dist / 2 - offset, h)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2 + offset, h), (x + c.dist, h)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2, h - 0.05), (x + c.dist / 2 - 2 * offset, h + 0.05)))
            hlines.append(cstyle)
            hlinec.append(((x + c.dist / 2 + 2 * offset, h - 0.05), (x + c.dist / 2, h + 0.05)))
            hlines.append(cstyle)
            axe.text(x + c.dist / 2, h - 0.07, '+%g' % max_dist, va='top',
                     ha='center', size=2. * font_size / 3, color=color_func(x))
        else:
            hlinec.append(((x, h), (x + c.dist, h)))
            hlines.append(cstyle)
        return (x + c.dist, h)

    __draw_edge = __draw_edge_nm if max_dist is None else __draw_edge_md


    vlinec = []
    vlines = []
    hlinec = []
    hlines = []
    nodes = []
    nodex = []
    nodey = []
    ali_lines = []

    # to align leaf names
    tree = tree.copy()
    max_x = max(n.get_distance(tree) for n in tree.iter_leaves())

    coords = {}
    node_pos = dict((n2, i) for i, n2 in enumerate(tree.get_leaves()[::-1]))
    node_list = tree.iter_descendants(strategy='postorder')
    node_list = chain(node_list, [tree])

    # reduce branch length
    cut_edge = set()
    if max_dist is not None:
        for n in tree.iter_descendants():
            if n.dist > max_dist:
                n.dist -= max_dist
                cut_edge.add(n)

    if name_offset is None:
        name_offset = max_x / 100.
    # draw tree
    for n in node_list:
        style = n._get_style()
        if len([i for i in n.iter_descendants()]) == len([i for i in tree.iter_descendants()]):
            ldist = n.dist
        x = sum(n2.dist for n2 in n.iter_ancestors()) + n.dist
        if n.is_leaf():
            name = n.name if label_func is None else label_func(n.name, **kwargs)
            y = node_pos[n]
            if align_names:
                axe.text(max_x + name_offset, y, name,
                         va='center', size=font_size, color=color_func(n.name, **kwargs))
                ali_lines.append(((x, y), (max_x + name_offset, y)))
            else:
                axe.text(x + name_offset, y, name,
                         va='center', size=font_size, color=color_func(n.name,**kwargs))
        else:
            y = np.mean([node_pos[n2] for n2 in n.children])
            node_pos[n] = y

            # draw vertical line
            vlinec.append(((x, node_pos[n.children[0]]), (x, node_pos[n.children[-1]])))
            vlines.append(style)

            # draw horizontal lines
            for child in n.children:
                cstyle = child._get_style()
                coords[child] = __draw_edge(child, x)
        nodes.append(style)
        nodex.append(x)
        nodey.append(y)

    # draw root
    # __draw_edge(tree, ldist)

    lstyles = ['-', '--', ':']
    hline_col = LineCollection(hlinec, colors=[l['hz_line_color'] for l in hlines],
                               linestyle=[lstyles[l['hz_line_type']] for l in hlines],
                               linewidth=[(l['hz_line_width'] + 1.) / 2 for l in hlines])
    vline_col = LineCollection(vlinec, colors=[l['vt_line_color'] for l in vlines],
                               linestyle=[lstyles[l['vt_line_type']] for l in vlines],
                               linewidth=[(l['vt_line_width'] + 1.) / 2 for l in vlines])
    ali_line_col = LineCollection(ali_lines, colors='k')

    axe.add_collection(hline_col)
    axe.add_collection(vline_col)
    axe.add_collection(ali_line_col)

    nshapes = dict((('circle', 'o'), ('square', 's'), ('sphere', 'o')))
    shapes = set(n['shape'] for n in nodes)
    for shape in shapes:
        indexes = [i for i, n in enumerate(nodes) if n['shape'] == shape]
        scat = axe.scatter([nodex[i] for i in indexes],
                           [nodey[i] for i in indexes],
                           s=0, marker=nshapes.get(shape, shape))
        scat.set_sizes([(nodes[i]['size']) ** 2 / 2 for i in indexes])
        scat.set_color([nodes[i]['fgcolor'] for i in indexes])
        scat.set_zorder(10)

    # scale line
    xmin, xmax = axe.get_xlim()
    ymin, ymax = axe.get_ylim()
    diffy = ymax - ymin
    dist = round_sig((xmax - xmin) / 5, sig=1)
    ymin -= diffy / 100.
    axe.plot([xmin, xmin + dist], [ymin, ymin], color='k')
    axe.plot([xmin, xmin], [ymin - diffy / 200., ymin + diffy / 200.], color='k')
    axe.plot([xmin + dist, xmin + dist], [ymin - diffy / 200., ymin + diffy / 200.],
             color='k')
    axe.text((xmin + xmin + dist) / 2, ymin - diffy / 200., dist, va='top',
             ha='center', size=font_size)
    axe.set_axis_off()
    return coords, axe

def plot_phylo_tree(dataset, features, out_path=None, fontsize=None, width = None):
    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file
    if 'tree' in dataset.keys():
        t=dataset['tree'].copy()
    else:
        t =dataset['variable_tree'].copy()

    dist = dataset['distances']
    hoods = []

    num = len(features)*3
    if num<15:
        num=15
    if num>30:
        num=30
    for feat in features:
        try:
            neighborhood = dist.loc[feat].sort_values(axis=0, ascending=True).iloc[:num]
        except:
            print('debug')
            import pdb; pdb.set_trace()
        hoods.append(neighborhood)
    hoods = pd.concat(hoods).sort_values(axis=0)
    # hoods = list(set(features))
    all_feats = hoods.index.drop_duplicates().values[:len(features)+num]


    def color_func(x, **kwargs):
        if x in features:
            return 'k'
        else:
            return '0.5'

    def label_func(x, **kwargs):
        return str(dataset['taxonomy'][x].loc['Species']).replace('_',' ')

    if features is not None and len(all_feats) > 0:

        #         keep_nodes.extend(otus_keep_cl)
        t.prune(all_feats, preserve_branch_length=True)

    # if out_path=='':
    if out_path[-1]=='/':
        out_path += 'tree.nw'
    elif out_path.endswith('.pdf'):
        out_path = out_path.replace('.pdf', 'tree.nw')
        # out_path = out_path.replace('')
    # out_path = 'tree.nw'
    t.write(out_path)

    ts = ete3.TreeStyle()
    ts.show_leaf_name = True
    if fontsize is None:
        fontsize=8
    h = (len(all_feats)*fontsize*1.5)/72
    if width is None:
        width = 4
    fig, ax = plt.subplots(figsize=(width, h))
    coords, ax = plot_tree(t, axe=ax, color_func=color_func, label_func=label_func, dataset=dataset, features=features, font_size=fontsize)
    return fig, ax, [n.name for n in t.get_leaves() if n.name in features]