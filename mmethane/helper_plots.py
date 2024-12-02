from math import floor
import seaborn as sns
from utilities.model_helper import *
import os
import sys
import scipy.stats as st
# from plot_results import plot_data_heatmaps, plot_activation_heatmaps

sys.path.append(os.path.abspath(".."))
# from utilities.util import *
from helper import *
import pandas as pd
try:
    import ete4
except:
    pass
import pickle as pkl

# from eval_learned_embeddings import eval_embeddings
import matplotlib
matplotlib.use("Agg")
import matplotlib.style as mplstyle
mplstyle.use('fast')
from sklearn.cluster import KMeans
def save_and_plot_post_training(model, train_ixs, test_ixs, log_dir, plot_traces = False, best_epoch = -1, save_grads=False):
    pd.Series(test_ixs).to_csv(log_dir + '/test_ixs.csv')
    pd.Series(train_ixs).to_csv(log_dir + '/train_ixs.csv')
    print(f'Best epoch: {best_epoch}')
    best_param_dict = {k:v[best_epoch] for k,v in model.logging_dict.items()}
    with open(log_dir + '/best_param_dict.pkl', 'wb') as f:
        pkl.dump(best_param_dict, f)
    with open(log_dir + '/score_dict.pkl', 'wb') as f:
        pkl.dump(model.scores_dict, f)
    with open(log_dir + '/running_loss_dict.pkl', 'wb') as f:
        pkl.dump(model.running_loss_dict, f)
    if save_grads:
        with open(log_dir + '/grad_dict.pkl', 'wb') as f:
            pkl.dump(model.grad_dict, f)

    # if len(list(model.logging_dict.keys())) < 15:
    if plot_traces:
        plot_param_traces(model.logging_dict, log_dir, model.model.order)

    # plot_tracked_metrics(model.scores_dict, log_dir, model.current_epoch,
    #                      val_every_n=1, log=True)
    plot_tracked_metrics(model.scores_dict, log_dir, model.current_epoch,
                         val_every_n=1, log=False)
    # plot_loss(model.running_loss_dict, log_dir, log=True)
    plot_loss(model.running_loss_dict, log_dir, log=False)


def plot_heatmaps(lit_model, dataset, logdir, test_dataset_dict=None):
    dtypes = lit_model.model.order

    if 'fold' in logdir:
        model_ls = [lit_model.train_model, lit_model.val_model]
    else:
        model_ls = [lit_model.model]
    dataset_dict = copy.deepcopy(dataset)
    if len(model_ls)>1:
        for dtype in dtypes:
            dataset_dict[dtype]['X']=pd.concat([dataset[dtype]['X'], test_dataset_dict[dtype]['X']], axis=0)
            dataset_dict[dtype]['y'] = pd.concat([dataset[dtype]['y'], test_dataset_dict[dtype]['y']], axis=0)
    det_acts, rule_acts = [],[]
    for model in model_ls:
        det_acts.append(model.detector_activations.detach())
        rule_acts.append(model.rule_activations.detach())

    det_acts = np.concatenate(det_acts)
    det_inds = lit_model.model.z_d
    rule_acts = np.concatenate(rule_acts)
    rule_inds = lit_model.model.z_r

    # per_subj_log_odds = lit_model.model.log_odds
    # rule_log_odds = lit_model.model.weight

    feat_wts_dict = {}
    feat_wts_ls = []
    thresh_dict={}
    radii_dict={}
    for i,name in enumerate(lit_model.model.module_names):
        feat_wts_dict[name] = lit_model.model.combo_modules[i].wts
        feat_wts_ls.append(lit_model.model.combo_modules[i].wts)
        thresh_dict[name] = lit_model.model.combo_modules[i].thresh
        radii_dict[name] = lit_model.model.combo_modules[i].kappa


    det_params_to_plot={}
    rule_acts_to_plot={}

    # radii_to_plot={}
    # thresh_to_plot={}
    for k,rule_ind in enumerate(rule_inds):
        if rule_ind>0:
            rule_acts_to_plot[k]=rule_acts[:,k]
            # rule_acts_to_plot[k] = rule_acts_to_plot[k].numpy()
            for j,det_ind in enumerate(det_inds[k]):
                if det_ind>0:
                    if k not in det_params_to_plot.keys():
                        det_params_to_plot[k]={}

                    dtype=dtypes[j]
                    # det_params_to_plot[k][j] = det_acts[:,k,j]*det_ind

                    if j>=feat_wts_dict[dtype].shape[1]:
                        jx = j-feat_wts_ls[0].shape[1]
                    else:
                        jx = j
                    # import pdb; pdb.set_trace()
                    feat_wts = feat_wts_dict[dtype][k,jx,:]
                    xdata = dataset_dict[dtype]['X']
                    if dtype=='metabs':
                        feats = [xdata[f]*feat_wts[i].numpy()/np.sum(feat_wts.numpy()) for i,f in
                                 enumerate(xdata.columns.values) if (feat_wts[i]/sum(feat_wts))>0.1]
                    else:
                        feats = [xdata[f]*feat_wts[i].numpy() for i,f in
                                 enumerate(xdata.columns.values) if feat_wts[i]>0.1]
                    if len(feats)<1:
                        feats_df = pd.DataFrame()
                    else:
                        feats_df = pd.concat(feats, axis=1)
                    dacts = det_acts[:,k,j]
                    # sel = [feat_wts[i] for i,f in enumerate(xdata.columns.values) if feat_wts[i]>0.1]
                    det_params_to_plot[k][j] = {'dtype':dtype,'activators':dacts,
                                                'thresh':thresh_dict[dtype][k,jx].detach().item(),
                                                'radius':radii_dict[dtype][k,jx].detach().item(),
                                                'features':feats_df,'y':dataset_dict[dtype]['y']}

    key_list = list(rule_acts_to_plot.keys()).copy()
    for k in key_list:
        if k not in det_params_to_plot.keys():
            del rule_acts_to_plot[k]



    if not os.path.isdir(os.path.join(logdir, 'plotting_data')):
        os.mkdir(os.path.join(logdir, 'plotting_data'))
    with open(os.path.join(logdir, 'plotting_data/rule_indicators.pkl'),'wb') as f:
        pkl.dump(rule_inds.numpy(), f)
    with open(os.path.join(logdir, 'plotting_data/detector_indicators.pkl'),'wb') as f:
        pkl.dump(det_inds.numpy(), f)
    with open(os.path.join(logdir, 'plotting_data/rule_activators.pkl'),'wb') as f:
        pkl.dump(rule_acts_to_plot, f)
    with open(os.path.join(logdir, 'plotting_data/detector_params.pkl'),'wb') as f:
        pkl.dump(det_params_to_plot, f)

    if not os.path.isdir(os.path.join(logdir, 'res_plots')):
        os.mkdir(os.path.join(logdir, 'res_plots'))
    # plot_data_heatmaps(rule_acts_to_plot, det_params_to_plot, os.path.join(logdir, 'res_plots'))
    # plot_activation_heatmaps(rule_acts_to_plot, det_params_to_plot, os.path.join(logdir, 'res_plots'))





def save_input_data(model, dataset_dict, test_dataset_dict, args, outpath):
    for name, dataset in dataset_dict.items():
        fig, ax = plt.subplots()
        X, y = dataset['X'], dataset['y']
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        ax.hist(X[y == 0].flatten(), alpha=0.5, label='-', bins=20)
        ax.hist(X[y == 1].flatten(), alpha=0.5, label='+', bins=20)
        ax.legend()
        fig.savefig(outpath+ '/' + name + '_training_data.png')

    with open(outpath +'/train_dataset_used.pkl', 'wb') as f:
        pkl.dump(dataset_dict, f)
    with open(outpath + '/test_dataset_used.pkl', 'wb') as f:
        pkl.dump(test_dataset_dict, f)

    if hasattr(model, 'class_dict'):
        for type, mod in model.class_dict.items():
            if hasattr(mod, 'dist_emb'):
                with open(outpath  + f'/{type}_emb_locs.pkl', 'wb') as f:
                    pkl.dump(mod.dist_emb, f)

def plot_input_data(dataset_dict, outpath):
    for name, dataset in dataset_dict.items():
        fig, ax = plt.subplots()
        X, y = dataset['X'], dataset['y']
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        ax.hist(X[y == 0].flatten(), alpha=0.5, label='-', bins=20)
        ax.hist(X[y == 1].flatten(), alpha=0.5, label='+', bins=20)
        ax.legend()
        fig.savefig(outpath+ '/' + name + '_training_data.png')

def round_sig(x, sig=2):
    return round(x, sig - int(floor(np.log10(abs(x)))) - 1)


def to_coord(x, y, xmin, xmax, ymin, ymax, plt_xmin, plt_ymin, plt_width, plt_height):
    x = (x - xmin) / (xmax - xmin) * plt_width + plt_xmin
    y = (y - ymin) / (ymax - ymin) * plt_height + plt_ymin
    return x, y


def plot_metab_groups(metabs, df, y_df, dir):
    for m1, m2 in itertools.combinations(metabs, 2):
        corr, pval = st.spearmanr(df[m1], df[m2])
        fig, ax = plt.subplots()

        ax.scatter(df[m1].loc[df.index.values[y_df == 0]], df[m2].loc[df.index.values[y_df == 0]], color='g',
                   label='NR')
        ax.scatter(df[m1].loc[df.index.values[y_df == 1]], df[m2].loc[df.index.values[y_df == 1]], color='r',
                   label='R')
        ax.set_title('correlation={0}'.format(np.round(corr, 3)))
        ax.set_xlabel(m1)
        # ax.set_ylabel(m2)
        ax.set_ylabel(m2)
        ax.legend()
        fig.savefig(dir + '/' + m1.replace('/', '_').replace(' ', '').replace(':', '').replace('.', '')
                    +
                    'VS' + m2.replace('/', '_').replace(' ', '').replace(':', '').replace('.', '') + '.png')
        plt.close(fig)


def plot_metab_groups_in_embedding_space(met_ids=None, dlocs=None, centers=None, rule_detect_ls=None, radii=None,
                                         dir=''):
    # mds = MDS(n_components=2, random_state=seed,
    #           dissimilarity='precomputed')
    # mds_transf = mds.fit_transform(dmat)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(dlocs[:, 0], dlocs[:, 1], marker='o', color='k', alpha=0.2)

    for i in range(len(radii)):
        # ax.text(centers[i][0], centers[i][1], metabs[i])
        # for ii in range(len(mets2[i])):
        #     ax.text(mds_df.loc[mets2[i][ii]].iloc[0], mds_df.loc[mets2[i][ii]].iloc[1], mets2[i][ii])
        p1 = ax.scatter(centers[i][0], centers[i][1], marker='*')
        if met_ids is not None:
            ax.scatter(dlocs[met_ids[i], 0], dlocs[met_ids[i], 1]
                       , color=p1.get_facecolor().squeeze()
                       )
        if rule_detect_ls is not None:
            circle1 = plt.Circle((centers[i][0], centers[i][1]), radii[i],
                                 alpha=0.2,
                                 label='Rule {0} Detector {1}'.format(rule_detect_ls[i][0], rule_detect_ls[i][1]),
                                 color=p1.get_facecolor().squeeze())
        else:
            circle1 = plt.Circle((centers[i][0], centers[i][1]), radii[i],
                                 alpha=0.2,
                                 color=p1.get_facecolor().squeeze())

        ax.add_patch(circle1)

    # ax.set_xlim([-4,4])
    # ax.set_ylim([-4,4])
    ax.legend()
    fig.savefig(dir + 'emb_space.png')
    plt.close(fig)



def remove_duplicate_legends(ax):

    #remove duplicates
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    ax.legend(newHandles, newLabels)

def plot_param_traces(param_dict, dir, order):
    if not os.path.isdir(dir + '/traces/'):
        os.mkdir(dir + '/traces/')
    i = 0
    color = None
    for key, value in param_dict.items():
        try:
            value = value.cpu()
        except:
            pass
        if 'nam' in key: 
            if i<=10:
                tr=True
            else:
                tr = False
            i+=1
            if not tr:
                continue
        # if 'eta' in key or 'emb_to_learn' in key:
        #     trace = np.vstack([v.detach().cpu().numpy().flatten() for v in value])

        # elif 'alpha' in key:
        #     try:
        #         if len(list(set(order))) > 1:
        #             trace= {}
        #
        #             for dt in list(set(order)):
        #                 trace[dt] = np.stack([v.detach().cpu().numpy()[:,np.where(np.array(order)==dt)[0]] for v in value],0)
        #         else:
        #             trace = np.stack([v.detach().cpu().numpy() for v in value],0)
        #     except:
        #         continue
        # else:
        try:
            trace = np.vstack([v.detach().cpu().numpy().flatten() for v in value])
        except:
            trace = np.vstack([v for v in value])

        # if 'alpha' in key and len(list(set(order)))>1:

        # if isinstance(trace, dict):
        #     num_rules = value[0].shape[0]
        #     if color is None:
        #         color = cm.rainbow(np.linspace(0, 1, num_rules))
        #     for k,tr in trace.items():
        #         fig, ax = plt.subplots()
        #         for rule in range(num_rules):
        #             ax.plot(np.arange(tr.shape[0]), tr[:,rule,:], linewidth=0.75, label=f'rule {rule}', color=color[rule])
        #         ax.set_title(key)
        #         ax.set_xlabel('Epochs')
        #         ax.set_ylabel('Value')
        #             # ax.legend()
        #         ax.legend()
        #         remove_duplicate_legends(ax)
        #         fig.savefig(dir + '/traces/' + key.replace('.', '_') + '_' + k + '.png')
        #         plt.close(fig)
        #
        # elif 'alpha' in key:
        #     num_rules = value[0].shape[0]
        #     if color is None:
        #         color = cm.rainbow(np.linspace(0, 1, num_rules))
        #     num_rules = value[0].shape[0]
        #     fig, ax = plt.subplots()
        #     for rule in range(num_rules):
        #         ax.plot(np.arange(trace.shape[1]), trace[:,rule], linewidth=0.75, label=f'rule {rule}', color=color[rule])
        #         ax.set_title(key)
        #         ax.set_xlabel('Epochs')
        #         ax.set_ylabel('Value')
        #     ax.legend()
        #     remove_duplicate_legends(ax)
        #     fig.savefig(dir + '/traces/' + key.replace('.', '_') + '.png')
        #     plt.close(fig)
        #
        # else:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(trace)), trace, linewidth=0.75)
        ax.set_title(key)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        fig.savefig(dir + '/traces/' + key.replace('.', '_') + '.png')
        plt.close(fig)


def plot_tracked_metrics(scores_dict, dir, epoch, val_every_n=1, log = False):
    if not os.path.isdir(dir + '/scores/'):
        os.mkdir(dir + '/scores/')
    for key, value in scores_dict.items():
        print('score value', key)
        try:
            value = value.cpu()
            print('score value thats a cuda', key)
        except:
            value = value
        if 'test' in key:
            continue
        if len(value) > 0:
            fig, ax = plt.subplots()
            try:
                value = np.array([v.detach().cpu().numpy() for v in value]).squeeze()
            except:
                pass
            ax.plot(np.arange(len(value)), value)
            ax.set_title(key)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            if log:
                ax.set_yscale('log')
                fig.savefig(dir + '/scores/' + key.replace(' ', '_') + '_log.png')
            else:
                fig.savefig(dir + '/scores/' + key.replace(' ', '_') + '.png')
            plt.close(fig)


def plot_metab_heatmap(metabs, x, y, rd_tuple, dir):
    metabs = [m for m in metabs if len(m) > 0]
    num_mets = np.sum([len(met) for met in metabs])
    if num_mets > 20:
        num_mets = 10
    elif num_mets < 4:
        num_mets = 4
    elif num_mets < 10:
        num_mets = num_mets / 3
    fig, ax = plt.subplots(2, len(metabs), figsize=(num_mets, 10),
                           gridspec_kw={'width_ratios': [len(met) for met in metabs],
                                        'height_ratios': [np.sum(y.values == 0), np.sum(y.values == 1)]},
                           constrained_layout=False)
    x.index = [v.split('-')[0] for v in x.index.values]
    y_sort = y.sort_values()
    data = x.loc[y_sort.index.values]
    cbaxes = fig.add_axes([.91, .3, .03, .4])
    if len(metabs) == 1:
        mets = metabs[0]
        sns.heatmap(data[mets].iloc[y_sort.values == 0], vmin=np.min(data.values.flatten()),
                    vmax=6, center=0,
                    ax=ax[0], cbar_ax=cbaxes, cmap='vlag')

        sns.heatmap(data[mets].iloc[y_sort.values == 1], vmin=np.min(data.values.flatten()),
                    vmax=6, center=0,
                    ax=ax[1], cbar_ax=cbaxes, cmap="vlag")
        ax[0].set_ylabel('NR')
        ax[1].set_ylabel('R')
        ax[0].set_xlabel('Rule {0} Detector {1}'.format(rd_tuple[0][0], rd_tuple[0][1]))
        ax[0].xaxis.set_label_position('top')
        ax[1].set_xticks(np.arange(len(mets)))
        ax[1].set_xticklabels(mets, fontsize=8)
        ax[0].tick_params(axis=u'both', which=u'both', length=0)
        # ax[0,i].axes.get_xaxis().set_visible(False)
        ax[1].tick_params(axis=u'both', which=u'both', length=0)
        ax[0].set_xticks([])
        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(dir + '/metab_groups.png')
        plt.close(fig)
    else:
        for i, mets in enumerate(metabs):
            sns.heatmap(data[mets].iloc[y_sort.values == 0], vmin=np.min(data.values.flatten()),
                        vmax=6, center=0,
                        ax=ax[0, i], cbar_ax=cbaxes, cmap='vlag')

            sns.heatmap(data[mets].iloc[y_sort.values == 1], vmin=np.min(data.values.flatten()),
                        vmax=6, center=0,
                        ax=ax[1, i], cbar_ax=cbaxes, cmap="vlag")
            if i == 0:
                ax[0, i].set_ylabel('NR')
                ax[1, i].set_ylabel('R')
            ax[0, i].set_xlabel('Rule {0} Detector {1}'.format(rd_tuple[i][0], rd_tuple[i][1]))
            ax[0, i].xaxis.set_label_position('top')
            ax[1, i].set_xticks(np.arange(len(mets)))
            ax[1, i].set_xticklabels(mets, fontsize=8)
            ax[0, i].tick_params(axis=u'both', which=u'both', length=0)
            # ax[0,i].axes.get_xaxis().set_visible(False)
            ax[1, i].tick_params(axis=u'both', which=u'both', length=0)
            ax[0, i].set_xticks([])
            if i != 0:
                ax[0, i].set_yticks([])
                ax[1, i].set_yticks([])
                ax[0, i].set_frame_on(False)
                ax[1, i].set_frame_on(False)
        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(dir + '/metab_groups.png')
        plt.close(fig)

def plot_joint_results(data_dict, labels, train_ix, param_dict, out_path, args, lightning_class, w_maybe_rules=False, best_params=None):
    x_data = []
    dtypes = []
    for name, data in data_dict.items():
        if isinstance(data['X'], pd.DataFrame):
            tmp = data['X'].values
        else:
            tmp = data['X']
        x_data.append(tmp)
        dtypes.append(name)

    data = np.concatenate(x_data, axis=1)

    # out_path = "/".join(ckpt_path.split("/")[:-1])
    # ckpt_file = [f for f in os.listdir(ckpt_path) if '.ckpt' in f]
    # model = lightning_class.load_from_checkpoint(ckpt_file,
    #                                              parser=parser, dataset_dict=data_dict, dir=out_path)

    epoch = -1
    epoch_for_anneal = args.epochs - 1

    if best_params is None:
        try:
            best_params = {k: param_dict[k][epoch].detach().cpu() for k in param_dict.keys()}
        except:
            best_params = {k: param_dict[k][epoch] for k in param_dict.keys()}

    train_data = data
    train_labels = labels[train_ix]

    # test_data = data[test_ix, :]
    # test_labels =labels[test_ix]

    # best_model = model.model

    k_rules = linear_anneal(epoch_for_anneal, args.max_k_bc, args.min_k_bc, args.epochs, 1, args.min_k_bc)
    if args.use_k_1:
        k_rules=1
    rules = binary_concrete(torch.Tensor(best_params['beta']), k_rules, hard=False, use_noise=False)

    k_rules_dict = {}
    k_otu_dict = {}
    if 'metabs' in args.dtype:
        k_rules_dict['metabs'] = linear_anneal(epoch_for_anneal, args.metabs_max_k_bc,
                                               args.metabs_min_k_bc, args.epochs, 1, args.metabs_min_k_bc)
        k_otu_dict['metabs'] = linear_anneal(epoch_for_anneal, args.metabs_max_k_otu, args.metabs_min_k_otu,
                                             args.epochs, 1, args.metabs_min_k_otu)
    if 'otus' in args.dtype:
        k_rules_dict['otus'] = linear_anneal(epoch_for_anneal, args.otus_max_k_bc,
                                             args.otus_min_k_bc, args.epochs, 1, args.otus_min_k_bc)
        k_otu_dict['otus'] = linear_anneal(epoch_for_anneal, args.otus_max_k_otu, args.otus_min_k_otu, args.epochs, 1,
                                           args.otus_min_k_otu)

    if args.use_k_1:
        k_rules_dict={k:1 for k in k_rules_dict.keys()}
    otu_wts_ls = []
    # if args.old == 1:
    #     detector_dict = {}
    # else:
    dtype_ls = lightning_class.model.order
    dls = [dtype_ls[0], dtype_ls[-1]]
    if len(list(set(dls)))==1:
        dls = [dtype_ls[0]]
    detectors = []
    dlims={}
    for dtype in dls:
        dtmp = binary_concrete(torch.Tensor(best_params[dtype + '_alpha']), k_rules, hard=False, use_noise=False)
        ncheck = two_clusters(dtmp)
        if ncheck is True:
            km1 = KMeans(n_clusters=2)
            km1.fit(dtmp.flatten().reshape(-1, 1))
            ia = dtmp.flatten()
            if all(ia[km1.labels_ == 0] <= ia[km1.labels_ == 1].min()):
                inactive = ia[km1.labels_ == 0]
                active = ia[km1.labels_ == 1]
            else:
                inactive = ia[km1.labels_ == 1]
                active = ia[km1.labels_ == 0]
            dl = (inactive.max() + active.min())/2
            fig, ax = plt.subplots();
            ax.hist(inactive, label='inactive', range=(0, 1));
            ax.hist(active, label='active', range=(0, 1))
            fig.legend()
            fig.savefig(out_path + f'/{dtype}_indicators.png')
            plt.close(fig)
        else:
            inds = dtmp.flatten().numpy()
            z = np.abs(st.zscore(dtmp.flatten().numpy()))
            fig, ax = plt.subplots()
            ax.hist(inds)
            fig.savefig(out_path + f'/{dtype}_indicators.png')
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.hist(z)
            fig.savefig(out_path + f'/{dtype}_indicator_zscores.png')
            plt.close(fig)
            if any(z>3):
                max_inds = inds[z>3]
                with open(out_path + f'/{dtype}_indicators.txt','w') as f:
                    f.writelines([str(s) for s in max_inds])
                zz = st.zscore(dtmp.flatten().numpy())
                if any(zz>3):
                    max_inds_f = inds[zz>3]
                    with open(out_path + f'/{dtype}_indicators_pos.txt', 'w') as f:
                        f.writelines([str(s) for s in max_inds_f])
                    min_inds = inds[zz<=3]
                    dl = (max_inds_f.min() + min_inds.max())/2
                else:
                    dl=0.5
            else:
                dl=0.5

            dl=0.5


        dlims[dtype]=dl
        detectors.append(dtmp)
    detectors = torch.cat(detectors, -1)
    # if args.old==1:
    #     new_detectors_dict = detectors_dict
    # if 'alpha_slope' in best_params.keys():
    #     slope_detectors = binary_concrete(torch.Tensor(best_params['alpha_slope']), k_rules, hard=False, use_noise=False)
    #     new_detectors_dict = {'slope':slope_detectors, 'abun': detectors}
    # else:
    #     slope_detectors = None
    #     new_detectors_dict = {'abun': detectors}
    # detector_dict={}
    n_detectors = 0

    otu_wts_dict = {}
    otu_wts_ls = []
    # dtype_ls = lightning_class.model.order
    for dtype in dls:
        # if args.old == 1:
        #     detector_dict[dtype] = binary_concrete(torch.Tensor(best_params[dtype + '_alpha']), k_rules_dict[dtype],
        #                                            hard=False, use_noise=False)
        n_detectors += param_dict[dtype + '_eta'][-1].shape[1]
        # dtype_ls.extend([dtype] * param_dict[dtype + '_eta'][-1].shape[0])
        # detector_dict[dtype] = binary_concrete(torch.Tensor(best_params[dtype+ '_alpha']), k_rules_dict[dtype], hard=False, use_noise=False)

        kappa = best_params[dtype + '_kappa'].unsqueeze(-1)
        if any(kappa.flatten() < 0):
            kappa = kappa.exp()

        dist = (best_params[dtype + '_eta'].reshape(
            best_params[dtype + '_eta'].shape[0], best_params[dtype + '_eta'].shape[1], 1,
            best_params[dtype + '_eta'].shape[-1]) - lightning_class.class_dict[dtype].dist_emb).norm(2,
                                                                                                                dim=-1)

        otu_wts_dict[dtype] = torch.sigmoid((kappa - dist) * k_otu_dict[dtype])
        otu_wts_ls.append(otu_wts_dict[dtype])
        # num_detectors_per_type[dtype]=otu_wts_dict[dtype].shape[1]


    fc_wts = best_params['weight'].view(-1).cpu().numpy()
    fc_bias = best_params['bias'].item()

    active_rules = 0
    rules_dict = {}
    met_list = []
    centers = []
    radii = []
    rule_detect_ls = []
    mets_list_2 = []
    met_ids_list = []

    thresholds = np.zeros((len(rules), int(n_detectors)))
    detector_selectors = np.zeros((len(rules), int(n_detectors)))
    rule_selectors = np.zeros(len(rules))
    # if w_maybe_rules:
    #     if torch.sum(rules)<1:
    #         rules = rules/torch.sum(rules)
    #
    # for rlim in [0.5,0.4,0.3,0.2,0.1]:
    #     num_rules = np.sum(rules.squeeze().numpy()>rlim)
    #     if num_rules >= 2:
    #         break
    # if num_rules==0:
    #     for rlim in [0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]:
    #         num_rules = np.sum(rules.squeeze().numpy()>rlim)
    #         if num_rules >= 2:
    #             break
    rlim = 0.1
    det_ls = []
    for i, r in enumerate(rules):
        if r > rlim:
            if w_maybe_rules and torch.sum(detectors[i,:])<1:
                detectors_norm = detectors[i,:]/torch.sum(detectors[i,:])
            else:
                detectors_norm = detectors[i,:]
            det_ls.append(detectors_norm.numpy())

    if len(det_ls)==0:
        print('No active rules')
        pd.DataFrame([]).to_csv('rules.csv')
        return

    det_total = np.concatenate(det_ls)
    # for dlim in [0.5, 0.4, 0.3, 0.2, 0.1]:
    #     num_dets = np.sum(det_total > dlim)
    #     if num_dets >= 2:
    #         break
    # if num_dets == 0:
    #     for dlim in [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]:
    #         num_dets = np.sum(det_total > dlim)
    #         if num_dets >= 2:
    #             break
    # dlim = 0.1
    for i, r in enumerate(rules):
        mets_in = []
        detectors_in = []
        # if args.old == 1:
        #     detectors = detector_dict[dtype_ls[i]]
        # otu_wts = otu_wts_dict[dtype]
        rule_selectors[i] = r
        if r > rlim:
            active_rules += 1
            active_detectors = 0
            log_odds = fc_wts[i] * r + fc_bias
            # if i>otu_wts_ls[0].shape[0]:
            #     dtype =
            # if args.old == 1:
            # if i >= detectors.shape[0]:
            #     ix = i - detector_dict[dtype_ls[i]].shape[0]
            # else:
            #     ix = i
            #     dls = detectors[ix, :]
            # else:
            #     dls = detectors[i,:]
            # for slope_or_abun,detectors in new_detectors_dict.items():
            if w_maybe_rules and torch.sum(detectors[i,:])<1:
                detectors_norm = detectors[i,:]/torch.sum(detectors[i,:])
            else:
                detectors_norm = detectors[i,:]
            for p, d in enumerate(detectors_norm):
                if p >= otu_wts_ls[0].shape[1]:
                    dtype = dtypes[-1]
                    px = p - otu_wts_ls[0].shape[1]
                else:
                    dtype = dtypes[0]
                    px = p
                dlim = dlims[dtype]

                detector_selectors[i, p] = d
                if d > dlim:
                    active_detectors += 1
                    if dtype=='metabs':
                        sel_ids = [l for l, ot in enumerate(otu_wts_dict[dtype][i, px, :]) if (ot/sum(otu_wts_dict[dtype][i, px, :])) > 0.1]
                        sel_wts = [np.round((ot/sum(otu_wts_dict[dtype][i, px, :])).cpu().numpy(), 3) for ot in otu_wts_dict[dtype][i, px, :] if
                                   (ot/sum(otu_wts_dict[dtype][i, px, :])) > 0.1]
                    else:
                        sel_ids = [l for l, ot in enumerate(otu_wts_dict[dtype][i, px, :]) if
                                   ot > 0.1]
                        sel_wts = [np.round(ot.cpu().numpy(), 3) for ot in otu_wts_dict[dtype][i, px, :] if
                                   ot > 0.1]

                    if len(sel_wts)==0:
                        continue
                    metabs = [lightning_class.class_dict[dtype].var_names[s] for s in sel_ids]

                    met_list.extend(metabs)
                    mets_list_2.append(metabs)
                    met_ids_list.append(sel_ids)
                    centers.append(best_params[dtype + '_eta'][i, px, :].detach().cpu().numpy())
                    radii.append(best_params[dtype + '_kappa'][i, px].detach().cpu().numpy())
                    if len(sel_ids) > 0:
                        rule_detect_ls.append((i, p))
                        # if not os.path.isdir(out_path + '/rule_{0}_detector_{1}/'.format(i,p)):
                        #     os.mkdir(out_path + '/rule_{0}_detector_{1}/'.format(i,p))

                    if len(best_params[dtype + '_thresh'].shape)>=2:
                        thresholds[i, p] = np.round(best_params[dtype + '_thresh'][i, px].item(), 3)
                    else:
                        thresholds[i,p] = np.nan
                    if dtype == 'metabs':
                        ms = []
                        for m in metabs:
                            if m in data_dict['metabs']['distances'].index.values:
                                ms.append(m)
                            else:
                                ms.append('**' + m)
                    elif 'taxonomy' in data_dict[dtype].keys():
                        ms = []
                        for m in metabs:
                            if isinstance(data_dict[dtype]['taxonomy'][m], pd.DataFrame) or isinstance(data_dict[dtype]['taxonomy'][m], pd.Series):
                                taxa = data_dict[dtype]['taxonomy'][m].values
                                # taxa = taxa.split('; ')
                                taxa = [t for t in taxa if t != 'NA']
                                taxa = [t for t in taxa if isinstance(t, str)]
                                # taxa = ' '.join(taxa[-2:])
                                ms.append(m + ' (' + taxa[-1].replace('_',' ') + ')')
                            else:
                                ms.append(m + ' (' + data_dict[dtype]['taxonomy'][m])
                    else:
                        ms = metabs

                    if args.kappa_prior == 'log-normal':
                        kappa = np.round(np.exp(best_params[dtype + '_kappa'][i, px].detach().cpu().numpy()), 3)
                    else:
                        kappa = np.round(best_params[dtype + '_kappa'][i, px].detach().cpu().numpy(), 3)
                    # for k,v in rules_dict.items():
                    #     if set(v['features']) == set(ms):
                    #         i_duplicate = int(k[0].split('Rule_')[-1])
                    #         log_odds = (fc_wts[i]+fc_wts[i_duplicate]) * (r+rules[i_duplicate]) + fc_bias
                    rules_dict[('Rule {0}'.format(i),
                                'Detector {0}'.format(p))] = {'features': ms, 'weights': sel_wts,
                                                              'Detector Threshold': np.round(thresholds[i, p], 3),
                                                              'Detector Radius': kappa,
                                                              'Detect Selector': detector_selectors[i, p],
                                                              'Rule Selector': rule_selectors[i],
                                                              'Rule Log Odds': np.round(log_odds.detach().cpu().numpy(), 3),
                                                              'type':dtype
                                                              }

                    mets_in.append(metabs)
                    detectors_in.append(p)

                    # if len(data.shape)==2:
                    #     fig_empty, ax_empty = plt.subplots()
                    #     if len(sel_ids) > 0:
                    #         if len(sel_ids) > 1:
                    #             _, bins, _ = ax_empty.hist(np.mean(data[:, sel_ids], 1), bins=20)
                    #
                    #             fig, ax = plt.subplots(figsize=(7.5, 6))
                    #             p1 = ax.hist(np.mean(train_data[:, sel_ids], 1)[train_labels == 0].flatten(), bins=bins,
                    #                          label='Outcome 0', alpha=0.5)
                    #             p2 = ax.hist(np.mean(train_data[:, sel_ids], 1)[train_labels == 1].flatten(), bins=bins,
                    #                          label='Outcome 1', alpha=0.5)
                    #             # if test_data.shape[0] == 1:
                    #             #     ax.axvline(x=np.mean(train_data[:, sel_ids]), label='Test participant\nlabel={0}, prob={1}'.format(test_labels, test_pred))
                    #         else:
                    #             _, bins, _ = ax_empty.hist(data[:, sel_ids], bins=20)
                    #
                    #             fig, ax = plt.subplots(figsize=(7.5, 6))
                    #             p1 = ax.hist(train_data[:, sel_ids][train_labels == 0].flatten(), bins=bins,
                    #                          label='Outcome 0', alpha=0.5)
                    #             p2 = ax.hist(train_data[:, sel_ids][train_labels == 1].flatten(), bins=bins,
                    #                          label='Outcome 1', alpha=0.5)
                    #             # if test_data.shape[0] == 1:
                    #             #     ax.axvline(x=train_data[:, sel_ids], label='Test participant\nlabel={0}, prob={1}'.format(test_labels, test_pred))
                    #
                    #         ax.axvline(x=thresholds[i, p],
                    #                    label='Threshold', color='k')
                    #         ax.set_title('rule log odds={0}, r={1}, d={2}'.format(np.round(log_odds.detach().cpu().numpy(), 3),
                    #                                                               np.round(r.detach().cpu().numpy(), 2),
                    #                                                               np.round(d.detach().cpu().numpy(), 2)),
                    #                      fontsize=10)
                    #
                    #         ax.legend()
                    #         fig.tight_layout()
                    #         fig.savefig(out_path + '/rule_{0}_detector_{1}_{2}.png'.format(i, p, slope_or_abun))
                    #         plt.close(fig_empty)
                    #         plt.close(fig)

            if active_detectors == 0:
                print('No active detectors in rule {0}'.format(i))

    if active_rules == 0:
        print('No active rules')
    else:
        print('{0} Active Rules'.format(active_rules))


    if w_maybe_rules:
        pd.DataFrame(rules_dict).T.to_csv(out_path + '/rules_extra.csv')
    else:
        pd.DataFrame(rules_dict).T.to_csv(out_path + '/rules.csv')
    return rules_dict




def plot_joint_results_nn(data_dict, labels, train_ix, param_dict, out_path,args, lightning_class):
    x_data = []
    dtypes = []
    for name, data in data_dict.items():
        x_data.append(data['X'].values)
        dtypes.append(name)

    data = np.concatenate(x_data, axis=1)

    # out_path = "/".join(ckpt_path.split("/")[:-1])
    # ckpt_file = [f for f in os.listdir(ckpt_path) if '.ckpt' in f]
    # model = lightning_class.load_from_checkpoint(ckpt_file,
    #                                              parser=parser, dataset_dict=data_dict, dir=out_path)
    epoch = -1
    epoch_for_anneal = args.epochs - 1
    if args.validate==1:
        ckpt_path = [d for d in os.listdir(out_path) if 'epoch=' in d]
        if len(ckpt_path)==0:
            print('Warning: Displaying results from last epoch')
        else:
            temp = ckpt_path[0]
            temp = temp.split('epoch=')[1].split(' ')[0]
            epoch = int(re.findall(r'\d+', temp)[0])
            epoch_for_anneal = epoch

    # else:
    #     temp = out_path.split('epoch=')[1].split(' ')[0]
    #     epoch = int(re.findall(r'\d+', temp)[0])
    best_params={}
    for k in param_dict.keys():
        try:
            best_params[k] = param_dict[k][epoch].detach().cpu()
        except:
            best_params[k] = param_dict[k][epoch]

    train_data = data
    train_labels = labels[train_ix]

    # test_data = data[test_ix, :]
    # test_labels =labels[test_ix]

    # best_model = model.model

    k_rules = linear_anneal(epoch_for_anneal, args.max_k_bc, args.min_k_bc, args.epochs, 1, args.min_k_bc)
    # rules = binary_concrete(torch.Tensor(best_params['beta']), k_rules, hard=False, use_noise=False)

    k_rules_dict = {}
    k_otu_dict = {}
    if 'metabs' in args.dtype:
        k_rules_dict['metabs'] = linear_anneal(epoch_for_anneal, args.metabs_max_k_bc,
                                               args.metabs_min_k_bc, args.epochs, 1, args.metabs_min_k_bc)
        k_otu_dict['metabs'] = linear_anneal(epoch_for_anneal, args.metabs_max_k_otu, args.metabs_min_k_otu,
                                             args.epochs, 1, args.metabs_min_k_otu)
    if 'otus' in args.dtype:
        k_rules_dict['otus'] = linear_anneal(epoch_for_anneal, args.otus_max_k_bc,
                                             args.otus_min_k_bc, args.epochs, 1, args.otus_min_k_bc)
        k_otu_dict['otus'] = linear_anneal(epoch_for_anneal, args.otus_max_k_otu, args.otus_min_k_otu, args.epochs, 1,
                                           args.otus_min_k_otu)



    otu_wts_ls = []
    detectors = binary_concrete(torch.Tensor(best_params['alpha']), k_rules, hard=args.hard_bc, use_noise=False).squeeze()
    # detector_dict={}
    n_detectors = 0

    otu_wts_dict = {}
    otu_wts_ls = []
    dtype_ls = []
    for dtype in dtypes:

        n_detectors += param_dict[dtype + '_eta'][-1].shape[1]
        dtype_ls.extend([dtype] * param_dict[dtype + '_eta'][-1].shape[0])
        # detector_dict[dtype] = binary_concrete(torch.Tensor(best_params[dtype+ '_alpha']), k_rules_dict[dtype], hard=False, use_noise=False)

        kappa = best_params[dtype + '_kappa'].unsqueeze(-1)
        if any(kappa.flatten() < 0):
            kappa = kappa.exp()

        dist = (best_params[dtype + '_eta'].unsqueeze(1) - lightning_class.class_dict[dtype].dist_emb).norm(2,
                                                                                                                    dim=-1)
        
        otu_wts_dict[dtype] = torch.sigmoid((kappa - dist) * k_otu_dict[dtype])
        otu_wts_ls.append(otu_wts_dict[dtype])
        # num_detectors_per_type[dtype]=otu_wts_dict[dtype].shape[1]


    # fc_wts = best_params['weight'].view(-1).cpu().numpy()
    # fc_bias = best_params['bias'].item()
    # otu_wts = torch.concatenate(otu_wts_ls, 0)

    active_rules = 0
    rules_dict = {}
    met_list = []
    centers = []
    radii = []
    rule_detect_ls = []
    mets_list_2 = []
    met_ids_list = []


    mets_in = []
    detectors_in = []
    active_detectors=0
    pix = -1
    for p, d in enumerate(detectors):
        dtype = dtype_ls[p]
        otu_wts = otu_wts_dict[dtype]
        if dtype == dtype_ls[-1]:
            pix = p - otu_wts_dict[dtype_ls[0]].shape[0]

        # otu_wts = otu_wts_dict[dtype]
        if d > 0.1:
            active_detectors += 1
            sel_ids = [l for l, ot in enumerate(otu_wts[pix, :]) if ot > 0.1]
            sel_wts = [np.round(ot.cpu().numpy(), 3) for ot in otu_wts[pix, :] if ot > 0.1]
            metabs = [lightning_class.class_dict[dtype].var_names[s] for s in sel_ids]

            met_list.extend(metabs)
            mets_list_2.append(metabs)
            met_ids_list.append(sel_ids)
            centers.append(best_params[dtype + '_eta'][pix, :].detach().cpu().numpy())
            radii.append(best_params[dtype + '_kappa'][pix].detach().cpu().numpy())
            if len(sel_ids) > 0:
                rule_detect_ls.append(p)
                # if not os.path.isdir(out_path + '/rule_{0}_detector_{1}/'.format(i,p)):
                #     os.mkdir(out_path + '/rule_{0}_detector_{1}/'.format(i,p))

            if dtype == 'metabs':
                ms = []
                for m in metabs:
                    if m in data_dict['metabs']['distances'].index.values:
                        ms.append(m)
                    else:
                        ms.append('**' + m)
            elif 'taxonomy' in data_dict[dtype].keys():
                ms = []
                for m in metabs:
                    taxa = data_dict[dtype]['taxonomy'][m].values
                    # taxa = taxa.split('; ')
                    taxa = [t for t in taxa if t != 'NA']
                    taxa = [t for t in taxa if isinstance(t, str)]
                    taxa = ' '.join(taxa[-2:])
                    ms.append(m + '(' + taxa + ')')
            else:
                ms = metabs
            rules_dict[('Detector {0}'.format(p))] = {'features': ms, 'weights': sel_wts,
                                                      'Detector Radius': np.round(
                                                          best_params[dtype + '_kappa'][
                                                              pix].detach().cpu().numpy(), 3),
                                                      'Detector Selector': d
                                                      }

            mets_in.append(metabs)
            detectors_in.append(p)
            # fig_empty, ax_empty = plt.subplots()
            # if len(sel_ids) > 0:
            #     if len(sel_ids) > 1:
            #         _, bins, _ = ax_empty.hist(np.mean(data[:, sel_ids], 1), bins=20)
            #
            #         fig, ax = plt.subplots(figsize=(7.5, 6))
            #         p1 = ax.hist(np.mean(train_data[:, sel_ids], 1)[train_labels == 0].flatten(), bins=bins,
            #                      label='Outcome 0', alpha=0.5)
            #         p2 = ax.hist(np.mean(train_data[:, sel_ids], 1)[train_labels == 1].flatten(), bins=bins,
            #                      label='Outcome 1', alpha=0.5)
            #         # if test_data.shape[0] == 1:
            #         #     ax.axvline(x=np.mean(train_data[:, sel_ids]), label='Test participant\nlabel={0}, prob={1}'.format(test_labels, test_pred))
            #     else:
            #         _, bins, _ = ax_empty.hist(data[:, sel_ids], bins=20)
            #
            #         fig, ax = plt.subplots(figsize=(7.5, 6))
            #         p1 = ax.hist(train_data[:, sel_ids][train_labels == 0].flatten(), bins=bins,
            #                      label='Outcome 0', alpha=0.5)
            #         p2 = ax.hist(train_data[:, sel_ids][train_labels == 1].flatten(), bins=bins,
            #                      label='Outcome 1', alpha=0.5)
            #         # if test_data.shape[0] == 1:
            #         #     ax.axvline(x=train_data[:, sel_ids], label='Test participant\nlabel={0}, prob={1}'.format(test_labels, test_pred))
            #     ax.set_title(f'detector {p}')
            #
            #     ax.legend()
            #     fig.tight_layout()
            #     fig.savefig(out_path + '/detector_{0}.png'.format(p))
            #     plt.close(fig_empty)
            #     plt.close(fig)




    if 'rules.csv' not in os.listdir(out_path):
        pd.DataFrame(rules_dict).T.to_csv(out_path + '/rules.csv')
    return rules_dict


def plot_loss(loss_dict, dir, log=False):
    if not os.path.isdir(dir + '/losses/'):
        os.mkdir(dir + '/losses')
    for param in loss_dict.keys():
        if len(loss_dict[param]) < 2:
            continue
        fig, ax = plt.subplots()
        val = np.array([v.detach().cpu().numpy() for v in loss_dict[param]]).squeeze()
        ax.plot(val)
        ax.set_title(param)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        if log:
            ax.set_yscale('log')
        if log:
            fig.savefig(dir + '/losses/' + param + '_log.png')
        else:
            fig.savefig(dir + '/losses/' + param + '.png')
        plt.close(fig)


def plot_metab_tree_for_rule(mets_keep, tree, cf_df, out_path):
    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file
    if isinstance(tree, str):
        t = ete4.TreeNode(tree)
    else:
        t = tree

    cf_df = cf_df.T.drop_duplicates().T

    # mets_cf = {m: pd.Series(cf_df[m].T.drop_duplicates().T) for m in mets_keep}
    super_nodes = [cf_df[m].loc['superclass'].upper() for m in mets_keep]
    keep_nodes = list(set(np.concatenate([cf_df[m].dropna().values for m in mets_keep])))
    keep_nodes = [k.upper() for k in keep_nodes]
    tree_nodes = [n.name for n in t.traverse()]
    keep_nodes = list(set(keep_nodes).intersection(set(tree_nodes)))
    if mets_keep is not None and len(mets_keep) > 0:

        mets = [m.replace('(', '_').replace(')', '_').replace(
            ':', '_').replace(',', '_').replace('[', '_').replace(']', '_').replace(';', '_') for m in mets_keep]
        keep_nodes.extend(mets)
        t.prune(keep_nodes, preserve_branch_length=True)
        for n in t.traverse():
            if n.is_leaf():
                if n.name in mets:
                    n.add_face(ete4.TextFace(n.name, fgcolor='red'), column=0, position='branch-top')
                n.name = ''
            else:
                if len(n.children) > 1 or n.children[0].name in mets or n.name in super_nodes:
                    n.add_face(ete4.TextFace('  ' + n.name + ' ', fgcolor='black'), column=0, position='branch-top')

    ts = ete4.TreeStyle()
    ts.show_leaf_name = True
    t.render(out_path, tree_style=ts)
    plt.close()


def plot_otu_tree_for_rule(otus_keep, tree, cf_df, out_path):
    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file
    if isinstance(tree, str):
        t = ete4.TreeNode(tree)
    else:
        t = tree

    cf_df = cf_df.T.drop_duplicates().T

    # mets_cf = {m: pd.Series(cf_df[m].T.drop_duplicates().T) for m in mets_keep}
    #     super_nodes = [cf_df[m].loc['Family'].upper() for m in mets_keep]
    otus_keep_cl = [m.split('(')[0] for m in otus_keep]
    otus_keep_map = {m.split('(')[0]: m for m in otus_keep}
    #     keep_nodes = list(set(np.concatenate([cf_df[m].dropna().values for m in otus_keep_cl])))
    #     keep_nodes = [k.upper() for k in keep_nodes]
    #     tree_nodes = [n.name for n in t.traverse()]
    #     keep_nodes = list(set(keep_nodes).intersection(set(tree_nodes)))
    if otus_keep is not None and len(otus_keep) > 0:

        #         keep_nodes.extend(otus_keep_cl)
        t.prune(otus_keep_cl, preserve_branch_length=True)
        for n in t.traverse():
            if n.is_leaf():
                if n.name in otus_keep_cl:
                    n.add_face(ete4.TextFace(
                        n.name + ', ' + otus_keep_map[n.name].split('(')[-1].split(' ')[-1].replace('_', ' ').replace(
                            ')', ''), fgcolor='red'), column=0, position='branch-top')
                n.name = ''
    #             else:
    #                 if len(n.children)>1 or n.children[0].name in mets or n.name in super_nodes:
    #                     n.add_face(ete4.TextFace('  ' + n.name + ' ', fgcolor='black'), column = 0, position = 'branch-top')

    ts = ete4.TreeStyle()
    ts.show_leaf_name = True
    t.render(out_path, tree_style=ts)
    plt.close()


def plot_distribution(model, distributions_to_plot, dir):
    # for name, dist in [('kappa_init', model.normal_kappa), ('eta_init', model.normal_emb),
    #                    ('thresh_init', model.uniform_thresh), ('num_alpha_init', model.negbin_det),
    #                    ('alpha_init', model.alpha_bc)]:
    for name, dist in distributions_to_plot.items():
        if name == 'num_alpha_init' or name == 'num_beta_init':
            try:
                init_vals = binary_concrete(torch.Tensor(model.init_args[name.split('num_')[-1] + '_init']),
                                            model.k_dict['k_beta']['min']).detach().cpu().numpy()
            except:
                init_vals = binary_concrete(torch.Tensor(model.init_args['metabs_alpha_init']),
                                            model.k_dict['k_beta']['min']).detach().cpu().numpy()
            if 'num_alpha' in name:
                init_vals = init_vals.sum(-1)
            else:
                init_vals = init_vals.sum()
        else:
            init_vals = model.init_args[name + '_init']
            # if name=='alpha_init' or name=='beta_init':
            #     tmp=binary_concrete(torch.Tensor(init_vals), k=1, hard=False, use_noise=False)
            #     init_vals = tmp.cpu().numpy()

        try:
            if not os.path.isdir(dir + '/priors/'):
                os.mkdir(dir + '/priors/')
        except:
            pass
        vals = dist.sample([1000]).cpu().numpy().flatten()

        # if name=='kappa_init':
        #     vals = np.exp(vals)
        min_val = np.min([np.min(vals), np.min(init_vals.flatten())])
        max_val = np.max([np.max(vals), np.max(init_vals.flatten())])
        fig, ax = plt.subplots()
        try:
            ax.hist(vals, density=True, label='Prior', alpha=0.5, range=(min_val, max_val), bins=15)
        except:
            print('ERROR: NANs IN ' + name)
        for val in init_vals.flatten():
            ax.axvline(x=val)
        # ax.hist(init_vals.flatten(), density=True, label='Initial values', alpha=0.5,
        #         range=(min_val, max_val), bins=20)
        ax.legend()
        mean = np.round(np.mean(vals), 3)
        std = np.round(np.std(vals), 3)
        ax.set_title(name + ', ' + model.dtype + ', mean={0}, stdev={1}'.format(mean, std))
        fig.savefig(dir + '/priors/' + name + '_' + model.dtype + '.png')
        plt.close(fig)


def plot_train_loss(train_loss, dir):
    if isinstance(train_loss, str):
        with open(train_loss, 'rb') as f:
            train_loss = pkl.load(f)
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.set_title('Train Loss')
    ax.set_xlabel('iterations')
    fig.savefig(dir + '/train_loss.png')


def plot_train_f1(train_f1, dir):
    if isinstance(train_f1, str):
        with open(train_f1, 'rb') as f:
            train_f1 = pkl.load(f)
    fig, ax = plt.subplots()
    ax.plot(train_f1)
    ax.set_title('Train F1')
    ax.set_xlabel('iterations')
    fig.savefig(dir + '/train_f1.png')


def plot_barplots(mditre_res_path, lr_res_path=None, rf_res_path=None, fig_title=''):
    mditre = pd.read_csv(mditre_res_path)
    mditre['Model'] = ['MDITRE'] * mditre.shape[0]
    concat_list = [mditre[['Model', 'F1', 'AUC']]]
    if lr_res_path is not None:
        lr = pd.read_csv(lr_res_path).iloc[:-2, :]
        lr['Model'] = ['LR'] * lr.shape[0]
        concat_list.append(lr[['Model', 'F1', 'AUC']])
    if rf_res_path is not None:
        rf = pd.read_csv(rf_res_path).iloc[:-2, :]
        rf['Model'] = ['RF'] * rf.shape[0]
        concat_list.append(rf[['Model', 'F1', 'AUC']])
    tot_df = pd.concat(concat_list)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(fig_title)
    sns.set_style("whitegrid")
    sns.boxplot(x='Model', y='F1', data=tot_df, ax=ax[0])

    # fig2, ax2 = plt.subplots()
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title(fig_title)
    sns.set_style("whitegrid")
    sns.boxplot(x="Model", y="AUC", data=tot_df, ax=ax[1])
    return fig

# if __name__=='__main__':
#     import json

