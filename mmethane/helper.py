import random
import torch
from torch.distributions.negative_binomial import NegativeBinomial
import os
import sys
sys.path.append(os.path.abspath(".."))
from utilities.data_utils import *
import shutil

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from typing import Any, Callable, Dict, Optional, Tuple
import lightning.pytorch as pl
from sklearn.cluster import KMeans

class MyEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current, epoch)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria(self, current: Tensor, epoch) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold) and epoch > self.patience:
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}, and epoch {epoch} > {self.patience}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold) and epoch > self.patience:
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

def walk_through_folders(base_dir=None, levels=2):
    dir_levels = 0
    return_folders=[]
    return_files=[]
    for root, dirs, files in os.walk(base_dir):
        dir_levels += 1
        if dir_levels == levels:
            return_folders.extend([os.path.join(root, d) for d in dirs])
            return_files.extend([os.path.join(root, f) for f in files])
            rf = copy.deepcopy(dirs)
            for folder in rf:
                dirs.remove(folder)

            dir_levels=levels-1
    return return_folders, return_files

def delete_res_names(base_dir='/Users/jendawk/logs/mditre-logs/', levels=2):
    folders, files = walk_through_folders(base_dir, levels)
    for folder in folders:
        shutil.move(folder, folder.split('_best')[0].split('_last')[0])

def seed_everything_custom(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def compose(fns):
    def F(x_tr, x_ts):
        for fn in fns:
            x_tr, x_ts = fn(x_tr, x_ts)
        return x_tr, x_ts
    return F

# def get_phylo_dist(seqs, newick_path="./inputs/newick_tree_full.nhx"):
#     # This function saves a distance matrix to out_path based on an input phylogenetic/classification tree file
#     # Inputs:
#     #  - seqs: the sequences or metabolites to get a distance matrix for
#     # - newick_path: the path of the newick tree (seqs needs to match the leaves on the newick tree)
#     # - out_path: the path to save the distance matrix to
#     t = ete3.TreeNode(newick_path)
#     nodes = [n.name for n in t.traverse() if n.is_leaf()]
#     dist = {}
#     for seq1 in seqs:
#         if seq1 not in nodes:
#             continue
#         if seq1 not in dist.keys():
#             dist[seq1] = {}
#
#         for seq2 in seqs:
#
#             if seq2 not in nodes:
#                 continue
#             if seq2 not in dist.keys():
#                 dist[seq2] = {}
#             else:
#                 if seq1 in dist[seq2].keys():
#                     continue
#             if seq1 != seq2:
#                 d = t.get_distance(seq1, seq2)
#                 dist[seq1][seq2] = d
#                 dist[seq2][seq1] = d
#             else:
#                 dist[seq1][seq2] = 0
#     return pd.DataFrame(dist)


# def merge_datasets(dataset_dict):
#     yls = []
#     for key, dataset in dataset_dict.items():
#         yls.append(set(dataset['X'].index.values))
#
#     yixs = set.intersection(*yls)
#     for key, dataset in dataset_dict.items():
#         dataset_dict[key]['y'] = dataset['y'].loc[yixs]
#         dataset_dict[key]['X'] = dataset['X'].loc[yixs]
#
#     y = dataset_dict[key]['y'].values
#     return dataset_dict, y

# @profile
# def split_and_preprocess_dataset(dataset_dict, train_ixs, test_ixs):
#     new_dict = {}
#     test_dict = {}
#     for key, dataset in dataset_dict.items():
#         new_dict[key] = copy.deepcopy(dataset)
#         test_dict[key] = copy.deepcopy(dataset)
#         new_dict[key]['X'] = dataset['X'].iloc[train_ixs,:]
#         new_dict[key]['y'] = dataset['y'].iloc[train_ixs]
#         test_dict[key]['X'] = dataset['X'].iloc[test_ixs, :]
#         test_dict[key]['y'] = dataset['y'].iloc[test_ixs]
#         if 'preprocessing' in dataset.keys() and isinstance(dataset['preprocessing'],dict):
#             print(key)
#             if key == 'metabs':
#                 if isinstance(dataset['preprocessing'], dict) and 'percent_present_in' in dataset['preprocessing'].keys():
#                     ppi=dataset['preprocessing']['percent_present_in']
#                 else:
#                     ppi=25
#                 if isinstance(dataset['preprocessing'], dict) and 'lod' in dataset['preprocessing'].keys():
#                     lod=dataset['preprocessing']['lod']
#                 else:
#                     lod=0
#                 temp_tr, temp_ts = filter_by_presence_func(new_dict[key]['X'], test_dict[key]['X'], ppi, lod)
#                 print(temp_tr.shape)
#                 if isinstance(dataset['preprocessing'], dict) and 'cov_percentile' in dataset['preprocessing'].keys():
#                     cp=dataset['preprocessing']['cov_percentile']
#                 else:
#                     cp=50
#                 temp_tr, temp_ts = filter_by_cov_func(temp_tr, temp_ts, cp)
#                 print(temp_tr.shape)
#                 temp_tr, temp_ts = transform_func(temp_tr, temp_ts)
#                 if 'N-carbamoylaspartate' not in temp_tr.columns.values:
#                     print('ERROR N-carbomylaspartate NOT IN FILTERED DATA')
#                 if 'vanillylmandelate (VMA)' not in temp_tr.columns.values:
#                     print('ERROR vanillylmandelate (VMA) NOT IN FILTERED DATA')
#             elif key == 'otus':
#                 if isinstance(dataset['preprocessing'], dict) and 'percent_present_in' in dataset['preprocessing'].keys():
#                     ppi=dataset['preprocessing']['percent_present_in']
#                 else:
#                     ppi=10
#                 if isinstance(dataset['preprocessing'], dict) and 'lod' in dataset['preprocessing'].keys():
#                     lod=dataset['preprocessing']['lod']
#                 else:
#                     lod=0
#                 temp_tr, temp_ts = filter_by_presence_func(new_dict[key]['X'], test_dict[key]['X'], ppi, lod)
#                 temp_tr = temp_tr.divide(temp_tr.sum(1),axis='index')
#                 temp_ts = temp_ts.divide(temp_ts.sum(1),axis='index')
#
#             new_dict[key]['X'], test_dict[key]['X'] = temp_tr, temp_ts
#             kept_features = new_dict[key]['X'].columns.values
#             if 'distances' in new_dict[key].keys():
#                 kept_sm = [k for k in kept_features if k in new_dict[key]['distances'].columns.values]
#                 new_dict[key]['distances'] = new_dict[key]['distances'][kept_sm].loc[kept_sm]
#                 test_dict[key]['distances'] = test_dict[key]['distances'][kept_sm].loc[kept_sm]
#
#             new_dict[key]['variable_names'] = kept_features
#             test_dict[key]['variable_names'] = kept_features
#
#             print(f'{temp_tr.shape[1]} features in {key} dataset after filtering. Test set has {temp_ts.shape[0]} samples and train set has {temp_tr.shape[0]} samples')
#
#     # if 'otus' in new_dict.keys():
#     #     new_dict['otus']['X'] = np.divide(new_dict['otus']['X'].T, new_dict['otus']['X'].sum(1)).T
#     #     test_dict['otus']['X'] = np.divide(test_dict['otus']['X'].T, test_dict['otus']['X'].sum(1)).T
#
#     return new_dict, test_dict

# def init_w_knn(model, dist_fit):
#     detector_otuids = list()
#     eta_init = np.zeros((model.num_rules, model.num_detectors, model.emb_dim), dtype=np.float32)
#     kappa_init = np.zeros((model.num_rules, model.num_detectors), dtype=np.float32)
#     dist = np.empty((model.num_rules, model.num_detectors, model.num_otus))
#     for i in range(model.num_rules):
#         assigned_otus_det = list()
#         kmeans = KMeans(n_clusters=model.num_detectors, random_state=model.args.seed + i).fit(dist_fit)
#         for j in range(model.num_detectors):
#             assigned_otus = list()
#             eta_init[i, j] = kmeans.cluster_centers_[j]
#             med_dist = list()
#             for k in range(model.dist_matrix.shape[0]):
#                 if kmeans.labels_[k] == j:
#                     med_dist.append(np.linalg.norm(kmeans.cluster_centers_[j] - dist_fit[k], axis=-1))
#                     cur_assig_otu = k
#             if len(med_dist) > 1:
#                 kappa_init[i, j] = np.mean(med_dist)
#             else:
#                 d = model.dist_matrix_embed[cur_assig_otu]
#                 kappa_init[i, j] = min(d[np.nonzero(d)])
#             for k in range(model.dist_matrix.shape[0]):
#                 if kmeans.labels_[k] == j:
#                     try:
#                         dist[i, j, k] = np.linalg.norm(kmeans.cluster_centers_[j] - dist_fit[k], axis=-1)
#                     except:
#                         print('debug')
#                     if dist[i, j, k] <= kappa_init[i, j]:
#                         assigned_otus.append(k)
#             # assigned_otu_names = [model.var_names[k] for k in assigned_otus]
#             assigned_otus_det.append(assigned_otus)
#
#         detector_otuids.append(assigned_otus_det)
#     return kappa_init, eta_init, detector_otuids

# def init_w_gmm(N_feat, N_clust, dim, r_desired=None, mu_met = None, r_met = None):
#     pi_met = st.dirichlet([10/N_clust]*N_clust).rvs().squeeze()
#     if mu_met is None:
#         mu_met = st.multivariate_normal(np.zeros(dim), 0.1*np.eye(dim, dim)).rvs(N_clust)
#     # r_met = st.lognorm(0.01, 1).rvs(N_met_clusters)
#     if r_met is None:
#         temp = st.norm(np.log(r_desired['mean']),np.sqrt(r_desired['var'])).rvs(N_clust)
#         r_met = np.exp(temp)
#     z_id = st.multinomial(1, pi_met).rvs(N_feat)
#     met_locs = np.zeros((N_feat, dim))
#     for clusters in range(N_clust):
#         cluster_id = np.where(z_id[:, clusters] == 1)[0]
#         met_locs[cluster_id, :] = st.multivariate_normal(mu_met[clusters,:], 0.01*r_met[clusters]*np.eye(dim,
#
#                                                                                      dim)).rvs(len(cluster_id))
#     ids = [np.where(z_id[:, i] == 1)[0] for i in range(z_id.shape[1])]
#     return met_locs, mu_met, r_met, ids

def evaluate_distances(dist_mat_df, group):
    base_distances = dist_mat_df[group].loc[group]
    base_distances = base_distances.values.flatten()
    base_distances = base_distances[base_distances!=0]
    avg_base = np.mean(base_distances)
    avg_rand = []
    rand = []
    for i in range(1000):
        random.seed(i)
        dist_shuf = dist_mat_df.copy(deep=True)
        var_names = dist_shuf.index.values
        random.shuffle(var_names)
        dist_shuf.columns = var_names
        dist_shuf.index = var_names
        rand_dists = dist_shuf[group].loc[group]
        rand_dists = rand_dists.values.flatten()
        rand_dists = rand_dists[rand_dists != 0]
        rand.extend(rand_dists)
        avg_rand.append(np.mean(rand_dists.flatten()))

    p_val = len(np.where(avg_rand < avg_base)[0]) / 1000
    fig, ax = plt.subplots()
    base_dists = np.array(base_distances)
    rand = np.array(rand)
    ax.hist(base_dists[base_dists!= 0], bins=20, range=(0, dist_mat_df.values.flatten().max()),
            alpha=0.5, density=True, label='Distances in Group')
    ax.hist(rand[rand!=0], bins=20, range=(0, dist_mat_df.values.flatten().max()),
            alpha=0.5, density=True, label='Permutated distances')
    ax.set_title('p={0}'.format(p_val))
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.hist(avg_rand, bins=50, range=(0, dist_mat_df.values.flatten().max()), label='Avg permeated distances\nfor 1000 permutations')
    ax2.axvline(x=avg_base, label='Avg distance for group', color='orange')
    ax2.legend()
    ax2.set_title('p={0}'.format(p_val))
    return fig, fig2, p_val


# def compute_dist(dist_emb, num_otus):
#     dist = np.zeros((num_otus, num_otus), dtype=np.float32)
#     for i in range(num_otus):
#         for j in range(num_otus):
#             dist[i, j] = np.linalg.norm(dist_emb[i] - dist_emb[j], axis=-1)
#     return dist
#
# def compute_dist_emb_mds(dist_matrix, emb_dim, seed):
#     mds = MDS(n_components=int(emb_dim), random_state=seed,
#               dissimilarity='precomputed')
#     mds_transf = mds.fit_transform(dist_matrix)
#     return mds_transf
#
# import scipy
# def test_d_dimensions(d, dist_matrix, seed):
#     pvals = []
#     for dd in d:
#         emb_dim = int(dd)
#         dist_emb = compute_dist_emb_mds(dist_matrix, emb_dim, seed).astype(
#             np.float32)
#         dist_matrix_embed = compute_dist(dist_emb, dist_matrix.shape[0])
#         ks = scipy.stats.kstest(dist_matrix.reshape(-1),
#                                 dist_matrix_embed.reshape(-1))
#         pvals.append(ks.pvalue)
#         if ks.pvalue >= 0.05:
#             emb_dim = int(dd)
#             return emb_dim, dist_emb, dist_matrix_embed
#
#     emb_dim = np.argmax(pvals)
#     if emb_dim == 0:
#         emb_dim=np.max(d)
#     dist_emb = compute_dist_emb_mds(dist_matrix, emb_dim, seed).astype(
#         np.float32)
#     dist_matrix_embed = compute_dist(dist_emb, dist_matrix.shape[0])
#     return emb_dim, dist_emb, dist_matrix_embed


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# def leave_one_out_cv(data, labels, folds=None, ddtype='week_one'):
#     if ddtype == 'all_data':
#         assert (data.shape[0] > 70)
#         # import pdb; pdb.set_trace()
#         patients = np.array([int(i.split('-')[0]) for i in data.index.values])
#         pdict = {}
#         for i, pt in enumerate(patients):
#             pdict[pt] = labels[i]
#
#         ix_all = []
#         for ii in pdict.keys():
#             pt_test = ii
#             pt_train = list(set(pdict.keys()) - set([ii]))
#             ixtrain = (np.concatenate(
#                 [np.where(patients == j)[0] for j in pt_train]))
#             ixtest = np.where(patients == pt_test)[0]
#             set1 = set([patients[ix] for ix in ixtest])
#             set2 = set([patients[ix] for ix in ixtrain])
#             set1.intersection(set2)
#
#             ix_all.append((ixtrain, ixtest))
#             assert (not set1.intersection(set2))
#
#     else:
#         ix_all = []
#         # CHANGE LINE!
#         for ixs in range(len(labels)):
#             ixtest = [ixs]
#             ixtrain = list(set(range(len(labels))) - set(ixtest))
#             ix_all.append((ixtrain, ixtest))
#
#     if folds is not None:
#         if isinstance(labels[0], str):
#             cl = np.where(labels == 'Non-recurrer')[0]
#             re = np.where(labels == 'Recurrer')[0]
#         else:
#             cl = np.where(labels == 0)[0]
#             re = np.where(labels == 1)[0]
#         random_select_cl = np.random.choice(cl, int(folds / 2))
#         random_select_re = np.random.choice(re, int(folds / 2))
#         random_select = np.append(random_select_cl, random_select_re)
#         ix_all = (np.array(ix_all)[random_select]).tolist()
#     return ix_all
def non_obtuse_angle(x,y):
    return np.arccos(np.abs(np.dot(x,y))/(
        np.linalg.norm(x)*np.linalg.norm(y)))

def linear_anneal(x, y2, y1, x2, x1, c):
    return ((y2 - y1) / (x2 - x1)) * x + c

def linear_annealing_factor(y2,y1,x2,x1):
    return ((y2 - y1) / (x2 - x1))

def exp_anneal(x,start_epoch,end_epoch,start_val,end_val):
    b = np.exp(np.log(end_val-start_val)/(end_epoch-start_epoch))
    return b**x + start_val

def cosine_annealing(epoch, t_min, t_max, max_epochs):
    t_new = t_min + 0.5*(t_max - t_min)*(1 - np.cos(np.pi*(epoch/max_epochs)))
    return t_new

def create_negbin(mean, var, device='cpu'):
    assert var != mean, 'NegBin Variance should not be = Mean!'
    p = float(var - mean) / float(var)
    r = float(mean ** 2) / float(var - mean)
    return NegativeBinomial(torch.tensor(r).to(device), probs=torch.tensor(p).to(device))

def get_one_hot(x,l=None):
    if l is None:
        l = len(np.unique(x))
    if torch.is_tensor(x):
        vec = torch.zeros(l)
    else:
        vec = np.zeros(l)
    vec[x] = 1
    return vec


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def two_clusters(inds):
    randi = []
    erawi = []
    sk = {}
    G = {}
    for n in [1, 2]:
        rand_inertia = []
        for i in range(20):
            unif = np.random.uniform(inds.flatten().min(),
                                     inds.flatten().max(),
                                     (len(inds.flatten()), 1))
            km_rand = KMeans(n_clusters=n, random_state=i)
            km_rand.fit(unif)
            rand_inertia.append(km_rand.inertia_)

        if n == 1:
            sk = np.std(np.log(rand_inertia))
        unif = np.random.uniform(inds.flatten().min(),
                                 inds.flatten().max(),
                                 (len(inds.flatten()), 1))
        km_eraw = KMeans(n_clusters=n)
        km_eraw.fit(inds.flatten().reshape(-1, 1))

        km_rand = KMeans(n_clusters=n, random_state=i)
        km_rand.fit(unif[:, 0:1])
        G[n] = np.log(km_rand.inertia_) - np.log(km_eraw.inertia_)
    sk2 = sk * 1.024695
    if G[1] >= G[2] - sk2:
        return False
    else:
        return True

if __name__=="__main__":
    delete_res_names()