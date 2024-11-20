import pandas as pd
import itertools
import numpy as np
import copy
from .data_utils import filter_by_presence_func, filter_by_cov_func, transform_func, get_epsilon
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.cluster import KMeans
import scipy.stats as st
import scipy
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from .model_helper import TruncatedNormal
import sklearn
import os

def recurse_parents(p):
    if len(p.get_leaves())>=2:
        return p
    else:
        recurse_parents(p.up)

def get_init_options_clades(dist_emb, tree, seed):
    nparents = []
    tree.prune(dist_emb.index.values)
    parents = list(set([l.up for l in tree.iter_leaves()]))
    for p in parents:
        newp = recurse_parents(p)
        nparents.append(newp)

    centroids, radii, pts = [], [], []
    for i, p in enumerate(nparents):
        tmp = p.get_leaves()
        cnames = [c.name for c in tmp if c.name != '']
        emb_locs = dist_emb.loc[cnames]
        centroid = np.mean(emb_locs, 0)
        radius = np.max(np.linalg.norm(emb_locs - centroid, axis=1))
        if radius < 0.01:
            radius = 0.01
        in_ix = np.where(np.linalg.norm(dist_emb - centroid, axis=1) <= radius)[0]
        if len(in_ix) > 35:
            print(f'{len(in_ix)} in INITIAL CLUSTER')
            nnew = int(np.floor(len(in_ix)/10))
            km = KMeans(n_clusters=nnew, random_state=seed).fit(dist_emb.values[in_ix, :])
            for i in range(nnew):
                cc = km.cluster_centers_[i]
                ii = in_ix[km.labels_ == i]
                rr = np.mean(np.linalg.norm(dist_emb.values[ii] - cc, axis=1))
                if rr<0.01:
                    rr=0.01
                centroids.append(cc)
                radii.append(rr)
                pts.append(ii.tolist())
        else:
            centroids.append(centroid)
            radii.append(radius)
            pts.append(in_ix.tolist())
    return centroids, radii, pts

def get_init_options_coresets(pts, rmax):
    d = squareform(pdist(pts))
    d[d == 0] = 99
    ix = np.arange(pts.shape[0])
    final_centers = []
    final_radii = []
    final_pts = []
    ls = ix.tolist()
    used_ixs = []
    while len(ls) > 0:
        pts_in_combo = []
        num_in_combo = []
        c_combo = []
        r_combo = []
        for i0, i1, i2 in itertools.combinations(ls, 3):
            try:
                p = pts[[i0, i1, i2], :]
            except:
                print('debug')
            c = np.mean(p, 0)
            dists = np.linalg.norm(pts - c, axis=1)

            ix_where = np.where(dists < rmax)[0]
            ix_where = list(set(ix_where) - set(used_ixs))
            if len(ix_where) > 0:
                num = len(ix_where)
                pts_in_combo.append(ix_where)
                num_in_combo.append(num)
                c_combo.append(c)
                r_combo.append(np.max(dists[ix_where]))
        if len(num_in_combo) == 0:
            break
        best_ix = np.argmax(num_in_combo)
        center = c_combo[best_ix]
        radius = r_combo[best_ix]
        cont_pts = pts_in_combo[best_ix]
        used_ixs.extend(cont_pts)
        for c in cont_pts:
            ls.remove(c)
        final_centers.append(center)
        final_radii.append(radius)
        final_pts.append(cont_pts)
    if len(ls) > 0:
        for p in ls:
            closest_ix = np.argmin(d[p, :])
            closest = pts[closest_ix, :]
            center = np.mean(pts[[closest_ix, p], :], 0)
            radius = np.max(np.linalg.norm(pts[[closest_ix, p], :] - center, axis=1))
            final_centers.append(center)
            final_radii.append(radius)
            final_pts.append([closest_ix, p])
    return final_centers, final_radii, final_pts

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def calculate_radii_prior(dataset_dict, dist_mat_emb, dtype, multiplier=1, key='Family'):
    """
    calculates the mean and variance of the prior distribution for metabolite/microbial radii.
    Mean is calculated as the median of medians of family/subclass level distances, and variance is the variance
    of the medians of family/subclass level distances.
    Inputs:
    - dataset_dict: output dataset dictionary from process_data.py (must contain keys "taxonomy" and "distances")
    - dtype: whether the data is metabolomics or sequences (options=['metabs','otus'])
    - multiplier: multiplier for variance, defaults to 1

    Outputs:
    - dictionary with keys "mean" and "variance", for the mean and variance of the truncated normal prior (or the exponentiated mean and variance for a lognormal prior)
    """
    if 'taxonomy' in dataset_dict.keys():
        if dtype=='metabs':
            if key=='Family':
                key='subclass'
            taxonomy=dataset_dict['taxonomy'].loc[key]
        elif dtype=='otus':
            taxonomy=dataset_dict['taxonomy'].loc[key]
        cats = taxonomy.unique()
        dists = []
        tracker={}
        for cat in cats:
            mets = taxonomy.index.values[taxonomy == cat]
            dvec = []
            for a, b in itertools.combinations(mets, 2):
                if a in dist_mat_emb.index.values and b in dist_mat_emb.index.values:
                    d = dist_mat_emb[a].loc[b]
                    if d==0 and a.split(': ')[-1].split('__')[-1]==b.split(': ')[-1].split('__')[-1]:
                        continue
                    elif d != np.nan:
                        dvec.append(d)

            if len(dvec)>0:
                tracker[cat]=dvec
            dists.append(dvec)

        ref_median = {'mean': np.median([np.median(d) for d in dists if len(d) > 1]),
                           'var': multiplier * np.var([np.median(d) for d in dists if len(d) > 1])}
        # self.ref_median = {'mean': np.median(dists)/self.args.divider, 'var': self.args.multiplier}
    else:
        # ref_median = {'metabs': {'mean': 0.25, 'var': self.args.multiplier},
        #               'otus': {'mean': 0.236, 'var': self.args.multiplier}}
        ref_median = {'metabs': {'mean': 0.25, 'var': multiplier * 0.03317107101445189},
                      'otus': {'mean': 0.236, 'var': multiplier * 0.079}}
        ref_median = ref_median[dtype]
    return ref_median

def preprocess_data(dataset, key):
    print(f'Filtering and processing {key}')
    temp_tr=dataset['X']
    new_dict=copy.deepcopy(dataset)
    if 'preprocessing' in dataset.keys() and dataset['preprocessing']:
        if isinstance(dataset['preprocessing'], dict) and 'percent_present_in' in dataset['preprocessing'].keys():
            ppi = dataset['preprocessing']['percent_present_in']
            if isinstance(dataset['preprocessing'], dict) and 'lod' in dataset['preprocessing'].keys():
                lod = dataset['preprocessing']['lod']
            else:
                lod = 0
            temp_tr = filter_by_presence_func(temp_tr, None, ppi, lod)
            print('After filtering {3} to keep only {3} with levels > {0} in {1}\% of participants, '
                  '{2} remain'.format(lod, ppi, temp_tr.shape[1], key))
        # print(temp_tr.shape)
        if isinstance(dataset['preprocessing'], dict) and 'cov_percentile' in dataset['preprocessing'].keys():
            cp = dataset['preprocessing']['cov_percentile']
            temp_tr = filter_by_cov_func(temp_tr, None, cp)
            print('After filtering {2} to keep only {2} with coefficients of variation in the '
                  'top {0} percentile of participants, {1} {2} remain'.format(
                cp, temp_tr.shape[1], key))
        # print(temp_tr.shape)
        if key == 'metabs':
            temp_tr = transform_func(temp_tr, None)

        elif key == 'otus':
            temp_tr = temp_tr.divide(temp_tr.sum(1), axis='index')

        new_dict['X'] = temp_tr
        kept_features = new_dict['X'].columns.values
        if 'distances' in new_dict.keys():
            kept_sm = [k for k in kept_features if k in new_dict['distances'].columns.values]
            new_dict['distances'] = new_dict['distances'][kept_sm].loc[kept_sm]

        new_dict['variable_names'] = kept_features

    print(
        f'{temp_tr.shape[1]} features in {key} dataset after filtering and {temp_tr.shape[0]} samples')
    return new_dict

def CLR(X):
    # X can be matrix of counts or relative abundances
    # X should be a matrix with rows of samples and columns of OTUs/ASVs
    indices, columns = None, None
    if isinstance(X, pd.DataFrame):
        indices, columns = X.index.values, X.columns.values
        X =X.values

    if any(np.sum(X,1)>2):
        if not np.all(X):
            X = X+1.
        X = X/np.sum(X,1)
    else:
        if not np.all(X):
            eps = get_epsilon(X)
            X = X+eps

    D = X.shape[1]
    Id = np.eye(D)
    Jd = np.ones(D)
    X_clr = np.log(X)@(Id - (1/D)*Jd)
    if indices is not None:
        X_clr = pd.DataFrame(X_clr, index=indices, columns=columns)
    return X_clr



def split_and_preprocess_dataset(dataset_dict, train_ixs, test_ixs, preprocess=True, logdir=None,
                                 standardize_otus=False, standardize_from_training_data=True,
                                 clr_transform_otus=False, sqrt_transform=False):
    """
    splits dataset into train and test splits given and filters/transforms
    based on the training set IF that filtering was not done in data preprocessing.
    Inputs:
    - dataset_dict: output dataset dictionary from process_data.py
    - train_ixs: list of index locations for train set
    - test_ixs: list of index locations for test set

    Outputs:
    - train_dataset_dict: dict with all the same keys as dataset dict, but with just the training subjects and the
    features after filtering
    - test_dataset_dict: dictionary with test subjects and filtered/transformed based on test set

    """
    new_dict = {}
    test_dict = {}
    write_lines=[]
    for key, dataset in dataset_dict.items():
        write_lines.append(key + '\n')
        new_dict[key] = copy.deepcopy(dataset)
        test_dict[key] = copy.deepcopy(dataset)
        if isinstance(dataset['X'], pd.DataFrame):
            new_dict[key]['X'] = dataset['X'].iloc[train_ixs,:]
            test_dict[key]['X'] = dataset['X'].iloc[test_ixs, :]
        else:
            new_dict[key]['X'] = dataset['X'][train_ixs,:]
            test_dict[key]['X'] = dataset['X'][test_ixs, :]
        if isinstance(dataset['y'], pd.Series):
            new_dict[key]['y'] = dataset['y'].iloc[train_ixs]
            test_dict[key]['y'] = dataset['y'].iloc[test_ixs]
        else:
            new_dict[key]['y'] = dataset['y'][train_ixs]
            test_dict[key]['y'] = dataset['y'][test_ixs]
        if 'X_mask' in dataset.keys():
            new_dict[key]['X_mask'] = dataset['X_mask'][train_ixs,:]
            test_dict[key]['X_mask'] = dataset['X_mask'][test_ixs,:]

        if 'preprocessing' in dataset.keys() and dataset['preprocessing'] and preprocess:
            print(key)

            ls = f'{dataset["y"].sum()} subjects have label=1; {len(dataset["y"])-dataset["y"].sum()} subjects have label=0'
            write_lines.append(ls + '\n')
            temp_tr, temp_ts = new_dict[key]['X'], test_dict[key]['X']
            if isinstance(dataset['preprocessing'], dict) and 'percent_present_in' in dataset['preprocessing'].keys():
                ppi=dataset['preprocessing']['percent_present_in']
                if isinstance(dataset['preprocessing'], dict) and 'lod' in dataset['preprocessing'].keys():
                    lod=dataset['preprocessing']['lod']
                else:
                    lod=0
                # lod=0
                temp_tr, temp_ts = filter_by_presence_func(temp_tr, temp_ts, ppi, lod)
                ls = 'After filtering {3} to keep only {3} with levels > {0} in {1}\% of participants, {2} remain'.format(lod, ppi, temp_tr.shape[1], key)
                write_lines.append(ls + '\n')
                print(ls)
            # print(temp_tr.shape)
            if isinstance(dataset['preprocessing'], dict) and 'cov_percentile' in dataset['preprocessing'].keys():
                cp=dataset['preprocessing']['cov_percentile']
                if cp>0:
                    temp_tr, temp_ts = filter_by_cov_func(temp_tr, temp_ts, cp)
                    ls = 'After filtering {2} to keep only {2} with coefficients of variation in the top {0} percentile of participants, {1} {2} remain'.format(
                        cp, temp_tr.shape[1], key)
                    write_lines.append(ls + '\n')
                    print('After filtering {2} to keep only {2} with coefficients of variation in the '
                          'top {0} percentile of participants, {1} {2} remain'.format(
                        cp, temp_tr.shape[1], key))
            # print(temp_tr.shape)
            if key == 'metabs':
                temp_tr, temp_ts = transform_func(temp_tr, temp_ts, standardize_from_training_data = standardize_from_training_data, log_transform=True)

            elif key == 'otus':
                # if (temp_tr>1).any().any():
                #     print('Transforming to RA')
                temp_tr = temp_tr.divide(temp_tr.sum(1),axis='index')
                temp_ts = temp_ts.divide(temp_ts.sum(1),axis='index')
                if clr_transform_otus:
                    temp_tr = CLR(temp_tr)
                    temp_ts = CLR(temp_ts)
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts,
                                                      standardize_from_training_data=standardize_from_training_data,
                                                      log_transform=False)
                    print("WARNING: CLR OTUS")
                elif sqrt_transform:
                    temp_tr = np.sqrt(temp_tr)
                    temp_ts = np.sqrt(temp_ts)
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts,
                                                      standardize_from_training_data=standardize_from_training_data,
                                                      log_transform=False)
                    print("WARNING: SQRT OTUS")
                    # temp_tr, temp_ts = temp_tr.T, temp_ts.T
                elif standardize_otus:
                    temp_tr, temp_ts = transform_func(temp_tr, temp_ts, standardize_from_training_data = standardize_from_training_data, log_transform=True)
                    print("WARNING: STANDARDIZED OTUS")

            new_dict[key]['X'], test_dict[key]['X'] = temp_tr, temp_ts
            kept_features = new_dict[key]['X'].columns.values
            if 'distances' in new_dict[key].keys():
                kept_sm = [k for k in kept_features if k in new_dict[key]['distances'].columns.values]
                new_dict[key]['distances'] = new_dict[key]['distances'][kept_sm].loc[kept_sm]
                test_dict[key]['distances'] = test_dict[key]['distances'][kept_sm].loc[kept_sm]

            new_dict[key]['variable_names'] = kept_features
            test_dict[key]['variable_names'] = kept_features

            ls = f'{temp_tr.shape[1]} features in {key} dataset after filtering. Test set has {temp_ts.shape[0]} samples and train set has {temp_tr.shape[0]} samples'
            write_lines.append(ls)
            print(f'{temp_tr.shape[1]} features in {key} dataset after filtering. Test set has {temp_ts.shape[0]} samples and train set has {temp_tr.shape[0]} samples')
            write_lines.append('\n')

    # if 'otus' in new_dict.keys():
    #     new_dict['otus']['X'] = np.divide(new_dict['otus']['X'].T, new_dict['otus']['X'].sum(1)).T
    #     test_dict['otus']['X'] = np.divide(test_dict['otus']['X'].T, test_dict['otus']['X'].sum(1)).T

    if logdir is not None:
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        with open(logdir + '/data_processing.txt','w') as f:
            f.writelines(write_lines)
    return new_dict, test_dict

# def init_w_knn(emb_locs, dist_mat, size:list, seed, dtype, prior_mean, kappa_max):
#     """
#     Inputs:
#     - emb_locs: [num_features x embedding_dimension] array of embedded feature 'locations'
#     - dist_mat: [num_features x num_features] array of distances between emb_locs
#     - size: list with size of locations/radii to initialize (i.e. for MDITRE, input is [num_rules, num_detectors]. For M2M, input will be [num_clusters])
#     - seed: random seed
#
#     Outputs:
#     - kappa_init: initial radii for all rules and detectors
#     - eta_init: initial centers for all rules and detectors
#     - detector ids: list of lists of which features are assigned to which cluster
#
#     """
#     rng = np.random.default_rng(seed=seed)
#     n_feats = emb_locs.shape[0]
#     emb_dim = emb_locs.shape[1]
#     detector_otuids = list()
#     if len(size)==1:
#         size = [1] + size
#
#     eta_init = np.zeros((size[0], size[1], emb_dim), dtype=np.float32)
#     kappa_init = np.zeros((size[0], size[1]), dtype=np.float32)
#     dist = np.empty((size[0], size[1], n_feats))
#     center_ixs = rng.choice(np.arange(n_feats),size[1]*size[0])
#     # pick otus without repitition
#     kmeans = KMeans(n_clusters=size[1]*size[0], random_state=seed, n_init="auto").fit(emb_locs)
#     # For each rule: (1) pick an OTU for center; (2) get radius from K-nn (max from K-nn)
#         # do K-nn by picking the K closest pairwise distances from the pre-computed distance matrix
#         # to be as principled as possible, get the K by the number closest to the prior radius
#     # Could sample from a constricted version of the prior instead
#         # decrease variance
#     # MOST RATIONAL: Select otu's at random for centers (without replacement), then sample from constricted prior for the radius
#         # X = mean of distribution (i.e. what I'm setting)
#         # [75% of X, 125% of X]
#     # Could also inject noise in the embedded locations for kmeans
#
#     kappa_dist = TruncatedNormal(prior_mean, (prior_mean*(1/12))**2, 0, kappa_max)
#     N_per_cluster = np.ceil(n_feats/(size[1]*size[0]))
#     cluster_id=0
#     for i in range(size[0]):
#         assigned_otus_det = list()
#         for j in range(size[1]):
#             # otu_center_ix = center_ixs[cluster_id]
#             assigned_otus = list()
#             eta_init[i, j] = kmeans.cluster_centers_[cluster_id]
#
#             # eta_init[i, j] = emb_locs[otu_center_ix]
#             med_dist = list()
#             samp_dist = kappa_dist.sample(1).item()
#             # assigned_otus = np.arange(n_feats)[dist_mat[otu_center_ix,:]<=kappa_init[i,j]]
#             # if len(assigned_otus)==1:
#             #     d=dist_mat[otu_center_ix,:]
#             #     kappa_init[i,j] = min(d[np.nonzero(d)])
#             #     assigned_otus = np.arange(n_feats)[dist_mat[otu_center_ix, :] <= kappa_init[i, j]]
#             for k in range(n_feats):
#                 if kmeans.labels_[k] == cluster_id:
#                     med_dist.append(np.linalg.norm(kmeans.cluster_centers_[cluster_id] - emb_locs[k], axis=-1))
#
#             otu_dist_from_center = np.linalg.norm(emb_locs - eta_init[i, j], axis=1)
#             if sum(otu_dist_from_center <= np.median(med_dist)) > 2:
#                 kappa_init[i,j]=np.median(med_dist)
#             elif sum(otu_dist_from_center<=samp_dist)>2:
#                 kappa_init[i,j]=samp_dist
#             else:
#                 kappa_init[i,j]=np.max(med_dist)
#             assigned_otus = np.arange(n_feats)[otu_dist_from_center<=kappa_init[i,j]]
#
#             #     kappa_init[i, j] = np.mean(med_dist)
#             if kappa_init[i,j]==0:
#                 kappa_init[i, j] += 0.01
#             #     # if dtype == 'metabs':
#             #     #     kappa_init[i, j] += 0.1
#             # else:
#             #     d = dist_mat[cur_assig_otu]
#             #     kappa_init[i, j] = min(d[np.nonzero(d)])
#             # for k in range(n_feats):
#             #     if kmeans.labels_[k] == cluster_id:
#             #         try:
#             #             dist[i, j, k] = np.linalg.norm(kmeans.cluster_centers_[cluster_id] - emb_locs[k], axis=-1)
#             #         except:
#             #             print('debug')
#             #         if dist[i, j, k] <= kappa_init[i, j]:
#             #             assigned_otus.append(k)
#             # # assigned_otu_names = [model.var_names[k] for k in assigned_otus]
#             assigned_otus_det.append(assigned_otus.tolist())
#             cluster_id+=1
#
#         detector_otuids.append(assigned_otus_det)
#     return np.squeeze(kappa_init), np.squeeze(eta_init), detector_otuids

def init_w_knn(emb_locs, dist_mat, size:list, seed, dtype):
    """
    Inputs:
    - emb_locs: [num_features x embedding_dimension] array of embedded feature 'locations'
    - dist_mat: [num_features x num_features] array of distances between emb_locs
    - size: list with size of locations/radii to initialize (i.e. for MDITRE, input is [num_rules, num_detectors]. For M2M, input will be [num_clusters])
    - seed: random seed

    Outputs:
    - kappa_init: initial radii for all rules and detectors
    - eta_init: initial centers for all rules and detectors
    - detector ids: list of lists of which features are assigned to which cluster

    """
    n_feats = emb_locs.shape[0]
    emb_dim = emb_locs.shape[1]
    detector_otuids = list()
    if len(size)==1:
        size = [1] + size

    eta_init = np.zeros((size[0], size[1], emb_dim), dtype=np.float32)
    kappa_init = np.zeros((size[0], size[1]), dtype=np.float32)
    dist = np.empty((size[0], size[1], n_feats))
    otu_center_ix = np.random.choice(np.arange(n_feats),100)
    # pick otus without repitition
    kmeans = KMeans(n_clusters=size[1]*size[0], random_state=seed, n_init="auto").fit(emb_locs)
    kmeans_old = {i: KMeans(n_clusters=size[1], random_state=seed + i, n_init="auto").fit(emb_locs) for i in range(size[0])}
    # For each rule: (1) pick an OTU for center; (2) get radius from K-nn (max from K-nn)
        # do K-nn by picking the K closest pairwise distances from the pre-computed distance matrix
        # to be as principled as possible, get the K by the number closest to the prior radius
    # Could sample from a constricted version of the prior instead
        # decrease variance
    # MOST RATIONAL: Select otu's at random for centers (without replacement), then sample from constricted prior for the radius
        # X = mean of distribution (i.e. what I'm setting)
        # [75% of X, 125% of X]
    # Could also inject noise in the embedded locations for kmeans

    cluster_id=0
    for i in range(size[0]):
        assigned_otus_det = list()
        for j in range(size[1]):
            assigned_otus = list()
            eta_init[i, j] = kmeans.cluster_centers_[cluster_id]
            med_dist = list()
            for k in range(n_feats):
                if kmeans.labels_[k] == cluster_id:
                    med_dist.append(np.linalg.norm(kmeans.cluster_centers_[cluster_id] - emb_locs[k], axis=-1))
                    cur_assig_otu = k
            if len(med_dist) > 1:
                kappa_init[i, j] = np.mean(med_dist)
                if kappa_init[i,j]==0:
                    kappa_init[i, j] += 0.01
                # if dtype == 'metabs':
                #     kappa_init[i, j] += 0.1
            else:
                d = dist_mat[cur_assig_otu]
                kappa_init[i, j] = min(d[np.nonzero(d)])
            for k in range(n_feats):
                if kmeans.labels_[k] == cluster_id:
                    try:
                        dist[i, j, k] = np.linalg.norm(kmeans.cluster_centers_[cluster_id] - emb_locs[k], axis=-1)
                    except:
                        print('debug')
                    if dist[i, j, k] <= kappa_init[i, j]:
                        assigned_otus.append(k)
            # assigned_otu_names = [model.var_names[k] for k in assigned_otus]
            assigned_otus_det.append(assigned_otus)
            cluster_id+=1

        detector_otuids.append(assigned_otus_det)
    return np.squeeze(kappa_init), np.squeeze(eta_init), detector_otuids
def init_selectors_by_LR(X, y_s, detector_ids, dtype, inner_folds = 5, scorer='f1'):
    # len of detector_ids is num_rules
    # len of detector_ids[0] is num_detectors
    if isinstance(y_s, pd.Series):
        y = y_s.values
    else:
        y = y_s
    lambda_min_ratio = 0.01
    path_len = 300
    num_rules = len(detector_ids)
    z_out = np.zeros((num_rules, len(detector_ids[0])))
    for k in range(len(detector_ids)):
        # X_in = []
        # for k,v in X_dict.items():
        #     X = v['X']
        X_tmp = np.zeros((X.shape[0], len(detector_ids[k])))
        # J = len(detector_ids[k])
        for j in range(len(detector_ids[k])):
            # k=0: 1,...J
            # k=1: J + 1,2,3,4,5...J
            # k=2: 2*J + 1,2,...J
            if dtype == 'otus':
                X_tmp[:,j] = X[:,detector_ids[k][j]].sum(1)
            else:
                X_tmp[:, j] = X[:, detector_ids[k][j]].mean(1)
            # X_in.append(X_tmp)
        if dtype=='otus':
            X_in = transform_func(X_tmp)
        else:
            X_in = X_tmp

        if k ==0:
            bval = True
            lam = 0
            while (bval):
                lam += 0.1
                model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear', random_state=k)
                model2.fit(X_in, y)
                if np.sum(np.abs(model2.coef_)) < 1e-8:
                    l_max = lam + 1
                    bval = False
            # l_max = 14
            l_path = np.logspace(np.log10(l_max * lambda_min_ratio), np.log10(l_max), path_len)
            Cs = [1 / l for l in l_path]
        model = sklearn.linear_model.LogisticRegressionCV(cv=int(np.min([inner_folds,y.sum()])),
                                                          penalty='l1', scoring=scorer,
                                                          Cs=Cs, solver='liblinear', class_weight='balanced',
                                                          random_state=k)
        model.fit(X_in, y)
        coefs = model.coef_.squeeze()
        # N_u, N_m, N_l = sum(abs(coefs)>np.median(abs(coefs))), sum((abs(coefs)<=np.median(abs(coefs))) & (abs(coefs)>0)), sum(abs(coefs)==0)
        # z_out[k, abs(coefs) > 0] = 0.7 * np.ones(N_u) - 0.4 * np.random.rand(N_u)
        # z_out[k,abs(coefs)>np.median(abs(coefs))] = 0.7*np.ones(N_u) + 0.1*np.random.rand(N_u)
        # z_out[k,(abs(coefs)<=np.median(abs(coefs))) & (abs(coefs)>0)] = 0.6*np.ones(N_m)-0.3*np.random.rand(N_m)
        # z_out[k, abs(coefs)==0] = 0.3*np.ones(N_l) - 0.2 * np.random.rand(N_l)

        N_u, N_m, N_l = sum(abs(coefs)>np.median(abs(coefs))), sum((abs(coefs)<=np.median(abs(coefs))) & (abs(coefs)>0)), sum(abs(coefs)==0)
        z_out[k,abs(coefs)>np.median(abs(coefs))] = 0.6*np.ones(N_u) + 0.1*np.random.rand(N_u)
        z_out[k,(abs(coefs)<=np.median(abs(coefs))) & (abs(coefs)>0)] = 0.55*np.ones(N_m)-0.1*np.random.rand(N_m)
        z_out[k, abs(coefs)==0] = 0.4*np.ones(N_l) - 0.1 * np.random.rand(N_l)
        # locs_9 = np.where(coefs>np.median(coefs))[0]
        # locs_5 = np.where(coefs<=np.median(coefs) and )
    return z_out


def init_w_gms(N_feat, N_clust, dim, r_desired=None, mu_met = None, r_mult = 0.01, r_met=None):
    """
    initializes embedding locations for unknown metabolites, when we learn the embedded locations from model

    Inputs:
    - N_feat: # of unknown features to initialize location of
    - N_clust: # of mixtures (for MDITRE, set to num_detectors)
    - dim: embedding dimension
    - r_desired: either a dictionary of the mean and variance to generate a N_clust radii from a log-normal distribution
      for the gaussian mixtures, or a list of N_clust radii for the gaussian mixtures
      - Can use input dictionary from **calculate_radii_prior()**
    - mu_met: array of [N_clust x emb_dim] centers around which to generate the GMs, or None
        - if None, centers are generated from multivariate normal centered at 0 with variance 0.1
    - r_mult: factor to multiply the radius by (added because the gaussian mixture model with input from calculated
    radii prior was too expansive);
        - default = 0.01

    Outputs:
    - met_locs: generated embedded initial locations
    - mu_met: generated (or input) GM centers
    - r_met: generated (or input) GM radii
    - ids: which features belong to which intial gaussian mixture

    """
    pi_met = st.dirichlet([10/N_clust]*N_clust).rvs().squeeze()
    if mu_met is None:
        mu_met = st.multivariate_normal(np.zeros(dim), 0.1*np.eye(dim, dim)).rvs(N_clust)
    # r_met = st.lognorm(0.01, 1).rvs(N_met_clusters)
    if r_met is None:
        if not isinstance(r_desired, dict):
            if isinstance(r_desired, list) and len(r_desired)==N_clust:
                r_met = r_desired
            else:
                r_met = [r_desired]*N_clust
        else:
            temp = st.norm(np.log(r_desired['mean']),np.sqrt(r_desired['var'])).rvs(N_clust)
            r_met = np.exp(temp)

    z_id = st.multinomial(1, pi_met).rvs(N_feat)
    met_locs = np.zeros((N_feat, dim))
    for clusters in range(N_clust):
        cluster_id = np.where(z_id[:, clusters] == 1)[0]
        met_locs[cluster_id, :] = st.multivariate_normal(mu_met[clusters,:], r_mult*r_met[clusters]*np.eye(dim,

                                                                                     dim)).rvs(len(cluster_id))
    ids = [np.where(z_id[:, i] == 1)[0] for i in range(z_id.shape[1])]
    return met_locs, mu_met, r_met, ids

def compute_dist(dist_emb, num_otus=None):
    """
    used mostly for computing the embedded distance matrix after embedding

    Inputs:
    - dist_emb: [num_features x embedding_dimension] array of embedded locations
    - num_otus: num_features

    Outputs:
    - dist: [num_features x num_features] distance matrix array

    """
    if num_otus is None:
        num_otus = dist_emb.shape[0]
    dist = np.zeros((num_otus, num_otus), dtype=np.float32)
    for i in range(num_otus):
        for j in range(num_otus):
            dist[i, j] = np.linalg.norm(dist_emb[i] - dist_emb[j], axis=-1)
    return dist

def compute_dist_emb_mds(dist_matrix, emb_dim, seed, rescale=True):
    """
    embeds a distance matrix using SVD

    Inputs:
    - dist_matrix: [num_features x num_features] distance matrix array (pre-embedding)
    - emb_dim: dimension to embed
    - seed: random seed

    Outputs:
    - [num_features x embedding_dimension] array of embedded locations

    """
    mds = MDS(n_components=int(emb_dim), random_state=seed,
              dissimilarity='precomputed', normalized_stress="auto")
    mds_transf = mds.fit_transform(dist_matrix)
    dist_emb = mds_transf.astype(np.float32)
    if rescale:
        range_emb = dist_emb.max().max() - dist_emb.min().min()
        dist_emb = dist_emb * (0.7 / range_emb)
    return dist_emb

def test_d_dimensions_pca(d, dist_matrix, seed, expl_var_cuttoff=0.1, rescale=True):
    scalar=StandardScaler()
    dist_mat_sc = scalar.fit_transform(dist_matrix)
    pca = PCA(d, random_state=seed)
    pca_res = pca.fit(dist_mat_sc)
    dim = np.where(pca_res.explained_variance_ratio_*100 < expl_var_cuttoff)[0][0]
    locs = pca.fit_transform(dist_mat_sc)
    locs_d = locs[:,:dim]
    if rescale:
        range_emb = locs_d.max().max()-locs_d.min().min()
        locs_d = locs_d*(0.7/range_emb)
    # locs_d = 0.25*((locs_d - np.min(locs_d, 0))/(np.max(locs_d,0) - np.mean(locs_d, 0)))
    dist_mat = squareform(pdist(locs_d))
    return dim, locs_d.astype(np.float32), dist_mat.astype(np.float32)

def compute_emb_pca(d, dist_matrix, seed, rescale=True):
    scalar=StandardScaler()
    dist_mat_sc = scalar.fit_transform(dist_matrix)
    pca = PCA(d, random_state=seed)
    emb_locs = pca.fit_transform(dist_mat_sc)
    if rescale:
        range_emb = emb_locs.max().max()-emb_locs.min().min()
        emb_locs = emb_locs*(0.7/range_emb)
    dist_mat = squareform(pdist(emb_locs))
    return emb_locs.astype(np.float32), dist_mat.astype(np.float32)

def test_d_dimensions(d, dist_matrix, seed, rescale=True):
    """
    Finds the lowest dimension that results in a Kolmogorov-Smirnov p-value greater than 0.05.
    If no dimension tested exceeds a 0.05 p-value, returns the dimension with the greatest p-value

    Inputs:
    - d: list of dimensions to test
    - dist_matrix: [num_features x num_features] distance matrix array (pre-embedding)
    - seed: random seed

    Outputs:
    - emb_dim: chosen embedding dimension
    - dist_emb: [num_features x embedding_dimension] array of embedded locations at chosen dimension
    - dist_matrix_emb: [num_features x num_features] array of distances b/w embeddings

    """
    if isinstance(dist_matrix, pd.DataFrame):
        dist_matrix = dist_matrix.values
    pvals = []
    for dd in d:
        emb_dim = int(dd)
        dist_emb = compute_dist_emb_mds(dist_matrix, emb_dim, seed, rescale).astype(
            np.float64)
        dist_matrix_embed = squareform(pdist(dist_emb))
        ks = scipy.stats.kstest(dist_matrix.reshape(-1),
                                dist_matrix_embed.reshape(-1))
        pvals.append(ks.pvalue)
        if ks.pvalue >= 0.1:
            emb_dim = int(dd)
            return emb_dim, dist_emb.astype(np.float32), dist_matrix_embed.astype(np.float32), ks.pvalue

    emb_dim = np.argmax(pvals) + d[0]
    if emb_dim == 0:
        emb_dim=np.max(d)
    dist_emb = compute_dist_emb_mds(dist_matrix, emb_dim, seed, rescale).astype(
        np.float32)
    dist_matrix_embed = squareform(pdist(dist_emb))
    return emb_dim, dist_emb.astype(np.float32), dist_matrix_embed.astype(np.float32), pvals[np.argmax(pvals)]

def merge_datasets(dataset_dict):
    """
    Ensures that both datasets have the same samples in the data X and labels y

    Inputs:
    - dataset_dict: dictionary of dataset-dictionaries generated from process_data.py, where each key is the data type (i.e. {'metabs': metabolite-data-dictionar, 'otus': sequence-data-dictionary)

    Outputs:
    - dataset_dict: same dictionary, but with ensuring that indices in X and y for both datasets match
    - y: [N_subjects] outcome labels (i.e. 'y' in either dataset dictionary)

    """
    yls_all = []
    yls = []
    for key, dataset in dataset_dict.items():
        yls.append(set(dataset['X'].index.values))
        yls_all.extend(list(dataset['X'].index.values))
    yls_all = np.unique(yls_all)

    yixs_tmp = list(set.intersection(*yls))
    yixs = [y for y in yls_all if y in yixs_tmp]
    for key, dataset in dataset_dict.items():
        dataset_dict[key]['y']=dataset_dict[key]['y'][~dataset_dict[key]['y'].index.duplicated(keep='first')]
        dataset_dict[key]['X'] = dataset_dict[key]['X'][~dataset_dict[key]['X'].index.duplicated(keep='first')]
        dataset_dict[key]['y'] = dataset['y'].loc[yixs]
        dataset_dict[key]['X'] = dataset['X'].loc[yixs]

    y = dataset_dict[key]['y']
    return dataset_dict, y

# Get stratified kfold train/test splits for cross-val
def cv_kfold_splits(X, y, num_splits=5, seed=42):
    """
    Inputs:
    - X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
    - y: [N_subjects] array of outcomes
    - num_splits: number of k-folds (defaults to 5)
    - seed: random seed

    Outputs:
    - train_ids: list of k lists, where each list is the train index location for that fold
    - test_ids: list of k lists, where each list is the test index location for that fold

    """
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


# Get leave-one-out train/test splits for cross-val
def cv_loo_splits(X, y):
    """
    Inputs:
    - X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
    - y: [N_subjects] array of outcomes

    Outputs:
    - train_ids: list of k lists, where each list is the train index location for that fold
    - test_ids: list of k lists, where each list is the test index location for that fold

    """
    skf = LeaveOneOut()

    train_ids = list()
    test_ids = list()
    for train_index, test_index in skf.split(X, y):
        train_ids.append(train_index)
        test_ids.append(test_index)

    return train_ids, test_ids


