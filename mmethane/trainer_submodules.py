from torch.distributions.uniform import Uniform
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
# from model_nam import featMDITRE
from loss import *
from helper_plots import plot_metab_groups_in_embedding_space, plot_distribution
from utilities.util import *
from utilities.model_helper import TruncatedNormal
import scipy.special as sc
from collections import Counter

class ModuleArguments():
    def __init__(self, args, dtype):
        self.dtype = dtype
        self.args_dict_new = {}
        self.args_dict = vars(args)
        for key, val in self.args_dict.items():
            if isinstance(key, str) and self.dtype in key:
                new_key = key.replace(self.dtype + '_', '')
                setattr(self, new_key, val)
        for key, val in self.args_dict.items():
            if key != 'dtype':
                if hasattr(self, key) is False:
                    setattr(self, key, val)


class empty_loss():
    def __init__(self, *args, **kwargs):
        self.loss_params = []

    def loss(self, *args, **kwargs):
        return 0, {}
    
class empty_model(nn.Module):
    def __init__(self, num_otu_centers=10, emb_dim=10, dtype=None, num_otus=None):
        super(empty_model, self).__init__()
        self.num_otu_centers = num_otu_centers
        self.emb_dim = emb_dim
        self.dtype=dtype
        self.layers = nn.ModuleList([])
        self.num_otus=num_otus

class moduleLit():
    # @profile
    def __init__(self, args, dataset, dtype, dir='', learn_embeddings = False, device = 'cpu', num_rules=None):

        self.device = device
        self.dir = dir
        self.dtype = dtype
        self.var_names = dataset['variable_names']
        self.X = dataset['X']
        self.Y = dataset['y']
        self.taxonomy = dataset['taxonomy']
        self.dist_matrix = dataset['distances']
        if isinstance(self.X, pd.DataFrame):
            assert((self.var_names==self.X.columns.values).all())
            self.X = self.X.values

        if isinstance(self.Y, pd.Series):
            self.Y = self.Y.values

        self.dataset = dataset
        self.tree = dataset['variable_tree']
        self.args = ModuleArguments(args, self.dtype)
        if self.args.data_met is not None:
            if 'infomax' in self.args.data_met:
                self.args.lr_kappa = self.args.lr_kappa * 5
                self.args.lr_eta = self.args.lr_eta * 5

        if self.dtype != 'metabs':
            self.learn_embeddings = False
        else:
            self.learn_embeddings = learn_embeddings

        # note: replace this with non-hardcoded version
        self.k_dict = {'k_otu': {'max': self.args.max_k_otu, 'min': self.args.min_k_otu},
                       'k_thresh': {'max': self.args.max_k_thresh, 'min': self.args.min_k_thresh},
                       'k_alpha':{'max':self.args.max_k_bc, 'min': self.args.min_k_bc}}

        self.num_features = self.X.shape[1]
        self.num_distances = self.dist_matrix.shape[1]
        if self.learn_embeddings == 1:
            self.num_emb_to_learn = self.num_features
        elif self.num_features > self.num_distances and self.dtype == 'metabs':
            self.num_emb_to_learn = self.num_features - self.num_distances
        else:
            self.num_emb_to_learn = 0

        self.num_otus = self.num_features

        self.num_rules = self.args.n_r
        self.num_detectors = self.args.n_d
        if self.args.add_interactions:
            self.num_detectors += self.args.n_d*2
            # self.interaction_detector_ids = np.random.choice(list(itertools.combinations(np.arange(self.args.n_d*2), 2)),
            #                                                  self.args.n_d*2)


        # build and initialize the model
        if self.args.method!='full_fc':
            self.set_model_hparams()
            self.set_init_params()
            self.dir = dir
            if self.args.method=='fc' or self.args.method=='full_fc':
                from models_fc import featMDITRE
            elif 'basic' in args.method:
                from models import featMDITRE
            else:
                ValueError(
                    'Warning: accepted method not provided. Choices are: "basic", "fc", "nam", or "full_fc". Default "basic" will be used.')

            self.model = featMDITRE(self.args.n_d, self.dist_emb, self.emb_dim, self.dtype, self.num_emb_to_learn,
                                    args = self.args, num_rules = self.args.n_r, num_feats = self.num_otus)
            plot_distribution(self, {f'kappa': self.kappa_prior}, self.dir)
            if device is not None:
                self.model = self.model.to(device)

            self.model.init_params(self.init_args, device = device)

            self.logging_dict = {b: [] for b, a in self.model.named_parameters()}

            self.loss_func = moduleLoss(self.model, self.args, self.kappa_prior,
                                        self.normal_emb,
                                        self.dtype, device=self.device)
        else:
            self.emb_dim = self.args.emb_dim
            self.model = empty_model(self.args.n_d, self.emb_dim, self.dtype, self.num_otus)
            if device is not None:
                self.model = self.model.to(device)
            self.logging_dict={}
            self.loss_func = empty_loss()


    def init_w_kmeans(self,emb_locs, size: list, dist_mat = None):
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
        if dist_mat is None:
            dist_mat = squareform(pdist(emb_locs))
        seed = self.args.seed
        n_feats = emb_locs.shape[0]
        emb_dim = emb_locs.shape[1]
        detector_otuids = list()
        if len(size) == 1:
            size = [1] + size
        kmeans_={}
        emb_std = np.std(emb_locs)
        noise=0
        for i in range(size[0]):
            kmeans_[i]=KMeans(n_clusters=size[1], random_state=seed + i).fit(emb_locs+noise)
            if self.args.kmeans_noise:
                noise = np.random.normal(0, 0.1 * emb_std, emb_locs.shape)

        eta_init = np.zeros((size[0], size[1], emb_dim), dtype=np.float32)
        kappa_init = np.zeros((size[0], size[1]), dtype=np.float32)
        dist = np.empty((size[0], size[1], n_feats))
        lines=[f'Num Rules x Num Detectors: {size}\n']
        for i in range(size[0]):
            assigned_otus_det = list()
            kmeans=kmeans_[i]
            lines.append(f'rule {i}')
            # kmeans = KMeans(n_clusters=size[1], random_state=seed + i, n_init="auto").fit(emb_locs)
            for j in range(size[1]):
                assigned_otus = list()
                eta_init[i, j] = kmeans.cluster_centers_[j]
                med_dist = list()
                lines.append(f'detector {j}')
                for k in range(n_feats):
                    if kmeans.labels_[k] == j:
                        med_dist.append(np.linalg.norm(kmeans.cluster_centers_[j] - emb_locs[k], axis=-1))
                        cur_assig_otu = k
                if len(med_dist) > 1:
                    kappa_init[i, j] = np.mean(med_dist)
                    if kappa_init[i, j] == 0:
                        kappa_init[i, j] += 0.01
                    # if dtype == 'metabs':
                    #     kappa_init[i, j] += 0.1
                else:
                    d = dist_mat[cur_assig_otu]
                    kappa_init[i, j] = min(d[np.nonzero(d)])
                for k in range(n_feats):
                    if kmeans.labels_[k] == j:
                        try:
                            dist[i, j, k] = np.linalg.norm(kmeans.cluster_centers_[j] - emb_locs[k], axis=-1)
                        except:
                            print('debug')
                        if dist[i, j, k] <= kappa_init[i, j]:
                            assigned_otus.append(k)
                # assigned_otu_names = [model.var_names[k] for k in assigned_otus]
                lines.append(f'kappa: {kappa_init[i,j]}')
                lines.append(f'# in cluster: {len(assigned_otus)}')
                lines.append('\n')
                assigned_otus_det.append(assigned_otus)

            detector_otuids.append(assigned_otus_det)

        with open(self.dir+'/init_clusters.txt','w') as f:
            for l in lines:
                f.write(l)
                f.write('\n')

        return np.squeeze(kappa_init), np.squeeze(eta_init), detector_otuids


    def set_model_hparams(self):

        # Set mean and variance for neg bin prior for detectors
        # self.negbin_det = create_negbin(self.args.z_mean, self.args.z_var)
        if self.learn_embeddings == 0:
            if self.args.emb_dim is None:
                if self.args.use_pca==1:
                    self.emb_dim, self.dist_emb, self.dist_matrix_embed = test_d_dimensions_pca(
                        self.dist_matrix.shape[0], self.dist_matrix, self.args.seed, self.args.expl_var_cutoff)
                    print(f'\nFor {self.dtype}, dimension {self.emb_dim} has an explained variance below the cutoff of {self.args.expl_var_cutoff}')
                else:
                    d = np.arange(2, 31)
                    self.emb_dim, self.dist_emb, self.dist_matrix_embed, pval = test_d_dimensions(d, self.dist_matrix, self.args.seed)
                    print(
                        f'\nFor {self.dtype}, dimension {self.emb_dim} has a p-value of {pval}')
            else:
                self.emb_dim = int(self.args.emb_dim)
                if self.args.use_pca==1:
                    self.dist_emb, self.dist_matrix_embed = compute_emb_pca(self.emb_dim, self.dist_matrix, self.args.seed)
                else:
                    self.dist_emb = compute_dist_emb_mds(self.dist_matrix, self.args.emb_dim, self.args.seed).astype(
                        np.float32)
                    self.dist_matrix_embed = compute_dist(self.dist_emb, self.dist_matrix.shape[0])
            pd.DataFrame(self.dist_emb, index=self.dist_matrix.index.values).to_csv(self.dir + '/' + self.dtype + '_emb_locs.csv', header=None, index=True)

        else:
            if self.args.emb_dim is None:
                self.args.emb_dim = 30.0
            self.emb_dim = int(self.args.emb_dim)

        # self.dist_emb.to_csv(self.dir + '/emb_locs.csv')
        # plt.figure(); plt.hist(self.dist_emb.values.flatten()); plt.savefig(self.dir + '/emb_locs.pdf'); plt.close();
        # self.ref_median=calculate_radii_prior(self.dataset, pd.DataFrame(self.dist_matrix_embed,
        #                                                                  index = self.dist_matrix.index.values,
        #                                                                  columns= self.dist_matrix.columns.values),
        #                                       self.dtype, self.args.multiplier)
        # print(self.ref_median)

        self.emb_mean = 0
        self.emb_var = 1e8
        mean = torch.ones(self.emb_dim, dtype=torch.float32, device=self.device) * self.emb_mean
        cov = torch.eye(self.emb_dim, dtype=torch.float32, device=self.device) * self.emb_var
        self.normal_emb = MultivariateNormal(mean, cov)


        self.kappa_min = 0
        if self.args.use_old_refs==1 and self.dtype=='otus':
            if 'cdi' in self.args.data_otu:
                self.kappa_prior_mean = np.exp(-0.98787415)
                self.kappa_prior_var = 0.7533012
                print('Using 16s reference values')
            else:
                self.kappa_prior_mean=np.exp(-1.1044373025931737)
                self.kappa_prior_var = 0.405738891096145
            if self.dtype=='metabs':
                self.kappa_prior_mean =0.3806578
                self.kappa_prior_var =0.33470775932073593

        else:
            self.ref_median = calculate_radii_prior(self.dataset, pd.DataFrame(self.dist_matrix_embed,
                                                                               index=self.dist_matrix.index.values,
                                                                               columns=self.dist_matrix.columns.values),
                                                    self.dtype, self.args.multiplier)

            self.kappa_prior_mean = self.ref_median['mean']
            self.kappa_prior_var = self.ref_median['var']
            print(self.ref_median)
        self.kappa_max = self.dist_matrix_embed.flatten().max() + 0.01*self.dist_matrix_embed.flatten().max()
        if self.args.kappa_prior == 'trunc-normal':
            self.kappa_prior = TruncatedNormal(self.kappa_prior_mean, self.kappa_prior_var, self.kappa_min,
                                               self.kappa_max, device=self.device)
        elif self.args.kappa_prior == 'log-normal':
            self.kappa_prior = Normal(torch.tensor(np.log(self.kappa_prior_mean), device=self.device), torch.tensor(np.sqrt(self.kappa_prior_var), device=self.device))
        else:
            raise ValueError('Provide a valid input argument for kappa prior')

            # (dist_fit, dist_mat, size:list, emb_dim, seed)
        if 'basic' not in self.args.method:
            size = [self.num_detectors]

        else:
            size = [self.num_rules, self.num_detectors]

        self.kappa_init, self.eta_init, self.detector_otuids = self.init_w_kmeans(
            self.dist_emb, size, dist_mat=self.dist_matrix_embed)

        self.emb_to_learn = None


    def set_init_params(self):
        if self.args.init_with_LR:
            self.z_out = init_selectors_by_LR(self.X, self.Y, self.detector_otuids, self.dtype)
            self.alpha_init = sc.logit(self.z_out)*self.args.init_multiplier
        else:
            if 'basic' not in self.args.method:

                self.alpha_init = np.random.normal(0, 1, (self.num_detectors)) * self.args.init_multiplier
            else:
                if self.args.alpha_init_zeros:
                    self.alpha_init = np.zeros((self.num_rules, self.num_detectors))
                else:
                    self.alpha_init = np.random.normal(0, 1, (self.num_rules, self.num_detectors)) * self.args.init_multiplier
        if self.args.method=='basic':
            if self.dtype == 'metabs':
                self.thresh_min, self.thresh_max = np.min(self.X.flatten()) - 0.01*np.min(self.X.flatten()), \
                                                np.max(self.X.flatten()) + 0.01*np.max(self.X.flatten())
            else:
                self.thresh_min, self.thresh_max = 0, 1
            self.uniform_thresh = Uniform(torch.tensor(self.thresh_min).to(self.device), torch.tensor(self.thresh_max).to(self.device))

            thresh_init = np.zeros((self.num_rules, self.num_detectors), dtype=np.float32)
            all_init_detectors = list()
            for l in range(self.num_rules):
                init_detectors = np.zeros((self.X.shape[0], self.num_detectors))
                for m in range(self.num_detectors):
                    if len(self.detector_otuids[l][m]) > 0:
                        x = self.X[:, self.detector_otuids[l][m]]
                        if self.dtype == 'metabs':
                            x_m = x.mean(1)
                        else:
                            x_m = x.sum(1)
                        thresh_init[l, m] = x_m.mean()
                    else:
                        thresh_init[l, m] = 0
        else:
            thresh_init = np.zeros(1)
        if self.args.kappa_prior == 'log-normal':
            self.kappa_init = np.log(self.kappa_init)
        self.init_args = {
            'thresh_init': thresh_init,
            'kappa_init': self.kappa_init,
            'eta_init': self.eta_init,
            'emb_init': self.emb_to_learn,
            'alpha_init':self.alpha_init
        }




