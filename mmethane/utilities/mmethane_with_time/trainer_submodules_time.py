from torch.distributions.uniform import Uniform
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
# from model_nam import featMDITRE
from loss_time import *
from viz import plot_metab_groups_in_embedding_space
from eval_learned_embeddings import eval_embeddings
from utilities.util import *
from model_helper import TruncatedNormal
from time_agg import time_inits_for_featMDITRE, time_priors_for_featMDITRE

class ModuleArguments():
    def __init__(self, args, dtype):
        self.dtype = dtype
        self.args_dict_new = {}
        self.args_dict = vars(args)
        for key, val in self.args_dict.items():
            if isinstance(key, str) and self.dtype in key:
                new_key = key.replace(self.dtype + '_', '')
                setattr(self, new_key, val)
            elif key != 'dtype':
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
        if 'X_mask' in dataset.keys():
            self.X_mask = dataset['X_mask']
        if len(self.X.shape)>2:
            self.num_time = self.X.shape[-1]
        else:
            self.num_time = 1
        self.dist_matrix = dataset['distances']
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.values
        if isinstance(self.Y, pd.Series):
            self.Y = self.Y.values

        self.dataset = dataset
        self.args = ModuleArguments(args, self.dtype)
        if 'infomax' in self.args.data_met:
            self.args.lr_kappa = self.args.lr_kappa * 5
            self.args.lr_eta = self.args.lr_eta * 5

        if self.dtype != 'metabs':
            self.learn_embeddings = False
        else:
            self.learn_embeddings = learn_embeddings

        # note: replace this with non-hardcoded version
        self.k_dict = {'k_otu': {'max': self.args.max_k_otu, 'min': self.args.min_k_otu},
                       'k_thresh': {'max': self.args.max_k_thresh, 'min': self.args.min_k_thresh}}
        if self.args.time==1:
            self.k_dict.update({'k_time':{'max':self.args.max_k_time,'min':self.args.min_k_time},
                                'k_slope':{'max':self.args.max_k_slope,'min':self.args.min_k_slope}})

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

        # build and initialize the model
        if self.args.method!='full_fc':
            self.set_model_hparams()
            self.set_init_params()
            self.dir = dir
            if self.args.method=='fc' or self.args.method=='full_fc':
                from model_fc import featMDITRE
            elif self.args.method=='nam':
                from model_nam import featMDITRE
            elif self.args.method=='nam_orig':
                from model_nam_orig import featMDITRE
            else:
                from models_time import featMDITRE
            self.model = featMDITRE(self.args.n_d, self.dist_emb, self.emb_dim, self.dtype, self.num_emb_to_learn,
                                    args = self.args, num_rules = self.args.n_r, num_feats = self.num_otus, num_time = self.num_time)
            if device is not None:
                self.model = self.model.to(device)

            self.model.init_params(self.init_args, device = device)

            self.logging_dict = {b: [] for b, a in self.model.named_parameters()}

            if self.args.time:
                normal_window, normal_center = time_priors_for_featMDITRE(self.num_time, self.device)
                self.loss_func = moduleLoss(self.model, self.args, self.kappa_prior,
                                            self.normal_emb,
                                            self.dtype, device=self.device, normal_window=normal_window,
                                            normal_center=normal_center)
            else:
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

    # @profile
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


        self.emb_mean = 0
        self.emb_var = 1e8
        mean = torch.ones(self.emb_dim, dtype=torch.float32, device=self.device) * self.emb_mean
        cov = torch.eye(self.emb_dim, dtype=torch.float32, device=self.device) * self.emb_var
        self.normal_emb = MultivariateNormal(mean, cov)


        if self.learn_embeddings == 0:
            # (dist_fit, dist_mat, size:list, emb_dim, seed)
            if self.args.method!='basic':
                self.kappa_init, self.eta_init, self.detector_otuids = init_w_knn(
                    self.dist_emb, self.dist_matrix_embed, [self.num_detectors], self.args.seed)
            else:
                self.kappa_init, self.eta_init, self.detector_otuids = init_w_knn(
                    self.dist_emb, self.dist_matrix_embed, [self.num_rules, self.num_detectors], self.args.seed)

            if self.num_distances < self.num_features and self.dtype == 'metabs':
                if self.args.method!='basic':
                    self.emb_to_learn, mu_met, r_met, z_ids = init_w_gms(self.num_features - self.num_distances,
                                                                        self.num_detectors,
                                                                        self.emb_dim, r_met=self.kappa_init,
                                                                        mu_met=self.eta_init)
                else:
                    self.emb_to_learn, mu_met, r_met, z_ids = init_w_gms(self.num_features - self.num_distances,
                                                    self.num_detectors,
                                                    self.emb_dim, r_met=self.kappa_init[0,:],
                                                    mu_met=self.eta_init[0, :, :])
                if isinstance(self.emb_to_learn, torch.Tensor):
                    self.emb_to_learn = self.emb_to_learn.numpy()
                eval_embeddings(self.emb_to_learn, self.dist_emb, path=self.dir + '/init_gmm')

                self.dist_matrix_embed = compute_dist(np.vstack((self.dist_emb, self.emb_to_learn)),
                                                      self.dist_emb.shape[0] + self.emb_to_learn.shape[0])
            else:
                self.emb_to_learn = None
        else:
            self.emb_to_learn, mu, r, ids = init_w_gms(self.num_otus, self.num_detectors,
                                                       self.emb_dim, self.ref_median)
            self.dist_matrix_embed = compute_dist(self.emb_to_learn, self.emb_to_learn.shape[0])
            self.dist_emb = self.emb_to_learn
            self.kappa_init, self.eta_init, self.detector_otuids = init_w_knn(
                self.emb_to_learn, self.dist_matrix_embed, [self.num_detectors],self.args.seed)
            plot_metab_groups_in_embedding_space(self.detector_otuids[0], self.dist_emb,
                                                 self.eta_init[0, :, :], radii=self.kappa_init[0, :], dir=self.dir + '/kmeans_')

        self.kappa_min = 0
        if self.args.use_old_refs==1:
            if 'cdi' in self.args.data_otu or 'semisyn' in self.args.data_otu:
                self.kappa_prior_mean = np.exp(-0.98787415)
                self.kappa_prior_var = 0.7533012
                print('Using 16s reference values')
            else:
                self.kappa_prior_mean=np.exp(-1.1044373025931737)
                self.kappa_prior_var = 0.405738891096145
        else:
            self.ref_median = calculate_radii_prior(self.dataset, pd.DataFrame(self.dist_matrix_embed,
                                                                               index=self.dist_matrix.index.values,
                                                                               columns=self.dist_matrix.columns.values),
                                                    self.dtype, self.args.multiplier)
            print(self.ref_median)
            self.kappa_prior_mean = self.ref_median['mean']
            self.kappa_prior_var = self.ref_median['var']
        self.kappa_max = self.dist_matrix_embed.flatten().max() + 0.01*self.dist_matrix_embed.flatten().max()
        if self.args.kappa_prior == 'trunc-normal':
            self.kappa_prior = TruncatedNormal(self.kappa_prior_mean, self.kappa_prior_var, self.kappa_min,
                                               self.kappa_max, device=self.device)
        elif self.args.kappa_prior == 'log-normal':
            self.kappa_prior = Normal(torch.tensor(np.log(self.kappa_prior_mean), device=self.device), torch.tensor(np.sqrt(self.kappa_prior_var), device=self.device))
        else:
            raise ValueError('Provide a valid input argument for kappa prior')

        if self.args.method == 'basic':
            if self.dtype == 'metabs':
                self.thresh_min, self.thresh_max = np.min(self.X.flatten()) - 0.01 * np.min(self.X.flatten()), \
                                                   np.max(self.X.flatten()) + 0.01 * np.max(self.X.flatten())
            else:
                self.thresh_min, self.thresh_max = 0, 1
            self.uniform_thresh = Uniform(torch.tensor(self.thresh_min).to(self.device),
                                          torch.tensor(self.thresh_max).to(self.device))

    def set_init_params(self):
        if self.args.time==1:
            self.init_args = time_inits_for_featMDITRE(self.num_time, self.num_rules, self.num_detectors,
                                                       self.X_mask, self.X, self.Y, self.detector_otuids)
            self.init_args['kappa_init'] = self.kappa_init
            self.init_args['eta_init'] = self.eta_init
            self.init_args['emb_init'] = self.emb_to_learn
        else:
            if self.args.method=='basic':
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
                'emb_init': self.emb_to_learn
            }




