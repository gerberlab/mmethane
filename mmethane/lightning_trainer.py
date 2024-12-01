import sys
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append(os.path.abspath(".."))
from torch.utils.data import Dataset
from lightning.pytorch import seed_everything

from utilities.util import split_and_preprocess_dataset, cv_kfold_splits, merge_datasets, cv_loo_splits
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
# from lightning.pytorch.loggers import CSVLogger
import argparse

from helper_plots import *
import json
from loss import mapLoss

import warnings

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
import datetime
# from trainer_submodules_nam import moduleLit
# from model_nam import ComboMDITRE
from trainer_submodules import moduleLit
from torch import optim
# from lightning.pytorch.callbacks import ModelCheckpoint
# from CustomCheckpoint import *
import pickle as pkl
import lightning.pytorch as pl

# torch.autograd.detect_anomaly()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_float32_matmul_precision('high')
START_TIME = time.time()
from torchmetrics import AUROC, Accuracy
from torchmetrics.classification import MulticlassF1Score
from utilities.model_helper import run_logreg, logreg_test
# from multiprocessing import Pool
import matplotlib
from lightning.pytorch.callbacks import ModelCheckpoint

matplotlib.use("Agg")
import matplotlib.style as mplstyle

mplstyle.use('fast')
import ray

global_parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
global_parser.add_argument('--use_ray', type=int, default=0)
global_args, _ = global_parser.parse_known_args()

if global_args.use_ray:
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": os.getcwd(),
                                                    "py_modules": ["./utilities/"],
                                                    "excludes":['./utilities/phylo_placement/refpkg/RDP-11-5_TS_Processed.refpkg/RDP-11-5_TS_Processed_Aln.fa',
                                                                '.git/objects/pack/pack-4e454680d6ad3af8da44da0a5a44658070856ad5.pack',
                                                                 '.git/objects/pack/pack-4e454680d6ad3af8da44da0a5a44658070856ad5.pack',
                                                                '.git/objects/pack/',
                                                               '*.job',
                                                                'core.*']})

# ray.init()
# sys.setrecursionlimit(2097152)
# TO DO:
#   - add filtering transforming into function (only filter/transform training data)
#   - fix plot results for two datasets

def conditional_decorator(dec, condition, **kwargs):
    def decorator(func):
        if condition:
            if kwargs is not None:
                # print(func)
                return dec(**kwargs)(func)
            else:
                # print(func)
                return dec()(func)
        else:
            # Return the function unchanged, not decorated.
            return func
    return decorator

def parse(parser):
    parser.add_argument('--num_inner_folds', type=int, default=5)
    parser.add_argument('--method', type=str, default='basic', choices=['basic', 'fc', 'nam_non_agg',
                                                                        'full_fc', 'nam', 'no_rules',
                                                                        'nam_with_interactions'])
    parser.add_argument('--init_multiplier', type=float, default=0.1)
    parser.add_argument('--plot_all_seeds', type=int, default=1)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--init_with_LR', type=int, default=0)
    parser.add_argument('--add_interactions', type=int, default=0)
    parser.add_argument('--add_logreg', type=int, default=0)
    parser.add_argument('--lr_logreg', type=float, default=0.001)
    parser.add_argument('--h_sizes', type=int, nargs='+', default=[12, 6])
    parser.add_argument('--standardize_from_training_data', type=int, default=1)
    parser.add_argument('--num_cpus', type=float, default=0.5)
    parser.add_argument('--num_gpus', type=float, default=0)

    # Main model specific parameters
    parser.add_argument('--n_r', type=int, default=10, help='Number of rules')
    parser.add_argument('--lr_beta', default=0.001, type=float,
                        help='Initial learning rate for binary concrete logits on rules.', nargs='+')
    parser.add_argument('--metabs_lr_alpha', default=0.001, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.', nargs='+')
    parser.add_argument('--otus_lr_alpha', default=0.001, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.', nargs='+')
    parser.add_argument('--min_k_bc', default=1, type=float, help='Min Temperature for binary concretes')
    parser.add_argument('--max_k_bc', default=10, type=float, help='Max Temperature for binary concretes')
    parser.add_argument('--lr_fc', default=0.01, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')
    parser.add_argument('--lr_bias', default=0.01, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')
    parser.add_argument('--w_var', type=float, default=1e4, help='Normal prior variance on weights.')
    parser.add_argument('--bias_var', type=float, default=1e4, help='Normal prior variance on bias.')
    parser.add_argument('--z_r_mean', type=float, default=1, help='NBD Mean active rules')
    parser.add_argument('--z_r_var', type=float, default=5, help='NBD variance of active rules')
    parser.add_argument('--z_mean', type=float, default=1, help='NBD Mean active detectors per rule')
    parser.add_argument('--z_var', type=float, default=5, help='NBD variance of active detectors per rule')
    parser.add_argument('--adj_rule_loss', type=int, default=0)

    # Metabolite model specific parameters
    parser.add_argument('--metabs_lr_kappa', default=0.001, type=float, help='Initial learning rate for kappa.',
                        nargs='+')
    parser.add_argument('--metabs_lr_eta', default=0.001, type=float, help='Initial learning rate for eta.', nargs='+')
    parser.add_argument('--metabs_lr_emb', default=0.001, type=float, help='Initial learning rate for emb.', nargs='+')
    parser.add_argument('--metabs_lr_thresh', default=0.0005, type=float, help='Initial learning rate for threshold.',
                        nargs='+')
    parser.add_argument('--metabs_min_k_otu', default=100, type=float,
                        help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--metabs_max_k_otu', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--metabs_min_k_thresh', default=1, type=float,
                        help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--metabs_max_k_thresh', default=10, type=float,
                        help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--metabs_min_k_bc', default=0.5, type=float, help='Min Temperature for binary concretes')
    parser.add_argument('--metabs_max_k_bc', default=10, type=float, help='Max Temperature for binary concretes')
    parser.add_argument('--metabs_n_d', type=int, default=10, help='Number of detectors')
    parser.add_argument('--metabs_emb_dim', type=float)
    parser.add_argument('--metabs_multiplier', type=float, default=10)
    parser.add_argument('--metabs_expl_var_cutoff', type=float, default=0.05)
    parser.add_argument('--metabs_use_pca', type=int, default=0)
    parser.add_argument('--metabs_lr_nam', type=float, default=0.0001)
    parser.add_argument('--metabs_neg_bin_prior', type=int, default=0)
    parser.add_argument('--metabs_bernoulli_prior', type=int, default=1)
    parser.add_argument('--metabs_init_clusters', type=str.lower,default='kmeans',
                        choices=['kmeans','kmeansconstrained','family','genus','coresets','clades', 'new'])
    parser.add_argument('--metabs_adj_n_d', type=int, default=0)
    parser.add_argument('--metabs_kappa_mult', type=float, default=1e3)
    parser.add_argument('--metabs_adj_kappa_loss', type=int, default=0)
    parser.add_argument('--metabs_adj_detector_loss', type=int, default=0)

    # Microbe model specific parameters
    parser.add_argument('--otus_lr_kappa', default=0.001, type=float, help='Initial learning rate for kappa.')
    parser.add_argument('--otus_lr_eta', default=0.001, type=float, help='Initial learning rate for eta.')
    parser.add_argument('--otus_lr_emb', default=0.001, type=float, help='Initial learning rate for emb.')
    parser.add_argument('--otus_lr_thresh', default=0.0001, type=float, help='Initial learning rate for threshold.')
    parser.add_argument('--otus_min_k_otu', default=100, type=float,
                        help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--otus_max_k_otu', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--otus_min_k_thresh', default=100, type=float,
                        help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--otus_max_k_thresh', default=1000, type=float,
                        help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--otus_min_k_bc', default=1, type=float, help='Min Temperature for binary concretes')
    parser.add_argument('--otus_max_k_bc', default=10, type=float, help='Max Temperature for binary concretes')
    parser.add_argument('--otus_n_d', type=int, default=10, help='Number of detectors')
    parser.add_argument('--otus_emb_dim', type=float)
    parser.add_argument('--otus_multiplier', type=float, default=1)
    parser.add_argument('--otus_expl_var_cutoff', type=float, default=0.05)
    parser.add_argument('--otus_use_pca', type=int, default=0)
    parser.add_argument('--otus_lr_nam', type=float, default=0.0001)
    parser.add_argument('--otus_neg_bin_prior', type=int, default=0)
    parser.add_argument('--otus_bernoulli_prior', type=int, default=1)
    parser.add_argument('--otus_init_clusters', type=str.lower,default='kmeans',
                        choices=['kmeans', 'kmeansconstrained', 'family', 'genus', 'coresets','clades','new'])
    parser.add_argument('--otus_adj_n_d', type=int, default=0)
    parser.add_argument('--otus_kappa_mult', type=float, default=1e3)
    parser.add_argument('--otus_adj_kappa_loss', type=int, default=0)
    parser.add_argument('--otus_adj_detector_loss', type=int, default=0)

    # Training Parameters
    # ('../datasets/HE/processed/he_pubchem/2_mets.pkl',
    #  '../datasets/HE/processed/he_cts/2_seqs.pkl'),
    parser.add_argument('--data_met', metavar='DIR',
                        help='path to metabolite dataset',
                        default='../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl')
    parser.add_argument('--data_otu', metavar='DIR',
                        help='path to otu dataset',
                        default='../datasets/ERAWIJANTARI/processed/erawijantari_cts/seqs.pkl')
    parser.add_argument('--run_name', type=str,
                        help='Name for log folder',
                        default="run_1",
                        )
    parser.add_argument('--min_epochs', default=100, type=int, metavar='N',
                        help='number of minimum epochs to run')
    parser.add_argument('--epochs', default=3000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', type=int, default=7,
                        help='Set random seed for reproducibility')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'one', 'eval'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--early_stopping', default=1, type=int)
    parser.add_argument('--validate', default=0, type=int)
    parser.add_argument('--test', default=1, type=int)
    # parser.add_argument('--emb_dim', type = float, default=20)
    parser.add_argument('--out_path', type=str, default='logs/', help = "path to logs")
    parser.add_argument('--num_anneals', type=float, default=1)
    parser.add_argument('--monitor', type=str, default='train_loss')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--dtype', type=str,
                        default=['metabs','otus'],
                        choices=['metabs', 'otus'],
                        help='Choose type of data', nargs='+')
    parser.add_argument('--schedule_lr', type=int, default=0,
                        help='Schedule learning rate')
    parser.add_argument('--parallel', type=int, default=6,
                        help='run in parallel')
    parser.add_argument('--only_mets_w_emb', type=int, default=1, help='whether or not keep only mets with embeddings')
    parser.add_argument('--only_otus_w_emb', type=int, default=1, help='whether or not keep only otus with embeddings')
    parser.add_argument('--learn_emb', type=int, default=0, help='whether or not to learn embeddings')
    parser.add_argument('--debug', type=int, default=0)
    # parser.add_argument('--lr_master',type=float, default=None)
    parser.add_argument('--from_saved_files', type=int, default=0)
    parser.add_argument('--use_k_1', type=int, default=1)
    parser.add_argument('--use_noise', type=int, default=0)
    parser.add_argument('--noise_anneal', type=float, default=[1, 1], nargs='+')
    parser.add_argument('--remote', type=int, default=0)
    parser.add_argument('--annealing_limit', type=float, default=[0.05, 0.95], nargs='+')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--anneal_type', type=str, default='linear', choices=['linear', 'cosine', 'exp'])
    parser.add_argument('--div_loss', type=int, default=0)
    # parser.add_argument('--p_d', type=float, default=0.1)
    parser.add_argument('--metabs_p_d', type=float, default=0.1)
    parser.add_argument('--otus_p_d', type=float, default=0.1)
    parser.add_argument('--p_r', type=float, default=0.1)
    parser.add_argument('--old', type=int, default=0)
    parser.add_argument('--neg_bin_prior', type=int, default=0)
    parser.add_argument('--bernoulli_prior', type=int, default=1)
    parser.add_argument('--kappa_eta_prior', type=int, default=1)
    parser.add_argument('--n_mult', type=int, default=1)
    parser.add_argument('--hard_otu', type=int, default=0)
    parser.add_argument('--hard_bc', type=int, default=0)
    parser.add_argument('--filter_data', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--kappa_prior', type=str, default='log-normal', choices=['log-normal', 'trunc-normal'])
    parser.add_argument('--use_old_refs', type=int, default=0)
    parser.add_argument('--z_loss_mult', type=float, default=1)
    parser.add_argument('--p_d_grid', nargs='+', type=float,
                        # default=None)
                        default=np.logspace(-4, -0.5, 6).tolist())
    parser.add_argument('--p_r_grid', nargs='+', type=float,
                        # default=None)
                        default=np.logspace(-4, -1, 4).tolist())
    # default=[1,2,5,7,10,12,15,17,20])
    parser.add_argument('--param_name', type=str, nargs='+', default=['p_d', 'p_r'])
    parser.add_argument('--nested_cv', type=int, default=0)
    parser.add_argument('--nested_cv_metric', type=str, default='f1')
    parser.add_argument('--adj_pd', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--plot_traces', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='NAdam', choices=['Adam','RMSprop','NAdam','RAdam','AdamW'])
    parser.add_argument('--eta_min', type=float, default=0.00001)
    parser.add_argument('--eta_min_frac', type=float, default=1e-2)
    parser.add_argument('--patience', type=float, default=100)
    parser.add_argument('--kmeans_noise', type=int, default=1)
    parser.add_argument('--alpha_init_zeros', type=int, default=1)

    # parser.add_argument('--full_fc', type=int, default=1)
    args, _ = parser.parse_known_args()
    return args, parser


class LitMDITRE(pl.LightningModule):
    # @profile
    def __init__(self, args, data_dict, dir, learn_embeddings=False):
        super().__init__()
        # self.save_hyperparameters()
        self.dir = dir
        self.args = args
        self.data_dict = data_dict
        self.learn_embeddings = learn_embeddings
        self.noise_factor = 1.
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_true, self.val_true, self.test_true = [], [], []
        self.F1Score = MulticlassF1Score(2,average='weighted')
        self.AUROC = AUROC(task='binary')
        self.Accuracy = Accuracy(task='binary')
        # self.automatic_optimization = False

    def parse_args(self):
        # self.args.n_r=self.args.n_r*len(self.args.dtype)
        # if self.args.fc==1:
        #     from model_fc import ComboMDITRE
        # else:
        #     from model_nam import ComboMDITRE
        self.k_dict = {'k_beta': {'max': self.args.max_k_bc, 'min': self.args.min_k_bc}}
                       # 'k_alpha': {'max': self.args.max_k_bc, 'min': self.args.min_k_bc}}
        self.class_dict = {}
        self.model_dict = {}

        if not isinstance(self.args.dtype, list):
            self.args.dtype = [self.args.dtype]
        self.loss_params = []
        self.num_detectors = 0
        self.n_d_per_class = {}


        for type in self.args.dtype:
            if type != 'metabs':
                self.learn_embeddings = False
            nfeats = self.data_dict[type]['X'].shape[1]
            nsubjs = self.data_dict[type]['X'].shape[0]
            if getattr(self.args, f'{type}_adj_n_d')==1:
                nd = int(np.ceil((nfeats+nsubjs)/(2*self.args.n_r)))
                setattr(self.args,f'{type}_n_d',nd)
                setattr(self.args, f'{type}_p_d',1/nd)
            self.class_dict[type] = moduleLit(self.args, copy.deepcopy(self.data_dict[type]), type,
                                              dir=self.dir, learn_embeddings=self.learn_embeddings, device=device)

            # for i in range(len(self.class_dict[type].detector_otuids)):
            #     self.detector_ids[i].extend(self.class_dict[type].detector_otuids[i])
            self.num_detectors += self.class_dict[type].num_detectors
            self.n_d_per_class[type] = self.class_dict[type].num_detectors
            self.model_dict[type] = self.class_dict[type].model
            self.loss_params.extend(self.class_dict[type].loss_func.loss_params)
            k_dict = {type + '_' + key: value for key, value in self.class_dict[type].k_dict.items()}
            self.k_dict.update(k_dict)

        self.k_step = {}
        for k_param, vals in self.k_dict.items():
            self.k_step[k_param] = self.k_dict[k_param]['min']

        self.set_model_hparams()
        self.set_init_params()

        # build and initialize the model
        self.model = ComboMDITRE(self.args, self.model_dict)
        self.model.init_params(self.init_args, device=device)

        self.logging_dict = {b: [] for b, a in self.model.named_parameters()}
        for type in self.args.dtype:
            log_dict = {type + '_' + key: value for key, value in self.class_dict[type].logging_dict.items()}
            self.logging_dict.update(log_dict)
            #         self.logging_dict[name + '_' + b].append(a.detach().clone())
            # self.logging_dict.update({type + '_' + key: [] for key,value in self.model.rules[type]})
        self.scores_dict = {'train f1': [], 'test f1': [], 'val f1': [], 'train auc': [], 'test auc': [],
                            'train loss': [], 'test loss': [], 'val loss': [], 'val auc': [], 'total val loss': [],
                            'train acc 0': [], 'train acc 1': [], 'val acc 0': [], 'val acc 1': [], 'test acc 0': [],
                            'test acc 1': [], 'num active detectors':[],'num active rules':[]}
        # self.loss_func = mapLoss(self.model, self.args, self.normal_wts, self.bernoulli_rules, self.bernoulli_det)
        self.loss_func = mapLoss(self.model, self.args, self.normal_wts, self.normal_bias, self.n_detectors_prior, device=device,
                                 n_r_prior=self.n_rules_prior, n_d = self.num_detectors)
        self.running_loss_dict = {}
        self.grad_dict = {}
        for b, a in self.model.named_parameters():
            # print(b, a.get_device())
            self.grad_dict[b] = []
        for name, model in self.model_dict.items():
            for b, a in model.named_parameters():
                # print(b, a.get_device())
                self.grad_dict[name + '_' + b] = []

    def get_loss(self, outputs, labels, kdict):

        loss_, reg_loss, loss_dict = self.loss_func.loss(outputs, labels, kdict['k_beta'])
        loss = loss_.clone()
        loss += reg_loss
        f=None
        if self.current_epoch == 1990:
            f=open(self.dir + '/frac_bc_loss.txt', 'w')
        for name, module in self.class_dict.items():
            reg_l, ldict = module.loss_func.loss(kdict[name + '_k_alpha'], len(labels), self.class_dict[name].num_features)
            if self.current_epoch == 1990:
                # with open(self.dir + '/frac_bc_loss.txt', 'w') as f:
                f.write(name)
                f.write(f'Fraction of BC loss: {len(labels) /(args.n_mult*self.class_dict[name].num_features)}\n')
                f.write(
                    f'Fraction of kappa loss: {len(labels) /(self.class_dict[name].num_features * 1e5)}\n')
                f.write('\n')
            loss += reg_l
            loss_dict.update(ldict)
            # if self.args.div_loss == 1:
            #     div_loss = diversity_loss(self.model.z_r, self.model.z_d, module.model.wts)
            #     loss_dict[name + '_diversity_loss'] = div_loss
            #     loss += div_loss
        if f is not None:
            f.close()
        return loss, loss_dict

    def set_model_hparams(self):
        self.num_rules = self.args.n_r
        self.wts_mean = 0
        # mean = torch.ones(self.num_rules, dtype=torch.float, device=device) * self.wts_mean
        # cov = torch.eye(self.num_rules, dtype=torch.float, device=device) * self.args.w_var
        self.normal_wts = Normal(self.wts_mean, self.args.w_var)
        self.normal_bias=Normal(self.wts_mean, self.args.bias_var)

        # Set mean and variance for neg bin prior for rules
        if self.args.neg_bin_prior == 1:
            self.n_rules_prior = create_negbin(self.args.z_r_mean, self.args.z_r_var, device=device)
            self.n_detectors_prior = create_negbin(self.args.z_mean, self.args.z_var, device=device)
        else:
            self.n_rules_prior = None
            self.n_detectors_prior = None

        # self.alpha_bc = BinaryConcrete(loc=1, tau=1 / self.k_dict['k_alpha']['min'])
        # self.beta_bc = BinaryConcrete(loc=1, tau=1 / self.k_dict['k_alpha']['min'])

    def set_init_params(self):
        beta_init = np.random.normal(0, 1, (self.num_rules)) * self.args.init_multiplier
        w_init = np.random.normal(0, 1, (1, self.num_rules))
        bias_init = np.zeros((1))

        self.init_args = {
            'w_init': w_init,
            'bias_init': bias_init,
            'beta_init': beta_init,
        }

    def forward(self, x_dict, hard_otu, hard_bc, noise_factor):
        return self.model(x_dict, self.k_step, hard_otu=hard_otu, hard_bc=hard_bc, noise_factor=noise_factor)

    def training_step(self, batch, batch_idx):

        for mclass in self.class_dict.values():
            if self.args.method == 'basic':
                mclass.model.thresh.data.clamp_(mclass.thresh_min, mclass.thresh_max)
            if self.args.kappa_prior == 'trunc-normal':
                mclass.model.kappa.data.clamp_(mclass.kappa_min, mclass.kappa_max)
        xdict, y = batch

        if self.args.use_noise == 1:
            if self.args.noise_anneal[0] != self.args.noise_anneal[1]:
                self.noise_factor = linear_anneal(self.args.epochs - self.current_epoch, self.args.noise_anneal[1],
                                                  self.args.noise_anneal[0], self.args.epochs, 0,
                                                  self.args.noise_anneal[0])
            else:
                self.noise_factor = 1.
        else:
            self.noise_factor = 0.

        for k_param, vals in self.k_dict.items():
            if self.current_epoch < int(self.args.annealing_limit[1] * self.args.epochs) and \
                    self.current_epoch > int(self.args.annealing_limit[0] * self.args.epochs):
                anneal_start = self.current_epoch - int(self.args.annealing_limit[0] * self.args.epochs)
                anneal_end = int(self.args.annealing_limit[1] * self.args.epochs)
                anneal_steps = int(self.args.annealing_limit[0] * self.args.epochs)
                if vals['min'] == vals['max']:
                    self.k_step[k_param] = vals['min']
                else:
                    if self.args.anneal_type == 'linear':
                        self.k_step[k_param] = linear_anneal(anneal_start, vals['max'], vals['min'], anneal_end,
                                                             anneal_steps, vals['min'])
                    elif self.args.anneal_type == 'exp':
                        self.k_step[k_param] = exp_anneal(anneal_start, anneal_steps, anneal_end, vals['min'],
                                                          vals['max'])
                    else:
                        self.k_step[k_param] = cosine_annealing(anneal_start, vals['min'], vals['max'],
                                                                int(self.args.epochs * (self.args.annealing_limit[1] -
                                                                                        self.args.annealing_limit[0])))
            self.log(k_param, self.k_step[k_param])
        if self.current_epoch % 1000 == 0:
            print('\nEpoch ' + str(self.current_epoch))
            print(f'{self.current_epoch} epochs took {time.time() - START_TIME} seconds')

        if self.current_epoch == self.args.epochs-1:
            print('debug')
        y_hat = self(xdict, hard_otu=args.hard_otu == 1, hard_bc=args.hard_bc == 1, noise_factor=self.noise_factor)
        if self.args.add_logreg and self.current_epoch == self.args.epochs - 1:
            self.logreg_model, self.train_mean, self.train_stdev = run_logreg(xdict, y)

        # opt = self.optimizers()
        # opt.zero_grad()
        self.loss, self.loss_dict = self.get_loss(y_hat, y, self.k_step)
        # self.manual_backward(self.loss)
        # opt.step()
        # sch = self.lr_schedulers()
        # sch.step()
        # for opt in self.optimizer_ls:
        #     opt.zero_grad()
        # self.manual_backward(self.loss)
        # for opt in self.optimizer_ls:
        #     opt.step()
        # for sch in self.scheduler_ls:
        #     sch.step()




        self.train_preds.extend(y_hat.sigmoid())
        self.train_true.extend(y)
        self.train_model = pkl.loads(pkl.dumps(self.model))
        return self.loss

    def on_train_epoch_end(self):
        # self.train_model = pkl.loads(pkl.dumps(self.model))
        for b, a in self.model.named_parameters():
            self.logging_dict[b].append(a.clone())

        for name, model in self.model_dict.items():
            for b, a in model.named_parameters():
                self.logging_dict[name + '_' + b].append(a.clone())
        for key in self.loss_dict.keys():
            if key not in self.running_loss_dict.keys():
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(self.loss_dict[key])

        y = torch.stack(self.train_true, 0)
        y_hat = torch.stack(self.train_preds, 0)
        f1 = self.F1Score(y_hat > 0.5, y)
        try:
            auc = self.AUROC(y_hat, y)
        except:
            print('ERROR: ONLY 1 CLASS IN AUC! (TRAIN)')
            print('y has length={0}'.format(len(y)))
            print(y)
            auc = np.nan

        ctrls, case = y == 0, y != 0
        acc_0 = self.Accuracy(y_hat[ctrls] > 0.5, y[ctrls])
        acc_1 = self.Accuracy(y_hat[case] > 0.5, y[case])
        self.log('acc 0', acc_0, logger=False)
        self.log('acc 1', acc_1, logger=False)
        self.log('total train loss', self.loss)
        self.log('train f1', f1)
        self.log('train auc', auc)
        self.log('train loss', self.loss_dict['train_loss'])
        self.scores_dict['train auc'].append(auc)
        self.scores_dict['train f1'].append(f1)
        self.scores_dict['train loss'].append(self.loss)
        self.scores_dict['train acc 0'].append(acc_0)
        self.scores_dict['train acc 1'].append(acc_1)
        if self.args.method!='full_fc':
            self.scores_dict['num active detectors'].append((self.model.z_d.flatten() > 0.1).sum().item())
            self.scores_dict['num active rules'].append((self.model.z_r.flatten() > 0.1).sum().item())
        self.train_true, self.train_preds = [], []
        return (None)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, hard_otu=args.hard_otu == 1, hard_bc=args.hard_bc == 1, noise_factor=self.noise_factor)
        if self.args.add_logreg and self.current_epoch == self.args.epochs - 1:
            y_hat_logreg, score = logreg_test(x, y, self.logreg_model, self.train_mean, self.train_stdev)
            print(f'LOG REG F1: {score}')
            f1 = self.F1Score(y_hat > 0.5, y)
            print(f'MDITRE F1: {f1}')

            y_hat = (y_hat + torch.tensor(y_hat_logreg[:, int(self.logreg_model.classes_[-1])])) / 2
            f1 = self.F1Score(y_hat > 0.5, y)
            print(f'COMBO SCORE: {f1}')
        self.val_loss, self.val_loss_dict = self.get_loss(y_hat, y, self.k_step)
        self.val_preds.extend(y_hat.sigmoid())
        self.val_true.extend(y)
        return self.val_loss

    def on_validation_epoch_end(self):
        self.val_model = pkl.loads(pkl.dumps(self.model))
        y, y_hat = torch.stack(self.val_true, 0), torch.stack(self.val_preds, 0)
        self.y_preds = y_hat
        self.y_true = y
        f1 = self.F1Score(y_hat > 0.5, y)
        try:
            f1 = self.F1Score(y_hat > 0.5, y)
            auc = self.AUROC(y_hat, y)
            ctrls, case = y == 0, y != 0
            acc_0 = self.Accuracy(y_hat[ctrls] > 0.5, y[ctrls])
            acc_1 = self.Accuracy(y_hat[case] > 0.5, y[case])
            self.log('val acc 0', acc_0)
            self.log('val acc 1', acc_1)
            self.log('val auc', auc)
            self.log('val f1', f1)
        except:
            f1, auc, acc_0, acc_1 = 0, 0, 0, 0

        self.log('total val loss', self.val_loss)
        #
        self.log('val loss', self.val_loss_dict['train_loss'])

        if len(self.scores_dict['train loss']) > 0:
            self.scores_dict['val f1'].append(f1)
            self.scores_dict['val auc'].append(auc)
            self.scores_dict['total val loss'].append(self.val_loss)
            self.scores_dict['val loss'].append(self.val_loss_dict['train_loss'])
            self.scores_dict['val acc 0'].append(acc_0)
            self.scores_dict['val acc 1'].append(acc_1)

        self.y_hat_val, self.y_val = y_hat, y
        self.val_true, self.val_preds = [], []
        return (None)
        # return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, hard_otu=args.hard_otu == 1, hard_bc=args.hard_bc == 1, noise_factor=0)
        if self.args.add_logreg and self.current_epoch == self.args.epochs - 1:
            y_hat_logreg, score = logreg_test(x, y, self.logreg_model, self.train_mean, self.train_stdev)
            y_hat = (y_hat + y_hat_logreg) / 2
        self.test_preds.extend(y_hat.sigmoid())
        self.test_true.extend(y)
        self.test_loss, _ = self.get_loss(y_hat, y, self.k_step)
        return self.test_loss

    def on_test_epoch_end(self):
        y, y_hat = torch.stack(self.test_true, 0), torch.stack(self.test_preds, 0)
        f1 = self.F1Score(y_hat > 0.5, y)
        ctrls, case = y == 0, y != 0
        acc_0 = self.Accuracy(y_hat[ctrls] > 0.5, y[ctrls])
        acc_1 = self.Accuracy(y_hat[case] > 0.5, y[case])
        # auc = roc_auc_score(y.detach().cpu().numpy(), self.y_hat.sigmoid().detach().cpu().numpy())
        self.scores_dict['test loss'].append(self.test_loss.detach().item())
        self.scores_dict['test f1'].append(f1)
        self.scores_dict['test acc 0'].append(acc_0)
        self.scores_dict['test acc 1'].append(acc_1)
        if len(y) > 1:
            try:
                auc = self.AUROC(y_hat, y)
                self.log('test auc', auc)
            except:
                auc = np.nan
                print('ERROR: ONLY 1 CLASS IN AUC! (TEST)')
                print('y has length={0}'.format(len(y)))
            self.scores_dict['test auc'].append(auc)
        self.log('test f1', f1)
        self.log('test loss', self.test_loss, logger=False)
        self.log('test ctrl acc', acc_0)
        self.log('test case acc', acc_1)
        self.y_preds = y_hat
        self.y_true = y

        self.test_true, self.test_preds = [], []
        return (None)

    def configure_optimizers(self):
        lr_ls = []
        self.optimizer_ls=[]
        self.scheduler_ls=[]
        for name, param in self.model.named_parameters():
            if 'lr_' + name in self.args.__dict__.keys():
                val = self.args.__dict__['lr_' + name]
                if isinstance(val, list):
                    val = val[0]
                lr_ls.append({'lr': val, 'params': [param]})
            elif 'lr_' + name.split('.')[0] in self.args.__dict__.keys():
                val = self.args.__dict__['lr_' + name.split('.')[0]]
                if isinstance(val, list):
                    val = val[0]
                lr_ls.append({'lr': val, 'params': [param]})
            elif len(name.split('.')) > 1:
                ix = int(name.split('.')[1])
                nm = name.split('.')[2].split('_')[0]
                dtype = self.model.module_names[ix]
                lr = self.args.__dict__[dtype + '_' + 'lr_' + nm]
                if isinstance(lr, list):
                    lr = lr[0]
                lr_ = {'lr': lr, 'params': [param]}
                lr_ls.append({'lr': lr, 'params': [param]})
            else:
                lr=0.001
                lr_ = {'params': [param]}
                lr_ls.append({'params': [param]})
            # lr_ls.append(lr_)
        if self.args.optimizer=='Adam':
            self.optimizer_0 = optim.Adam(lr_ls, weight_decay=self.args.weight_decay)
        elif self.args.optimizer=='RMSprop':
            self.optimizer_0 = optim.RMSprop(lr_ls, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'NAdam':
            self.optimizer_0 = optim.NAdam(lr_ls, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'RAdam':
            self.optimizer_0 = optim.RAdam(lr_ls, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            self.optimizer_0 = optim.AdamW(lr_ls, weight_decay=self.args.weight_decay)
        else:
            ValueError("Please provide correct argument for optimizer. Options are Adam or RMSprop.\n"
                       "To add additional options, see lightning_trainer.py, lines 617-620")
        if args.schedule_lr == 1:
            self.scheduler_0 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_0,
                                                                    int(self.args.epochs / self.args.num_anneals),
                                                                    eta_min=self.args.eta_min)
            # eta_min=1e-5)
            return [self.optimizer_0], [self.scheduler_0]
        else:
            return [self.optimizer_0]
        #     if self.args.optimizer=='Adam':
        #         self.optimizer_ls.append(optim.Adam([lr_], weight_decay=self.args.weight_decay))
        #     elif self.args.optimizer=='RMSprop':
        #         self.optimizer_ls.append(optim.RMSprop([lr_], weight_decay=self.args.weight_decay))
        #     elif self.args.optimizer == 'NAdam':
        #         self.optimizer_ls.append(optim.NAdam([lr_], weight_decay=self.args.weight_decay))
        #     elif self.args.optimizer == 'RAdam':
        #         self.optimizer_ls.append(optim.RAdam([lr_], weight_decay=self.args.weight_decay))
        #     elif self.args.optimizer == 'AdamW':
        #         self.optimizer_ls.append(optim.AdamW([lr_], weight_decay=self.args.weight_decay))
        #     else:
        #         ValueError("Please provide correct argument for optimizer. Options are Adam or RMSprop.\n"
        #                    "To add additional options, see lightning_trainer.py, lines 617-620")
        #     if self.args.schedule_lr == 1:
        #         self.scheduler_ls.append(optim.lr_scheduler.CosineAnnealingLR(self.optimizer_ls[-1],
        #                                                                  int(self.args.epochs / self.args.num_anneals),
        #                                                                  eta_min=self.args.eta_min_frac*lr))
        # if self.args.schedule_lr == 1:
        #     return self.optimizer_ls, self.scheduler_ls
        # else:
        #     return self.optimizer_ls


class mditreDataset(Dataset):
    def __init__(self, dataset_dict):
        self.x_dict = {}
        for name, dataset in dataset_dict.items():
            x = dataset['X']
            if isinstance(x, pd.DataFrame):
                x = x.values
            self.x_dict[name] = torch.tensor(x, device=device, dtype=torch.float)

        y = dataset['y']
        if isinstance(y, pd.Series):
            y = y.values

        self.y = torch.tensor(y, device=device, dtype=torch.float)
        # self.idxs = idxs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {name: self.x_dict[name][idx] for name in self.x_dict.keys()}, self.y[idx]

@conditional_decorator(ray.remote, global_args.use_ray, num_returns=4)
def hyperparam_training(inner_train_ixs, inner_test_ixs, param_val, train_dataset_dict, logger, batch_size, args, outpath, inner_fold=0):
    inner_train_dataset_dict, val_dataset_dict = split_and_preprocess_dataset(train_dataset_dict,
                                                                              inner_train_ixs, inner_test_ixs,
                                                                              preprocess=False)
    val_loader = DataLoader(mditreDataset(val_dataset_dict), batch_size=batch_size, shuffle=False)
    inner_train_loader = DataLoader(mditreDataset(inner_train_dataset_dict), batch_size=batch_size,
                                    shuffle=True)

    if param_val is not None:
        hypertuning_file = logger.log_dir + '/hyperparam_tuning.txt'
        prev_val_score = 0
        # best_param = self.args.param_grid[0]
        inner_callbacks = [ModelCheckpoint(save_last=False,
                                           dirpath=logger.log_dir,
                                           save_top_k=0,
                                           verbose=False)]

        # param_scores = {}
        # args.__dict__['hard_otu'] = 1
        # args.__dict__['hard_bc'] = 1
        for i, param_name in enumerate(args.param_name):
            args.__dict__[param_name] = param_val[i]
            # if self.args.adj_pd:
            #     if param_name=='p_d':
            #         self.args.__dict__[param_name] = 2*param_val
            # if param_name == 'z_mean' or param_name == 'z_r_mean':
            #     args.__dict__[param_name.split('mean')[0] + 'var'] = param_val + 2
        model = LitMDITRE(args, inner_train_dataset_dict,
                          dir=outpath + f'seed_{args.seed}',
                          learn_embeddings=args.learn_emb == 1)
        model.parse_args()
        st_inner = time.time()
        inner_trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, min_epochs=args.min_epochs,
                                   check_val_every_n_epoch=1,
                                   callbacks=inner_callbacks, log_every_n_steps=1,
                                   enable_progress_bar=False)
        inner_trainer.fit(model, train_dataloaders=inner_train_loader, val_dataloaders=val_loader)
        val_score = model.scores_dict[f'val {args.nested_cv_metric}'][-1]
        # param_scores[param_val] = val_score
        if os.path.isfile(hypertuning_file):
            write_method = 'a'
        else:
            write_method = 'w'
        with open(hypertuning_file, write_method) as f:
            f.write(
                ', '.join([f'{p}={param_val[i]}' for i,p in enumerate(args.param_name)]) + f', inner_fold={inner_fold}\n')
            f.write('\n'.join([f'{k}: {v[-1]}' for k, v in model.scores_dict.items() if len(v) > 0]) + '\n')
            f.write(
                f'Training {args.epochs} epochs took {np.round((time.time() - st_inner) / 60,5)} minutes')
            f.write('\n\n')

        # for file in os.listdir(logger.log_dir):
        #     if file.startswith('events.out.tfevents.'):
        #         os.remove(os.path.join(logger.log_dir, file))
        return param_val, val_score, model.y_val, model.y_hat_val

@conditional_decorator(ray.remote, global_args.use_ray, num_gpus=0)
class CVTrainer():
    def __init__(self, args, OUTPUT_PATH, y):
        self.args = args
        self.outpath = OUTPUT_PATH
        self.Y = y
        self.y = y.values
        self.overwrite_previous=False

    def check_if_fold_already_finished(self, fold, log_dir):
        if os.path.isdir(log_dir):
            pred_ls = {}
            for file_folder in os.listdir(log_dir):

                if os.path.isdir(log_dir + '/' + file_folder):
                    inner_files = os.listdir(log_dir + '/' + file_folder)
                    for file in inner_files:
                        if 'pred_results' in file:
                            # if file_folder != 'last':
                            #     k = 'best'
                            # else:
                            #     k = file_folder
                            pred_ls = pd.read_csv(log_dir + '/' + file_folder + '/' + file, index_col=0)
                            print('Fold {0} testing already finished'.format(fold))
                            return pred_ls
                else:
                    if 'pred_results' in file_folder:
                        pred_ls = pd.read_csv(log_dir + '/' + file_folder, index_col=0)
                        print('Fold {0} testing already finished'.format(fold))
                        return pred_ls
                    # pred_ls.append(pd.read_csv(tb_logger.log_dir + '/' + file))
            # if len(pred_ls.keys())>0:
            #     print('Fold {0} testing already finished'.format(fold))
            #     return pred_ls

    # @profile
    def test_model(self, model, trainer, test_ixs, ckpt_path, test_loader):

        out = trainer.test(model=model, ckpt_path=ckpt_path, dataloaders=test_loader)
        preds_df = pd.DataFrame({'ixs': test_ixs, 'subj_IDs': self.Y.index.values[test_ixs], 'true': model.y_true.cpu(),
                                 'preds': model.y_preds.cpu()}).set_index('ixs')
        if ckpt_path[0] == '.':
            out_path = '.' + ckpt_path.split('.')[1]
        else:
            out_path = ckpt_path.split('.')[0]
        print('ckpt_path: ' + ckpt_path)
        print('outpath: ' + out_path)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        if 'epoch=' in ckpt_path:
            epoch = int(re.findall('epoch=\d*', ckpt_path)[0].split('epoch=')[-1])
        else:
            epoch = -1
        try:
            scores_at_epoch = {k: v[-1].detach().cpu().numpy() for k, v in model.scores_dict.items() if len(v) > 0}
        except:
            try:
                scores_at_epoch = {k: model.scores_dict[k][epoch] for k in model.scores_dict.keys() if 'test' not in k}
            except:
                scores_at_epoch = {}
                for k in model.scores_dict.keys():
                    if 'test' not in k and 'val' not in k:
                        if len(model.scores_dict[k])==0:
                            continue
                        scores_at_epoch[k] = model.scores_dict[k][epoch]

        for k in scores_at_epoch.keys():
            try:
                scores_at_epoch[k] = scores_at_epoch[k].detach().cpu().numpy()
            except:
                continue
        pd.Series(scores_at_epoch).to_csv(self.output_path + '/scores_at_eval.csv')
        try:
            preds_df.to_csv(self.output_path + '/pred_results_f1_{0}'.format(
                np.round(scores_at_epoch['val f1'], 3)).replace('.', '-') + '.csv')
        except:
            preds_df.to_csv(self.output_path + '/pred_results.csv')
        return preds_df

    # @ray.remote(num_returns=4)

    # @profile
    # @ray.remote
    def train_loop(self, dataset_dict, train_ixs, test_ixs, fold):

        outer_start = time.time()
        self.train_ixs = train_ixs
        self.test_ixs = test_ixs
        if self.args.batch_size is None:
            self.batch_size = len(train_ixs) + 100
        else:
            self.batch_size = self.args.batch_size

        if isinstance(fold, str):
            vers = fold
        else:
            vers = f'fold_{fold}'
        tb_logger = TensorBoardLogger(save_dir=self.outpath, name=f'seed_{self.args.seed}', version=vers)
        self.output_path = tb_logger.log_dir
        if self.overwrite_previous is False:
            pred_ls = self.check_if_fold_already_finished(fold, tb_logger.log_dir)
            if pred_ls is not None:
                return pred_ls

        monitor = args.monitor.replace('_', ' ')

        callbacks = [ModelCheckpoint(save_last=False,
                                     dirpath=tb_logger.log_dir,
                                     save_top_k=self.args.validate,
                                     verbose=False,
                                     monitor=monitor,
                                     every_n_epochs=None, every_n_train_steps=None, train_time_interval=None,
                                     mode='min' if 'loss' in monitor else 'max',
                                     filename='{epoch}' + '-{' + monitor + ':.2f}',
                                     )
                         ]
        if args.early_stopping == 1:
            callbacks.extend([EarlyStopping(monitor=monitor, patience=self.args.patience, min_delta = 1e-2)])
        # else:
        # callbacks.append(LearningRateMonitor())
        trainer = pl.Trainer(logger=tb_logger, max_epochs=self.args.epochs, min_epochs=self.args.min_epochs,
                             callbacks=callbacks,
                             enable_progress_bar=False,
                             )

        if self.args.filter_data:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=True,
                                                                                 sqrt_transform = self.args.method=='full_fc',
                                                                                 # clr_transform_otus=self.args.method == 'full_fc',
                                                                                 clr_transform_otus=False,
                                                                                 standardize_otus=False,
                                                                                 # standardize_otus=self.args.method == 'full_fc',
                                                                                 standardize_from_training_data=self.args.standardize_from_training_data,
                                                                                 logdir=tb_logger.log_dir)
        else:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=False,
                                                                                 sqrt_transform=self.args.method == 'full_fc',
                                                                                 # clr_transform_otus=self.args.method == 'full_fc',
                                                                                 standardize_from_training_data=self.args.standardize_from_training_data,
                                                                                 clr_transform_otus=False,
                                                                                 standardize_otus=False,
                                                                                 # standardize_otus=self.args.method == 'full_fc',
                                                                                 logdir=tb_logger.log_dir)
            if 'otus' in dataset_dict.keys():
                dataset_dict['otus']['X'] = dataset_dict['otus']['X'].divide(dataset_dict['otus']['X'].sum(1),
                                                                             axis='index')
        # if self.args.n_r is None or self.args.n_r == 0:
        #     self.args.n_r = 10
        # if self.args.p_r is None or self.args.p_r==0:
        #     self.args.p_r = 0.1
        # if (self.args.metabs_n_d is None or self.args.metabs_n_d==0) and 'metabs' in train_dataset_dict.keys():
        #     self.args.metabs_n_d = int(np.floor(dataset_dict['metabs']['X'].shape[1]/15))
        #     if self.args.metabs_p_d is None or self.args.metabs_p_d==0:
        #         self.args.metabs_p_d=np.floor(dataset_dict['metabs']['X'].shape[1])/100
        #         if self.args.metabs_p_d >0.5:
        #             self.args.metabs_p_d = 0.5
        # if (self.args.otus_n_d is None or self.args.otus_n_d==0) and 'otus' in train_dataset_dict.keys():
        #     self.args.otus_n_d = int(np.floor(dataset_dict['otus']['X'].shape[1]/15))
        #     if self.args.otus_p_d is None or self.args.otus_p_d==0:
        #         self.args.otus_p_d=np.floor(dataset_dict['otus']['X'].shape[1]/100)
        #         if self.args.otus_p_d >0.5:
        #             self.args.otus_p_d = 0.5
        train_loader = DataLoader(mditreDataset(train_dataset_dict), batch_size=self.batch_size,
                                  shuffle=True)


        if fold=='EVAL':
            sts = f'seed_{args.seed}/{fold}/'
        else:
            sts = f'seed_{args.seed}/fold_{fold}/'
        self.lit_model = LitMDITRE(self.args, train_dataset_dict,
                                   dir=self.outpath + sts,
                                   learn_embeddings=self.args.learn_emb == 1)
        self.lit_model.parse_args()
        self.train_dataset_dict = copy.deepcopy(train_dataset_dict)
        with open(self.outpath + f'/seed_{self.args.seed}' + '/commandline_args_eval.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        args_dict = {}
        for n, model_dict in self.lit_model.class_dict.items():
            args_dict.update(model_dict.args.__dict__)
        args_dict.update(self.args.__dict__)
        try:
            with open(self.outpath + f'/seed_{self.args.seed}' + '/commandline_args_eval.txt', 'w') as f:
                json.dump(args_dict, f, indent=2)
        except:
            pass

        if self.args.validate == 1:
            inner_train_ixs, inner_val_ixs = train_test_split(np.arange(len(train_ixs)), test_size=0.1,
                                                              stratify=self.y[train_ixs],
                                                              random_state=self.args.seed)
            inner_train_dataset_dict, val_dataset_dict = split_and_preprocess_dataset(train_dataset_dict,
                                                                                      inner_train_ixs,
                                                                                      inner_val_ixs,
                                                                                      preprocess=False)
            val_loader = DataLoader(mditreDataset(val_dataset_dict), batch_size=self.batch_size, shuffle=False)
            train_loader = DataLoader(mditreDataset(inner_train_dataset_dict), batch_size=self.batch_size,
                                      shuffle=True)
            trainer.fit(self.lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            torch.save(self.lit_model.model.state_dict(), os.path.join(tb_logger.log_dir,'trained_state_dict.pt'))
            test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
            # self, model, trainer, test_ixs, ckpt_path, test_loader
            path = self.outpath + f'seed_{args.seed}' + '/' + vers + '/'
            tmp = [f for f in os.listdir(path) if ('.ckpt' in f and 'epoch' in f)][0]
            ckpt_path = path + tmp
            preds = self.test_model(self.lit_model, trainer, self.test_ixs, ckpt_path, test_loader)
            best_epoch = int(tmp.split('epoch=')[-1].split('-')[0])
        else:
            best_epoch = -1
            # if self.args.nested_cv and (self.args.param_grid is not None and len(self.args.param_grid)>1):
            if self.args.nested_cv:
                print('HYPERPARAMETER TRAINING')
                if not os.path.isdir(tb_logger.log_dir):
                    os.mkdir(tb_logger.log_dir)
                tr_y = self.y[train_ixs]
                if np.sum(tr_y) < self.args.num_inner_folds:
                    nfolds = np.sum(tr_y)
                elif (len(tr_y) - np.sum(tr_y)) < self.args.num_inner_folds:
                    nfolds = (len(tr_y) - np.sum(tr_y))
                else:
                    nfolds = self.args.num_inner_folds
                tr_ixs, ts_ixs = cv_kfold_splits(np.zeros(tr_y.shape[0]), tr_y, num_splits=nfolds, seed=args.seed)
                input_list = list(itertools.product(list(zip(tr_ixs, ts_ixs, list(range(len(tr_ixs))))), self.args.p_d_grid,
                                               self.args.p_r_grid))
                # hyperparam_results=[]
                # with open(f"{tb_logger.log_dir}/time.txt", "w") as f:
                #     f.write(f"INNER FOLD LOG\n")
                inner_start = time.time()
                running_results=[]
                # hyperparam_training(inner_train_ixs, inner_test_ixs, param_val, train_dataset_dict, logger, batch_size, args, outpath, inner_fold=0):
                for (inner_train_ixs, inner_test_ixs, fold_iter), pa, pb in input_list:
                    if global_args.use_ray:
                        res = hyperparam_training.remote(inner_train_ixs,
                                                                                   inner_test_ixs,
                                                                                   (pa, pb),
                                                                                   train_dataset_dict,
                                                                                   tb_logger, self.batch_size,
                                                                                   self.args, self.outpath,
                                                                                   inner_fold=fold_iter)
                    else:
                        res = hyperparam_training(inner_train_ixs,inner_test_ixs,(pa, pb),
                                                                                   train_dataset_dict,tb_logger,self.batch_size,
                                                                                   self.args, self.outpath,inner_fold=fold_iter)
                    running_results.append(res)
                    # inputs_ran.append(inputs)
                    inner_time = time.time() - inner_start
                param_ls_, score_ls_, y_true_, y_pred_ = list(zip(*running_results))
                if global_args.use_ray:
                    param_ls_, score_ls_, y_true_, y_pred_ = ids_to_vals(param_ls_), ids_to_vals(
                        score_ls_), ids_to_vals(y_true_), ids_to_vals(y_pred_)
                if len(param_ls_) != len(list(set(param_ls_))):
                    score_ls, param_ls = [], []
                    for param_val in list(set(param_ls_)):
                        param_val = ids_to_vals(param_val)
                        pixs = np.array([i for i, p in enumerate(param_ls_) if p == param_val])
                        # y_true_ = [list(y.numpy()) for y in y_true_]
                        # y_pred_ = [list(y.numpy()) for y in y_pred_]
                        y_val_true_tot = []
                        y_val_pred_tot = []
                        for pi in pixs:
                            y_val_true_tot.extend(list(ids_to_vals(y_true_[pi].numpy())))
                            y_val_pred_tot.extend(list(ids_to_vals(y_pred_[pi].numpy())))
                        # y_val_true_tot = list(itertools.chain.from_iterable(np.array(y_true_)[pixs]))
                        # y_val_pred_tot = list(itertools.chain.from_iterable(np.array(y_pred_)[pixs]))
                        if self.args.nested_cv_metric == 'f1':
                            score_ls.append(f1_score(np.array(y_val_true_tot), np.array(y_val_pred_tot) > 0.5, average='weighted'))
                        elif self.args.nested_cv_metric == 'auc':
                            score_ls.append(roc_auc_score(np.array(y_val_true_tot), np.array(y_val_pred_tot), average='weighted'))
                        else:
                            score_ls.append(np.median(np.array(ids_to_vals(score_ls_)[pixs])))
                        param_ls.append(param_val)
                else:
                    score_ls, param_ls = score_ls_, param_ls_
                best_param = np.array(param_ls)[np.argmax(np.array(score_ls))]
                best_param_file = tb_logger.log_dir + '/hparam_best.txt'
                if os.path.isfile(best_param_file):
                    met = 'a'
                else:
                    met = 'w'
                with open(best_param_file, met) as f:
                    f.write(f'\nFold {fold} best param: {best_param}\n')
                # with open(tb_logger.log_dir + '/hyperparam_tuning.txt',
                for param_name in self.args.param_name:
                    setattr(self.lit_model.args, param_name, best_param)

            test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
            trainer.fit(self.lit_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            torch.save(self.lit_model.model.state_dict(), os.path.join(tb_logger.log_dir,'trained_state_dict.pt'))
            preds = pd.DataFrame(
                {'ixs': test_ixs, 'subj_IDs': self.Y.index.values[test_ixs], 'true': self.lit_model.y_true.cpu(),
                 'preds': self.lit_model.y_preds.cpu()}).set_index('ixs')
            try:
                scores_at_epoch = {k: v[-1].detach().cpu().numpy() for k, v in self.lit_model.scores_dict.items() if
                                   len(v) > 0}
            except:
                try:
                    scores_at_epoch = {k: v[-1].detach() for k, v in self.lit_model.scores_dict.items() if len(v) > 0}
                except:
                    scores_at_epoch = {k: v[-1] for k, v in self.lit_model.scores_dict.items() if len(v) > 0}
            try:
                pd.Series(scores_at_epoch).to_csv(self.output_path + '/scores_at_eval.csv')
            except:
                pass
            preds.to_csv(self.output_path + '/pred_results_f1_{0}'.format(
                np.round(scores_at_epoch['val f1'], 3)).replace('.', '-') + '.csv')

        if self.args.method != 'full_fc':
            if 'basic' in self.args.method:
                rules_dict = plot_joint_results(self.train_dataset_dict, self.y, train_ixs, self.lit_model.logging_dict,
                                   self.output_path, self.args,
                                   self.lit_model, w_maybe_rules=True)
                rules_dict = plot_joint_results(self.train_dataset_dict, self.y, train_ixs, self.lit_model.logging_dict,
                                   self.output_path, self.args,
                                   self.lit_model, w_maybe_rules=False)
            else:
                rules_dict = plot_joint_results_nn(self.train_dataset_dict, self.y, train_ixs, self.lit_model.logging_dict,
                                      self.output_path, self.args,
                                      self.lit_model)


        if len(rules_dict)==0:
            self.args.metabs_p_d += 0.05
            self.args.otus_p_d += 0.05
            self.overwrite_previous = True
            self.train_loop(dataset_dict, train_ixs, test_ixs, fold)
            if os.path.isfile(self.outpath +'/'+ sts + '/logfile.txt'):
                with open(self.outpath +'/'+ sts + '/logfile.txt','a') as f:
                    f.write(f"RETRAINING WITH {self.args.metabs_p_d}\n")
            else:
                with open(self.outpath +'/'+ sts + '/logfile.txt','w') as f:
                    f.write(f"RETRAINING WITH {self.args.metabs_p_d}\n")

        elif len(rules_dict)>20:
            if self.args.metabs_p_d>0.01:
                self.args.metabs_p_d -= 0.01
                self.args.otus_p_d -= 0.01
            else:
                self.args.metabs_p_d = self.args.metabs_p_d/2
                self.args.otus_p_d = self.args.otus_p_d / 2
            self.overwrite_previous = True
            self.train_loop(dataset_dict, train_ixs, test_ixs, fold)
            if os.path.isfile(self.outpath +'/'+ sts + '/logfile.txt'):
                with open(self.outpath +'/'+ sts + '/logfile.txt','a') as f:
                    f.write(f"len(rules_dict) {len(rules_dict)}\n")
                    f.write(f"RETRAINING WITH {self.args.metabs_p_d}\n")
            else:
                with open(self.outpath +'/'+ sts + '/logfile.txt', 'w') as f:
                    f.write(f"len(rules_dict) {len(rules_dict)}\n")
                    f.write(f"RETRAINING WITH {self.args.metabs_p_d}\n")
        # elif [(r['type']=='otus')]
        save_input_data(self.lit_model, train_dataset_dict, test_dataset_dict, self.args,
                        self.outpath + f'seed_{self.args.seed}/')
        with open(tb_logger.log_dir + '/running_loss_dict.pkl', 'wb') as f:
            pkl.dump(self.lit_model.running_loss_dict, f)

        if self.args.method!='full_fc':
            plot_heatmaps(self.lit_model, train_dataset_dict, tb_logger.log_dir, test_dataset_dict=test_dataset_dict)
        # if self.args.remote == 0 and (self.args.seed == 0 or self.args.plot_all_seeds == 1 or self.args.seed==1):
        if self.args.seed<10 and (self.args.plot_all_seeds==1 or self.args.seed ==0 or self.args.seed ==1):
            save_and_plot_post_training(self.lit_model, train_ixs, test_ixs, tb_logger.log_dir,
                                        plot_traces=self.args.plot_traces == 1, best_epoch=best_epoch)



        outer_time = time.time() - outer_start
        with open(self.outpath + f"seed_{self.args.seed}/time.txt", "a") as f:
            f.write(f"{np.round(outer_time / 60, 3)} minutes for fold {fold}\n")

        # with open(tb_logger.log_dir+'/param_dict.pkl','wb') as f:
        #     pkl.dump(self.lit_model.logging_dict, f)

        return preds


def ids_to_vals(ids):
    if isinstance(ids, ray.ObjectID):
        ids = ray.get(ids)
    if isinstance(ids, ray.ObjectID):
        return ids_to_vals(ids)
    if isinstance(ids, list):
        results = []
        for id in ids:
            results.append(ids_to_vals(id))
        return results
    if isinstance(ids,tuple):
        ids = list(ids)
        results = []
        for id in ids:
            results.append(ids_to_vals(id))
        results=tuple(results)
        return results
    return ids


def check_inputs_for_eval(OUTPUT_PATH, args_dict):
    saved_args_path = f'{OUTPUT_PATH}/seed_{args_dict["seed"]}/commandline_args_eval.txt'
    print('seed', args_dict['seed'])
    if not os.path.isfile(saved_args_path):
        ValueError('ERROR: EVALUTATION PARAMETERS NOT FOUND! Make sure to train model before evaluation')
    with open(saved_args_path, 'r') as f:
        saved_args = json.load(f)
    for k, v in args_dict.items():
        if k == 'cv_type' or k == 'parallel' or k == 'seed' or k == 'run_name' or k == 'out_path':
            continue
        if k not in saved_args:
            print(f'Warning: {k} not in saved argument')
        else:
            if saved_args[k] != v:
                print(
                    f'WARNING: value of argument {k} is {saved_args[k]} in saved arguments, but {v} in new arguments!')
                args_dict[k] = saved_args[k]


# @profile
def run_training_with_folds(args, OUTPUT_PATH=''):
    st = time.time()
    seed_everything(args.seed, workers=True)
    # torch.use_deterministic_algorithms(True)
    dataset_dict = {}
    if args.data_met is not None and 'metabs' in args.dtype:
        import pickle as pkl
        print(args.data_met)
        dataset_dict['metabs'] = pd.read_pickle(args.data_met)
        # with open(args.data_met, 'rb') as f:
        #     dataset_dict['metabs'] = pkl.load(f)

        if not isinstance(dataset_dict['metabs']['distances'], pd.DataFrame) and \
                dataset_dict['metabs']['distances'].shape[0] == dataset_dict['metabs']['X'].shape[1]:
            dataset_dict['metabs']['distances'] = pd.DataFrame(dataset_dict['metabs']['distances'],
                                                               index=dataset_dict['metabs']['X'].columns.values,
                                                               columns=dataset_dict['metabs']['X'].columns.values)
        if args.only_mets_w_emb == 1:
            mets = dataset_dict['metabs']['distances'].columns.values
            dataset_dict['metabs']['X'] = dataset_dict['metabs']['X'][mets]

        dataset_dict['metabs']['variable_names'] = dataset_dict['metabs']['X'].columns.values
        data_path = '/'.join(args.data_met.split('/')[:-1])
        if 'taxonomy' not in dataset_dict['metabs'].keys():
            if 'tmp' in os.listdir(data_path):
                if 'classy_fire_df.csv' in os.listdir(data_path + '/tmp/'):
                    classifications = pd.read_csv(data_path + '/tmp/classy_fire_df.csv', index_col=0)
                else:
                    classifications = pd.read_csv('inputs/classy_fire_df.csv', index_col=0)
            else:
                classifications = pd.read_csv('inputs/classy_fire_df.csv', index_col=0)
            dataset_dict['metabs']['taxonomy'] = classifications.loc['subclass']

    if args.data_otu is not None and 'otus' in args.dtype:
        import pickle as pkl
        print(args.data_otu)
        # with open(args.data_otu, 'rb') as f:
        #     dataset_dict['otus'] = pkl.load(f)
        dataset_dict['otus'] = pd.read_pickle(args.data_otu)
        if args.only_otus_w_emb == 1:
            otus = dataset_dict['otus']['distances'].columns.values
            dataset_dict['otus']['X'] = dataset_dict['otus']['X'][otus]
            # dataset_dict['otus']['X']=pd.DataFrame((dataset_dict['otus']['X'].values.T/dataset_dict['otus']['X'].values.sum(1).T).T,
            #                                        index= dataset_dict['otus']['X'].index.values, columns = dataset_dict['otus']['X'].columns.values)
            assert(dataset_dict['otus']['distances'].shape[0]==dataset_dict['otus']['distances'].shape[1])

        dataset_dict['otus']['variable_names'] = dataset_dict['otus']['X'].columns.values

        # dataset_dict['otus']['X'] = dataset_dict['otus']['X'].divide(dataset_dict['otus']['X'].sum(1),axis='index')

    dataset_dict, y = merge_datasets(dataset_dict)

    for key in dataset_dict.keys():
        assert ((y.index.values == dataset_dict[key]['y'].index.values).all())
        assert ((y.index.values == dataset_dict[key]['X'].index.values).all())
    if isinstance(y, np.ndarray):
        Y = pd.Series(y)
    else:
        Y = copy.deepcopy(y)
        y = y.values

    # dist_dict = {k:dataset_dict[k]['distances'] for k in dataset_dict.keys()}
    # num_feat_dict = {k:dataset_dict[k]['X'].shape[1] for k in dataset_dict.keys()}
    # parser, dist_dict, dir, num_feat_dict, learn_embeddings = False)
    os.makedirs(OUTPUT_PATH + f'seed_{args.seed}', exist_ok=True)
    with open(OUTPUT_PATH + f'seed_{args.seed}/dataset_used.pkl', 'wb') as f:
        pkl.dump(dataset_dict, f)
    # cv_trainer = CVTrainer.remote(args, OUTPUT_PATH, Y)
    Y.to_csv(OUTPUT_PATH + f'seed_{args.seed}/y_after_merge.csv')
    if args.cv_type == 'kfold':
        if np.sum(y) / 2 < args.kfolds:
            args.kfolds = int(np.sum(y) / 2)
            print(f"{args.kfolds}-fold cross validation due to only {np.sum(y)} case samples")
        elif np.sum(y == 0) / 2 < args.kfolds:
            args.kfolds = int(np.sum(y == 0) / 2)
            print(f"{args.kfolds}-fold cross validation due to only {np.sum(y == 0)} control samples")
        train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y, num_splits=args.kfolds, seed=args.seed)
    elif args.cv_type == 'loo':
        train_ixs, test_ixs = cv_loo_splits(np.zeros(y.shape[0]), y)
    elif args.cv_type == 'one':
        # train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y, num_splits=args.kfolds, seed=args.seed)
        # train_ixs, test_ixs = [train_ixs[0]], [test_ixs[0]]
        train_ixs, test_ixs = [np.arange(y.shape[0])], [np.arange(y.shape[0])]
    elif args.cv_type == 'eval':
        train_ixs, test_ixs = [np.arange(y.shape[0])], [np.arange(y.shape[0])]
        args.parallel = 1
    else:
        print("Please enter valid option for cv_type. Options are: 'kfold','loo','one'")
        return

    folds = list(range(len(train_ixs)))
    if args.cv_type == 'eval' or args.cv_type=='one':
        folds = ['EVAL']
    else:
        folds.append('EVAL')
        train_ixs.append(np.arange(y.shape[0]))
        test_ixs.append(np.arange(y.shape[0]))

    rem_folds = []
    for fi in range(len(train_ixs)):
        if args.cv_type != 'loo' and (len(np.unique(y[train_ixs[fi]])) == 1 or len(np.unique(y[test_ixs[fi]])) == 1):
            print(
                f'FOLD {fi} REMOVED; {len(np.unique(y[train_ixs[fi]]))} train classes, {len(np.unique(y[test_ixs[fi]]))} test classes')
            rem_folds.append(fi)
    if len(rem_folds) > 0:
        for fi in rem_folds:
            folds.pop(fi)
            train_ixs.pop(fi)
            test_ixs.pop(fi)

    # ray.shutdown()
    # ray.init(ignore_reinit_error=True, runtime_env={"working_dir": os.getcwd(),
    #                       "py_modules": ["../utilities/"]})

    if global_args.use_ray or args.parallel<2:
        preds = []
        for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs):
            print('FOLD {0}'.format(fold))
            if global_args.use_ray:
                cv_trainer = CVTrainer.remote(args, OUTPUT_PATH, Y)
                ckpt_preds = cv_trainer.train_loop.remote(dataset_dict, train_idx, test_idx, fold)
            else:
                cv_trainer = CVTrainer(args, OUTPUT_PATH, Y)
                ckpt_preds = cv_trainer.train_loop(dataset_dict, train_idx, test_idx, fold)
            preds.append(ckpt_preds)
            # args = cv_trainer.args
            # args.n_r = cv_trainer.args.n_r
            # args.metabs_n_d = cv_trainer.args.metabs_n_d
            # args.otus_n_d = cv_trainer.args.otus_n_d
            # if args.use_ray:
        # preds = ray.get(preds)
        if global_args.use_ray:
            preds = ids_to_vals(preds)
    else:
        cv_trainer = CVTrainer(args, OUTPUT_PATH, Y)
        preds = Parallel(n_jobs=args.parallel)(delayed(cv_trainer.train_loop)(dataset_dict, train_idx, test_idx, fold)
                                 for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs))


    if len(preds) > 1:
        final_preds = pd.concat(preds[:-1])
        f1 = f1_score(final_preds['true'], final_preds['preds'] > 0.5, average='weighted')
        auc = roc_auc_score(final_preds['true'], final_preds['preds'], average='weighted')
        final_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
                           '/' + 'pred_results_f1_{0}_auc_{1}'.format(
            np.round(f1, 3), np.round(auc, 3)).replace('.', '-') + '.csv')
        print('AUC: {0}'.format(auc))

    if args.cv_type != 'one':
        eval_preds = preds[-1]
        f1 = f1_score(eval_preds['true'], eval_preds['preds'] > 0.5, average='weighted')
        if len(np.unique(eval_preds['true'])) == 1:
            auc = np.nan
        else:
            auc = roc_auc_score(eval_preds['true'], eval_preds['preds'], average='weighted')
        if 'EVAL' in os.listdir(OUTPUT_PATH + f'seed_{args.seed}'):
            eval_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
                              '/EVAL/' + 'pred_results_f1_{0}_auc_{1}'.format(
                np.round(f1, 3), np.round(auc, 3)).replace('.', '-') + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    args, parser = parse(parser)

    if not os.path.isabs(args.out_path):
        args.out_path = os.getcwd() + '/' + args.out_path

    print("OUT PATH:", args.out_path)
    if args.cv_type == 'eval':
        check_inputs_for_eval(args.out_path + '/' + args.run_name + '/', args.__dict__)

    if args.method == 'fc' or args.method == 'full_fc':
        from models_fc import ComboMDITRE
    elif 'basic' in args.method:
        from models import ComboMDITRE
    else:
        ValueError(
            'Warning: accepted method not provided. Choices are: "basic", "fc", "nam", or "full_fc". Default "basic" will be used.')
        # from models import ComboMDITRE

    # ray.init()
    seed_everything_custom(args.seed)
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    # if './datasets/' not in args.data:
    #     args.data = './datasets/cdi/' + args.data
    # with open(os.path.join(args.out_path, 'total_time.txt','w'))
    st = time.time()
    run_training_with_folds(args, OUTPUT_PATH=args.out_path + '/' + args.run_name + '/')
    et = time.time() - st
    print(f"TRAINING {args.epochs} TOOK {np.round(et / 60, 3)} MINUTES")
    # with open(os.path.join(args.out_path, 'total_time.txt'), 'w') as f:
    #     f.write(f"TRAINING {args.epochs} TOOK {np.round(et / 60, 3)} MINUTES")

