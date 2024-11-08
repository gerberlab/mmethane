import numpy as np
import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append(os.path.abspath(".."))
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from model_helper import CustomBernoulli
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from sklearn.metrics import f1_score, roc_auc_score

from utilities.util import split_and_preprocess_dataset, cv_kfold_splits, merge_datasets, cv_loo_splits
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
# from lightning.pytorch.loggers import CSVLogger
import argparse
import shutil

from viz import *
import json
from loss_time import mapLoss

from joblib import Parallel, delayed
import datetime
# from trainer_submodules_nam import moduleLit
# from model_nam import ComboMDITRE
from trainer_submodules_time import moduleLit
from torch import optim
from lightning.pytorch.callbacks import ModelCheckpoint
# from CustomCheckpoint import *
import pickle as pkl
import lightning.pytorch as pl
# torch.autograd.detect_anomaly()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_float32_matmul_precision('high')
START_TIME=time.time()
from torchmetrics import F1Score, AUROC, Accuracy

# sys.setrecursionlimit(2097152)
# TO DO:
#   - add filtering transforming into function (only filter/transform training data)
#   - fix plot results for two datasets

def parse(parser):
    # Main model specific parameters
    parser.add_argument('--n_r', type=int, default=10, help='Number of rules')
    parser.add_argument('--lr_beta', default=0.005, type=float,
                        help='Initial learning rate for binary concrete logits on rules.', nargs='+')
    parser.add_argument('--lr_alpha', default=0.005, type=float,
                        help='Initial learning rate for binary concrete logits on detectors.', nargs='+')
    parser.add_argument('--min_k_bc', default=1, type=float, help='Min Temperature for binary concretes')
    parser.add_argument('--max_k_bc', default=10, type=float, help='Max Temperature for binary concretes')
    parser.add_argument('--lr_fc', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')
    parser.add_argument('--lr_bias', default=0.001, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')
    parser.add_argument('--w_var', type=float, default=1e5, help='Normal prior variance on weights.')
    parser.add_argument('--z_r_mean', type=float, default=1, help='NBD Mean active rules')
    parser.add_argument('--z_r_var', type=float, default=5, help='NBD variance of active rules')
    parser.add_argument('--z_mean', type=float, default=1,help='NBD Mean active detectors per rule')
    parser.add_argument('--z_var', type=float, default=5,help='NBD variance of active detectors per rule')

    # Metabolite model specific parameters
    parser.add_argument('--metabs_lr_kappa', default=0.0001, type=float,help='Initial learning rate for kappa.', nargs='+')
    parser.add_argument('--metabs_lr_eta', default=0.0001, type=float,help='Initial learning rate for eta.', nargs='+')
    parser.add_argument('--metabs_lr_emb', default=0.001, type=float,help='Initial learning rate for emb.', nargs='+')
    parser.add_argument('--metabs_lr_thresh', default=0.0005, type=float,help='Initial learning rate for threshold.', nargs='+')
    parser.add_argument('--metabs_lr_slope', default=0.00001, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--metabs_lr_time', default=0.01, type=float,
                        help='Initial learning rate for sigma.')
    parser.add_argument('--metabs_min_k_otu', default=100, type=float,help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--metabs_max_k_otu', default=1000, type=float,help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--metabs_min_k_thresh', default=1, type=float,help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--metabs_max_k_thresh', default=10, type=float,help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--metabs_min_k_bc', default=1, type=float,help='Min Temperature for binary concretes')
    parser.add_argument('--metabs_max_k_bc', default=10, type=float,help='Max Temperature for binary concretes')
    parser.add_argument('--metabs_n_d', type=int, default=8, help='Number of detectors')
    parser.add_argument('--metabs_emb_dim', type=float, default=29)
    parser.add_argument('--metabs_multiplier', type=float, default=10)
    parser.add_argument('--metabs_expl_var_cutoff', type=float, default=0.1)
    parser.add_argument('--metabs_use_pca', type=int, default=0)
    parser.add_argument('--metabs_lr_nam', type=float, default=0.001)
    parser.add_argument('--metabs_min_k_time', type=float, default=1)
    parser.add_argument('--metabs_max_k_time', type=float, default=10)
    parser.add_argument('--metabs_min_k_slope', type=float, default=1e3)
    parser.add_argument('--metabs_max_k_slope', type=float, default=1e4)


    # Microbe model specific parameters
    parser.add_argument('--otus_lr_kappa', default=0.001, type=float,help='Initial learning rate for kappa.')
    parser.add_argument('--otus_lr_eta', default=0.001, type=float,help='Initial learning rate for eta.')
    parser.add_argument('--otus_lr_emb', default=0.001, type=float,help='Initial learning rate for emb.')
    parser.add_argument('--otus_lr_thresh', default=0.0001, type=float,help='Initial learning rate for threshold.')
    parser.add_argument('--otus_lr_slope', default=0.00001, type=float,
                        help='Initial learning rate for threshold.')
    parser.add_argument('--otus_lr_time', default=0.01, type=float,
                        help='Initial learning rate for sigma.')
    parser.add_argument('--otus_min_k_otu', default=100, type=float,help='Max Temperature on heavyside logistic for otu selection')
    parser.add_argument('--otus_max_k_otu', default=1000, type=float,help='Min Temperature on heavyside logistic for otu selection')
    parser.add_argument('--otus_min_k_thresh', default=1, type=float,help='Max Temperature on heavyside logistic for threshold')
    parser.add_argument('--otus_max_k_thresh', default=10, type=float,help='Min Temperature on heavyside logistic for threshold')
    parser.add_argument('--otus_min_k_bc', default=1, type=float,help='Min Temperature for binary concretes')
    parser.add_argument('--otus_max_k_bc', default=10, type=float,help='Max Temperature for binary concretes')
    parser.add_argument('--otus_n_d', type=int, default=8, help='Number of detectors')
    parser.add_argument('--otus_emb_dim', type=float, default=19)
    parser.add_argument('--otus_multiplier', type=float, default=1)
    parser.add_argument('--otus_expl_var_cutoff', type=float, default=0.1)
    parser.add_argument('--otus_use_pca', type=int, default=0)
    parser.add_argument('--otus_lr_nam', type=float, default=0.001)
    parser.add_argument('--otus_min_k_time', type=float, default=1)
    parser.add_argument('--otus_max_k_time', type=float, default=10)
    parser.add_argument('--otus_min_k_slope', type=float, default=1e3)
    parser.add_argument('--otus_max_k_slope', type=float, default=1e4)

    # Training Parameters
    parser.add_argument('--data_met', metavar='DIR',
                        help='path to metabolite dataset',
                        default='../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets_xdl.pkl')
    parser.add_argument('--data_otu', metavar='DIR',
                        help='path to otu dataset',
                        default = 'mditre_datasets/david.pkl')
    parser.add_argument('--data_name', type=str,
                        help='Name of the dataset, will be used for log dirname',
                        default = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M'),
                        )
    parser.add_argument('--min_epochs', default=100, type=int, metavar='N',
                        help='number of minimum epochs to run')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', type=int, default=0,
                        help='Set random seed for reproducibility')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'one','None','eval'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--early_stopping', default=0, type=int)
    parser.add_argument('--validate', default=0, type = int)
    parser.add_argument('--test', default=1, type = int)
    # parser.add_argument('--emb_dim', type = float, default=20)
    parser.add_argument('--out_path', type = str, default='/Users/jendawk/logs/mditre-logs/')
    parser.add_argument('--num_anneals', type=float, default=1)
    parser.add_argument('--monitor', type=str, default='train_loss')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--dtype', type=str,
                        default=['otus'],
                        choices=['metabs', 'otus'],
                        help='Choose type of data', nargs='+')
    parser.add_argument('--schedule_lr', type=int, default=0,
                        help='Schedule learning rate')
    parser.add_argument('--parallel', type=int, default=0,
                        help='run in parallel')
    parser.add_argument('--only_mets_w_emb', type=int, default=0, help='whether or not keep only mets with embeddings')
    parser.add_argument('--only_otus_w_emb', type=int, default=1, help='whether or not keep only otus with embeddings')
    parser.add_argument('--learn_emb', type=int, default=0, help='whether or not to learn embeddings')
    parser.add_argument('--debug', type=int, default=0)
    # parser.add_argument('--lr_master',type=float, default=None)
    parser.add_argument('--from_saved_files', type=int, default=0)
    parser.add_argument('--use_k_1', type=int, default=0)
    parser.add_argument('--use_noise', type=int, default=0)
    parser.add_argument('--noise_anneal', type=float, default=[0,0], nargs='+')
    parser.add_argument('--remote', type=int, default=0)
    parser.add_argument('--annealing_limit', type=float, default=[0,1], nargs='+')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--anneal_type', type=str, default='linear',choices=['linear','cosine','exp'])
    parser.add_argument('--div_loss', type=int, default=0)
    parser.add_argument('--p_d', type=float, default=0.5)
    parser.add_argument('--p_r', type=float, default=0.5)
    parser.add_argument('--old', type=int, default=0)
    parser.add_argument('--neg_bin_prior', type=int, default=0)
    parser.add_argument('--bernoulli_prior', type=int, default=1)
    parser.add_argument('--kappa_eta_prior', type=int, default=1)
    parser.add_argument('--hard_otu', type=int, default=1)
    parser.add_argument('--hard_bc', type=int, default=1)
    parser.add_argument('--filter_data', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--kappa_prior', type=str, default='log-normal', choices=['log-normal','trunc-normal'])
    parser.add_argument('--use_old_refs', type=int, default=1)
    parser.add_argument('--z_loss_mult', type=float, default=1)
    parser.add_argument('--param_grid', nargs='+', type=float, default=None)
                        # default=[0.1,0.2,0.3,0.4,0.5,0.5,0.7,0.8,0.9])
                        # default=[1,2,5,7,10,12,15,17,20])
    parser.add_argument('--param_name', type=str, nargs='+', default=['p_d', 'p_r'])
    parser.add_argument('--nested_cv', type=int, default=0)
    parser.add_argument('--nested_cv_metric', type=str, default='f1')
    parser.add_argument('--adj_pd', type=int, default=1)
    parser.add_argument('--h_sizes', type=int, nargs='+', default=[0,0])
    parser.add_argument('--method', type=str, default='basic', choices=['basic','fc','nam','full_fc','nam_orig'])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--plot_traces', type=int, default=1)
    parser.add_argument('--time', type=int, default=1)
    # parser.add_argument('--full_fc', type=int, default=1)
    args,_ = parser.parse_known_args()

    args,_ = parser.parse_known_args()

    return args, parser
class LitMDITRE(pl.LightningModule):
    # @profile
    def __init__(self,args,data_dict, dir, learn_embeddings=False):
        super().__init__()
        # self.save_hyperparameters()
        self.dir=dir
        self.args = args
        self.data_dict = data_dict
        self.learn_embeddings = learn_embeddings
        self.noise_factor = 1.
        self.train_preds,self.val_preds,self.test_preds=[],[],[]
        self.train_true, self.val_true,self.test_true=[],[],[]
        self.F1Score = F1Score(task='binary')
        self.AUROC=AUROC(task='binary')
        self.Accuracy = Accuracy(task='binary')

    def parse_args(self):
        # if self.args.fc==1:
        #     from model_fc import ComboMDITRE
        # else:
        #     from model_nam import ComboMDITRE
        self.k_dict = {'k_beta': {'max': self.args.max_k_bc, 'min': self.args.min_k_bc},
                       'k_alpha': {'max': self.args.max_k_bc, 'min': self.args.min_k_bc}}
        self.class_dict = {}
        self.model_dict = {}

        if not isinstance(self.args.dtype, list):
            self.args.dtype = [self.args.dtype]
        self.loss_params=[]
        self.num_detectors = 0
        self.n_d_per_class={}
        for type in self.args.dtype:
            if type != 'metabs':
                self.learn_embeddings=False
            self.class_dict[type] = moduleLit(self.args, copy.deepcopy(self.data_dict[type]), type,
                                              dir=self.dir, learn_embeddings=self.learn_embeddings, device=device)
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

        self.logging_dict = {b: [] for b,a in self.model.named_parameters()}
        for type in self.args.dtype:
            log_dict = {type + '_' + key: value for key, value in self.class_dict[type].logging_dict.items()}
            self.logging_dict.update(log_dict)
                #         self.logging_dict[name + '_' + b].append(a.detach().clone())
                # self.logging_dict.update({type + '_' + key: [] for key,value in self.model.rules[type]})
        self.scores_dict = {'train f1': [], 'test f1': [], 'val f1': [], 'train auc': [], 'test auc': [],
                            'train loss': [], 'test loss': [], 'val loss': [], 'val auc': [], 'total val loss': [],
                            'train acc 0':[], 'train acc 1':[], 'val acc 0':[], 'val acc 1':[], 'test acc 0':[], 'test acc 1':[]}
        # self.loss_func = mapLoss(self.model, self.args, self.normal_wts, self.bernoulli_rules, self.bernoulli_det)
        self.loss_func = mapLoss(self.model, self.args, self.normal_wts, self.n_detectors_prior, device = device, n_r_prior = self.n_rules_prior)
        self.running_loss_dict={}
        self.grad_dict={}
        for b, a in self.model.named_parameters():
            # print(b, a.get_device())
            self.grad_dict[b]=[]
        for name, model in self.model_dict.items():
            for b, a in model.named_parameters():
                # print(b, a.get_device())
                self.grad_dict[name + '_' + b]=[]



    def get_loss(self, outputs, labels, kdict):
        loss_, reg_loss, loss_dict = self.loss_func.loss(outputs, labels, kdict['k_alpha'], kdict['k_beta'])
        loss = loss_.clone()
        loss += reg_loss
        for name, module in self.class_dict.items():
            reg_l, ldict = module.loss_func.loss()
            loss += reg_l
            loss_dict.update(ldict)
            # if self.args.div_loss == 1:
            #     div_loss = diversity_loss(self.model.z_r, self.model.z_d, module.model.wts)
            #     loss_dict[name + '_diversity_loss'] = div_loss
            #     loss += div_loss
        return loss, loss_dict


    def set_model_hparams(self):
        self.num_rules = self.args.n_r
        self.wts_mean = 0
        mean = torch.ones(self.num_rules, dtype=torch.float, device=device) * self.wts_mean
        cov = torch.eye(self.num_rules, dtype=torch.float, device=device) * self.args.w_var
        self.normal_wts = MultivariateNormal(mean, cov)

        # Set mean and variance for neg bin prior for rules
        if self.args.neg_bin_prior==1:
            self.n_rules_prior = create_negbin(self.args.z_r_mean, self.args.z_r_var, device=device)
            self.n_detectors_prior = create_negbin(self.args.z_mean, self.args.z_var, device=device)
        else:
            self.n_rules_prior = CustomBernoulli(self.args.p_r, device)
            self.n_detectors_prior = CustomBernoulli(self.args.p_d, device)

        self.alpha_bc = BinaryConcrete(loc=1, tau=1 / self.k_dict['k_alpha']['min'])
        self.beta_bc = BinaryConcrete(loc=1, tau=1 / self.k_dict['k_alpha']['min'])

    def set_init_params(self):
        beta_init = np.random.normal(0, 0.1, (self.num_rules))
        w_init = np.random.normal(0, 1, (1, self.num_rules))
        bias_init = np.zeros((1))

        self.init_args = {
            'w_init': w_init,
            'bias_init': bias_init,
            'beta_init': beta_init,
        }
        if self.args.method!='basic':
            self.init_args['alpha_init'] = np.random.normal(0, 1, (self.num_detectors)) * 1e-3
        else:
            self.init_args['alpha_init'] = np.random.normal(0, 1, (self.num_rules, self.num_detectors)) * 1e-3

    def forward(self, x_dict, hard_otu, hard_bc, noise_factor):
        return self.model(x_dict, self.k_step, hard_otu=hard_otu, hard_bc=hard_bc, noise_factor=noise_factor)


    def training_step(self, batch, batch_idx, optimizer_idx=None):

        for mclass in self.class_dict.values():
            if self.args.method=='basic':
                mclass.model.thresh.data.clamp_(mclass.thresh_min, mclass.thresh_max)
            if self.args.kappa_prior == 'trunc-normal':
                mclass.model.kappa.data.clamp_(mclass.kappa_min, mclass.kappa_max)
        xdict, y = batch


        if self.args.use_noise==1:
            if self.args.noise_anneal[0]!=self.args.noise_anneal[1]:
                self.noise_factor = linear_anneal(self.args.epochs - self.current_epoch, self.args.noise_anneal[1],
                                                  self.args.noise_anneal[0], self.args.epochs,0, self.args.noise_anneal[0])
            else:
                self.noise_factor=1.
        else:
            self.noise_factor=0.

        for k_param, vals in self.k_dict.items():
            if self.current_epoch < int(self.args.annealing_limit[1]*self.args.epochs) and \
                    self.current_epoch > int(self.args.annealing_limit[0]*self.args.epochs):
                anneal_start = self.current_epoch- int(self.args.annealing_limit[0]*self.args.epochs)
                anneal_end = int(self.args.annealing_limit[1]*self.args.epochs)
                anneal_steps = int(self.args.annealing_limit[0] * self.args.epochs)
                if vals['min']==vals['max']:
                    self.k_step[k_param] = vals['min']
                else:
                    if self.args.anneal_type=='linear':
                        self.k_step[k_param] = linear_anneal(anneal_start,vals['max'], vals['min'],anneal_end,anneal_steps,vals['min'])
                    elif self.args.anneal_type=='exp':
                        self.k_step[k_param] = exp_anneal(anneal_start,anneal_steps,anneal_end,vals['min'],vals['max'])
                    else:
                        self.k_step[k_param] = cosine_annealing(anneal_start,vals['min'],vals['max'],
                                                                int(self.args.epochs*(self.args.annealing_limit[1]-self.args.annealing_limit[0])))
        if self.current_epoch%1000==0:
            print('\nEpoch ' + str(self.current_epoch))
            print(f'{self.current_epoch} epochs took {time.time() - START_TIME} seconds')

        y_hat = self(xdict, hard_otu=args.hard_otu==1, hard_bc=args.hard_bc==1, noise_factor=self.noise_factor)
        self.loss, self.loss_dict = self.get_loss(y_hat, y, self.k_step)

        self.train_preds.extend(y_hat.sigmoid())
        self.train_true.extend(y)

        return self.loss


    def on_train_epoch_end(self):
        for b, a in self.model.named_parameters():
            self.logging_dict[b].append(a.clone())

        for name, model in self.model_dict.items():
            for b, a in model.named_parameters():
                self.logging_dict[name + '_' +b].append(a.clone())
            if self.args.method=='nam':
                if 'mean_non_zeros' not in self.logging_dict.keys():
                    self.logging_dict['mean_non_zeros'] = [model.non_zeros_mean]
                else:
                    self.logging_dict['mean_non_zeros'].append(model.non_zeros_mean)
                if 'median_non_zeros' not in self.logging_dict.keys():
                    self.logging_dict['median_non_zeros'] = [model.non_zeros_median]
                else:
                    self.logging_dict['median_non_zeros'].append(model.non_zeros_median)
        for key in self.loss_dict.keys():
            if key not in self.running_loss_dict.keys():
                self.running_loss_dict[key]=[]
            self.running_loss_dict[key].append(self.loss_dict[key])

        y = torch.stack(self.train_true,0)
        y_hat = torch.stack(self.train_preds,0)
        f1 = self.F1Score(y_hat > 0.5, y)
        try:
            auc = self.AUROC(y_hat, y)
        except:
            print('ERROR: ONLY 1 CLASS IN AUC! (TRAIN)')
            print('y has length={0}'.format(len(y)))
            print(y)
            auc = np.nan
        
        ctrls, case = y==0, y!=0
        acc_0 = self.Accuracy(y_hat[ctrls] > 0.5, y[ctrls])
        acc_1 = self.Accuracy(y_hat[case] > 0.5, y[case])
        self.log('acc 0', acc_0, logger=False)
        self.log('acc 1', acc_1, logger=False)
        self.log('total train loss', self.loss, logger=False)
        self.log('train f1', f1, logger=False)
        self.log('train auc', auc, logger=False)
        self.log('train loss', self.loss_dict['train_loss'], logger=False)
        self.scores_dict['train auc'].append(auc)
        self.scores_dict['train f1'].append(f1)
        self.scores_dict['train loss'].append(self.loss)
        self.scores_dict['train acc 0'].append(acc_0)
        self.scores_dict['train acc 1'].append(acc_1)
        self.train_true,self.train_preds=[],[]
        return (None)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, hard_otu=args.hard_otu==1, hard_bc=args.hard_bc==1, noise_factor=self.noise_factor)
        self.val_loss, self.val_loss_dict = self.get_loss(y_hat, y, self.k_step)
        self.val_preds.extend(y_hat.sigmoid())
        self.val_true.extend(y)
        return self.val_loss

    def on_validation_epoch_end(self):
        y,y_hat = torch.stack(self.val_true,0),torch.stack(self.val_preds,0)
        self.y_preds =y_hat
        self.y_true= y
        f1 = self.F1Score(y_hat > 0.5,y)
        try:
            f1 = self.F1Score(y_hat > 0.5,y)
            auc = self.AUROC(y_hat,y)
            ctrls, case = y==0, y!=0
            acc_0 = self.Accuracy(y_hat[ctrls] > 0.5, y[ctrls])
            acc_1 = self.Accuracy(y_hat[case] > 0.5, y[case])
            self.log('val acc 0', acc_0)
            self.log('val acc 1', acc_1)
            self.log('val auc', auc)
            self.log('val f1', f1)
        except:
            f1, auc, acc_0, acc_1 = 0,0,0,0
        
        self.log('total val loss', self.val_loss)
        #
        self.log('val loss',self.val_loss_dict['train_loss'])

        if len(self.scores_dict['train loss'])>0:
            self.scores_dict['val f1'].append(f1)
            self.scores_dict['val auc'].append(auc)
            self.scores_dict['total val loss'].append(self.val_loss)
            self.scores_dict['val loss'].append(self.val_loss_dict['train_loss'])
            self.scores_dict['val acc 0'].append(acc_0)
            self.scores_dict['val acc 1'].append(acc_1)

        self.y_hat_val, self.y_val = y_hat, y
        self.val_true, self.val_preds = [],[]
        return (None)
        # return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, hard_otu=args.hard_otu==1, hard_bc=args.hard_bc==1, noise_factor=0)
        self.test_preds.extend(y_hat.sigmoid())
        self.test_true.extend(y)
        self.test_loss, _ = self.get_loss(y_hat, y, self.k_step)
        return self.test_loss

    def on_test_epoch_end(self):
        y, y_hat = torch.stack(self.test_true,0), torch.stack(self.test_preds,0)
        f1 = self.F1Score(y_hat > 0.5, y)
        ctrls, case = y==0, y!=0
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
        self.y_preds =y_hat
        self.y_true= y

        self.test_true, self.test_preds = [],[]
        return (None)


    def configure_optimizers(self):
        lr_ls=[]
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
            elif len(name.split('.'))>1:
                ix = int(name.split('.')[1])
                nm = name.split('.')[2].split('_')[0]
                dtype = self.model.module_names[ix]
                lr = self.args.__dict__[dtype + '_' + 'lr_' + nm]
                if isinstance(lr, list):
                    lr = lr[0]
                lr_ls.append({'lr': lr, 'params': [param]})
            else:
                lr_ls.append({'params':[param]})
        self.optimizer_0 = optim.Adam(lr_ls, weight_decay=self.args.weight_decay)
        # self.optimizer_0 = optim.RMSprop(lr_ls, weight_decay=self.args.weight_decay)
        if args.schedule_lr == 1:
            self.scheduler_0 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_0,
                                                                   int(self.args.epochs/self.args.num_anneals))
                                                                   # eta_min=1e-5)
            return [self.optimizer_0],[self.scheduler_0]
        else:
            return [self.optimizer_0]


class mditreDataset(Dataset):
    def __init__(self, dataset_dict):

        self.x_dict = {}
        for name, dataset in dataset_dict.items():
            x = dataset['X']
            if isinstance(x, pd.DataFrame):
                x = x.values
            if 'X_mask' in dataset.keys():
                self.x_dict[name] = torch.tensor(x, device=device, dtype=torch.float), torch.tensor(dataset['X_mask'], device=device, dtype=torch.float)
            else:
                self.x_dict[name] = torch.tensor(x, device=device, dtype=torch.float)

        y = dataset['y']
        if isinstance(y, pd.Series):
            y = y.values

        self.y = torch.tensor(y, device=device, dtype=torch.float)
        # self.idxs = idxs

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {name: (self.x_dict[name][0][idx],self.x_dict[name][1][idx]) for name in self.x_dict.keys()}, self.y[idx]


class CVTrainer():
    def __init__(self, args, OUTPUT_PATH, y):
        self.args = args
        self.outpath = OUTPUT_PATH
        self.y = y

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

        out = trainer.test(model=model, ckpt_path = ckpt_path, dataloaders = test_loader)
        preds_df = pd.DataFrame({'ixs': test_ixs, 'true': model.y_true.cpu(), 'preds': model.y_preds.cpu()}).set_index('ixs')
        if ckpt_path[0]=='.':
            out_path = '.' + ckpt_path.split('.')[1]
        else:
            out_path = ckpt_path.split('.')[0]
        print('ckpt_path: ' + ckpt_path)
        print('outpath: ' + out_path)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        if 'epoch=' in ckpt_path:
            epoch = int(re.findall('epoch=\d*',ckpt_path)[0].split('epoch=')[-1])
        else:
            epoch=-1
        try:
            scores_at_epoch = {k:v[-1].detach().cpu().numpy() for k,v in model.scores_dict.items() if len(v)>0}
        except:
            try:
                scores_at_epoch = {k: model.scores_dict[k][epoch] for k in model.scores_dict.keys() if 'test' not in k}
            except:
                scores_at_epoch = {}
                for k in model.scores_dict.keys():
                    if 'test' not in k and 'val' not in k:
                        scores_at_epoch[k] = model.scores_dict[k][epoch]


        for k in scores_at_epoch.keys():
            try:
                scores_at_epoch[k] = scores_at_epoch[k].detach().cpu().numpy()
            except:
                continue
        pd.Series(scores_at_epoch).to_csv(self.output_path + '/scores_at_eval.csv')
        try:
            preds_df.to_csv(self.output_path + '/pred_results_f1_{0}'.format(
                np.round(scores_at_epoch['val f1'], 3)).replace('.','-') + '.csv')
        except:
            preds_df.to_csv(self.output_path + '/pred_results.csv')
        return preds_df


    # @profile
    def train_loop(self, dataset_dict, train_ixs, test_ixs, fold):
        self.train_ixs = train_ixs
        self.test_ixs = test_ixs
        if self.args.batch_size is None:
            self.batch_size=len(train_ixs)+100
        else:
            self.batch_size = self.args.batch_size

        if isinstance(fold, str):
            vers = fold
        else:
            vers = f'fold_{fold}'
        tb_logger = TensorBoardLogger(save_dir=self.outpath, name=f'seed_{self.args.seed}', version=vers)
        self.output_path = tb_logger.log_dir
        pred_ls = self.check_if_fold_already_finished(fold, tb_logger.log_dir)
        if pred_ls is not None:
            return pred_ls

        monitor = args.monitor.replace('_',' ')
        callbacks = [ModelCheckpoint(save_last=True,
                dirpath=tb_logger.log_dir,
                save_top_k=1,
                verbose=False,
                monitor=monitor,
                every_n_epochs=None,every_n_train_steps=None,train_time_interval=None,
                mode='min' if 'loss' in monitor else 'max',
                filename='{epoch}' + '-{' + monitor + ':.2f}')
            ]
        if args.early_stopping == 1:
            callbacks.extend([EarlyStopping(monitor=monitor, patience=200)])
        # else:
            # callbacks.append(LearningRateMonitor())
        trainer = pl.Trainer(logger=tb_logger, max_epochs = self.args.epochs, min_epochs = self.args.min_epochs,
                         callbacks=callbacks, 
                         log_every_n_steps=50,
                         enable_progress_bar=False,
                             )

        if self.args.filter_data:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=True, standardize_otus=self.args.method!='basic')
        else:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=False)
            if 'otus' in dataset_dict.keys():
                dataset_dict['otus']['X'] = dataset_dict['otus']['X'].divide(dataset_dict['otus']['X'].sum(1),
                                                                             axis='index')
        train_loader = DataLoader(mditreDataset(train_dataset_dict), batch_size=self.batch_size,
                                  shuffle=True)

        self.lit_model = LitMDITRE(self.args, train_dataset_dict,
                                   dir=self.outpath + f'seed_{args.seed}',
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

        if self.args.validate==1:
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
            test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
            # self, model, trainer, test_ixs, ckpt_path, test_loader
            path = self.outpath + f'seed_{args.seed}' + '/' +vers + '/'
            tmp = [f for f in os.listdir(path) if ('.ckpt' in f and 'epoch' in f)][0]
            ckpt_path = path + tmp
            preds = self.test_model(self.lit_model, trainer, self.test_ixs, ckpt_path, test_loader)
            
        else:
            test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
            trainer.fit(self.lit_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            preds = pd.DataFrame({'ixs': test_ixs, 'true': self.lit_model.y_true.cpu(), 'preds': self.lit_model.y_preds.cpu()}).set_index('ixs')
            try:
                scores_at_epoch = {k:v[-1].detach().cpu().numpy() for k,v in self.lit_model.scores_dict.items() if len(v)>0}
            except:
                try:
                    scores_at_epoch = {k:v[-1].detach() for k,v in self.lit_model.scores_dict.items() if len(v)>0}
                except:
                    scores_at_epoch = {k:v[-1] for k,v in self.lit_model.scores_dict.items() if len(v)>0}
            try:
                pd.Series(scores_at_epoch).to_csv(self.output_path + '/scores_at_eval.csv')
            except:
                pass
            preds.to_csv(self.output_path + '/pred_results_f1_{0}'.format(
                np.round(scores_at_epoch['val f1'], 3)).replace('.','-') + '.csv')
        if self.args.method!='full_fc':
            if self.args.method=='basic':
                plot_joint_results(self.train_dataset_dict, self.y, train_ixs, self.lit_model.logging_dict, self.output_path, parser,
                                    self.lit_model)
            else:
                plot_joint_results_nn(self.train_dataset_dict, self.y, train_ixs, self.lit_model.logging_dict, self.output_path, parser,
                    self.lit_model)


        if self.args.remote == 0 and self.args.seed==0:
            save_input_data(self.lit_model, train_dataset_dict, test_dataset_dict, self.args, self.outpath + f'seed_{self.args.seed}/')
            save_and_plot_post_training(self.lit_model, train_ixs, test_ixs, tb_logger.log_dir, plot_traces = self.args.plot_traces==1)

        # test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
        # model, train_ixs, test_ixs, make_outputs=True
        # preds = self.test_model(self.lit_model, self.train_ixs, self.test_ixs, make_outputs=True)

        # path = self.outpath + f'seed_{args.seed}' + '/' +vers + '/'
        # files = [f for f in os.listdir(path) if '.ckpt' in f]
        # preds_ls = {}
        # for file in files:
        #     ckpt_path = path + file
        #     if isinstance(fold, str) or self.args.remote==0:
        #         plot_stuff=True
        #     else:
        #         plot_stuff=False
        #     preds = self.test_model(self.lit_model, trainer, self.train_ixs, self.test_ixs, test_loader, ckpt_path = ckpt_path, make_outputs=plot_stuff)
        #     if 'last' in ckpt_path.split('/')[-1]:
        #         ckpt_name = 'last'
        #     else:
        #         ckpt_name = 'best'
        #     # os.remove(ckpt_path)
        #     preds_ls[ckpt_name] = preds

        return preds

# @profile
def run_training_with_folds(args, OUTPUT_PATH = ''):
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

        if not isinstance(dataset_dict['metabs']['distances'], pd.DataFrame) and dataset_dict['metabs']['distances'].shape[0]==dataset_dict['metabs']['X'].shape[1]:
            dataset_dict['metabs']['distances'] = pd.DataFrame(dataset_dict['metabs']['distances'],
                                                                   index=dataset_dict['metabs']['X'].columns.values,
                                                                   columns=dataset_dict['metabs']['X'].columns.values)
        if args.only_mets_w_emb==1:
            mets=dataset_dict['metabs']['distances'].columns.values
            dataset_dict['metabs']['X'] = dataset_dict['metabs']['X'][mets]

        dataset_dict['metabs']['variable_names'] = dataset_dict['metabs']['X'].columns.values
        data_path = '/'.join(args.data_met.split('/')[:-1])
        if 'taxonomy' not in dataset_dict['metabs'].keys():
            if 'tmp' in os.listdir(data_path):
                if 'classy_fire_df.csv' in os.listdir(data_path + '/tmp/'):
                    classifications = pd.read_csv(data_path + '/tmp/classy_fire_df.csv', index_col=0)
                else:
                    classifications = pd.read_csv('inputs/classy_fire_df.csv', index_col = 0)
            else:
                classifications = pd.read_csv('inputs/classy_fire_df.csv', index_col = 0)
            dataset_dict['metabs']['taxonomy'] = classifications.loc['subclass']

    if args.data_otu is not None and 'otus' in args.dtype:
        import pickle as pkl
        print(args.data_otu)
        # with open(args.data_otu, 'rb') as f:
        #     dataset_dict['otus'] = pkl.load(f)
        dataset_dict['otus'] = pd.read_pickle(args.data_otu)
        if isinstance(dataset_dict['otus']['X'], pd.DataFrame):
            if args.only_otus_w_emb==1:
                otus=dataset_dict['otus']['distances'].columns.values
                dataset_dict['otus']['X'] = dataset_dict['otus']['X'][otus]

            dataset_dict['otus']['variable_names'] = dataset_dict['otus']['X'].columns.values

        # dataset_dict['otus']['X'] = dataset_dict['otus']['X'].divide(dataset_dict['otus']['X'].sum(1),axis='index')
    if not args.time:
        dataset_dict, y = merge_datasets(dataset_dict)
    else:
        y = dataset_dict['otus']['y']
    if isinstance(y, pd.Series):
        y=y.values
    if isinstance(y, pd.DataFrame):
        y = y.values.squeeze()
    # dist_dict = {k:dataset_dict[k]['distances'] for k in dataset_dict.keys()}
    # num_feat_dict = {k:dataset_dict[k]['X'].shape[1] for k in dataset_dict.keys()}
    # parser, dist_dict, dir, num_feat_dict, learn_embeddings = False)
    os.makedirs(OUTPUT_PATH + f'seed_{args.seed}', exist_ok=True)
    with open(OUTPUT_PATH + f'seed_{args.seed}/dataset_used.pkl','wb') as f:
        pkl.dump(dataset_dict,f)
    cv_trainer = CVTrainer(args, OUTPUT_PATH, y)

    if args.cv_type == 'kfold':
        if np.sum(y) / 2 < args.kfolds:
            args.kfolds = int(np.sum(y) / 2)
            print(f"{args.kfolds}-fold cross validation due to only {np.sum(y)} case samples")
        elif np.sum(y == 0) / 2 < args.kfolds:
            args.kfolds = int(np.sum(y == 0) / 2)
            print(f"{args.kfolds}-fold cross validation due to only {np.sum(y == 0)} control samples")
        train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y,num_splits=args.kfolds, seed=args.seed)
    elif args.cv_type=='loo':
        train_ixs, test_ixs = cv_loo_splits(np.zeros(y.shape[0]), y)
    elif args.cv_type=='one':
        train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y, num_splits=args.kfolds, seed=args.seed)
        train_ixs, test_ixs = [train_ixs[0]], [test_ixs[0]]
    elif args.cv_type=='eval':
        train_ixs, test_ixs = [],[]
        args.parallel=1
    else:
        print("Please enter valid option for cv_type. Options are: 'kfold','loo','one'")
        return

    if args.cv_type != 'one':
        folds = list(range(len(train_ixs))) + ['EVAL']
        train_ixs = train_ixs + [np.arange(y.shape[0])]
        test_ixs = test_ixs + [np.arange(y.shape[0])]
    else:
        folds = [0]

    rem_folds = []
    for fi in range(len(train_ixs)):
        if args.cv_type!='loo' and (len(np.unique(y[train_ixs[fi]]))==1 or len(np.unique(y[test_ixs[fi]]))==1):
            print(f'FOLD {fi} REMOVED; {len(np.unique(y[train_ixs[fi]]))} train classes, {len(np.unique(y[test_ixs[fi]]))} test classes')
            rem_folds.append(fi)
    if len(rem_folds)>0:
        for fi in rem_folds:
            folds.pop(fi)
            train_ixs.pop(fi)
            test_ixs.pop(fi)

    if args.parallel > 1:
        preds = Parallel(n_jobs=args.parallel)(delayed(cv_trainer.train_loop)(dataset_dict, train_idx, test_idx, fold)
                                 for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs))
    else:
        preds=[]
        for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs):
            print('FOLD {0}'.format(fold))
            ckpt_preds = cv_trainer.train_loop(dataset_dict, train_idx, test_idx, fold)
            preds.append(ckpt_preds)


    if len(preds)>1:
        final_preds = pd.concat(preds[:-1])
        f1 = f1_score(final_preds['true'], final_preds['preds'] > 0.5)
        auc = roc_auc_score(final_preds['true'], final_preds['preds'])
        final_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
                            '/' +'pred_results_f1_{0}_auc_{1}'.format(
            np.round(f1, 3),np.round(auc, 3)).replace('.','-') + '.csv')
        print('AUC: {0}'.format(auc))

    if args.cv_type != 'one':
        eval_preds = preds[-1]
        f1 = f1_score(eval_preds['true'], eval_preds['preds'] > 0.5)
        if len(np.unique(eval_preds['true'])) == 1:
            auc=np.nan
        else:
            auc = roc_auc_score(eval_preds['true'],eval_preds['preds'])
        eval_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
                            '/EVAL/' +'pred_results_f1_{0}_auc_{1}'.format(
            np.round(f1, 3),np.round(auc, 3)).replace('.','-') + '.csv')

    # preds_unzip = {key: [i[key] for i in preds_ls] for key in preds_ls[0]}
    # for pname, preds in preds_unzip.items():
    #     if len(preds[0]) > 0:
    #         if len(preds)>1:
    #             final_preds = pd.concat(preds[:-1])
    #             f1 = f1_score(final_preds['true'], final_preds['preds'] > 0.5)
    #             auc = roc_auc_score(final_preds['true'], final_preds['preds'])
    #             final_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
    #                                '/' + pname + '_pred_results_f1_{0}_auc_{1}'.format(
    #                 np.round(f1, 3),np.round(auc, 3)).replace('.','-') + '.csv')
    #             print('AUC: {0}'.format(auc))

    #         if args.cv_type != 'one':
    #             eval_preds = preds[-1]
    #             f1 = f1_score(eval_preds['true'], eval_preds['preds'] > 0.5)
    #             if len(np.unique(eval_preds['true'])) == 1:
    #                 auc=np.nan
    #             else:
    #                 auc = roc_auc_score(eval_preds['true'],eval_preds['preds'])
    #             eval_preds.to_csv(OUTPUT_PATH + f'seed_{args.seed}' +
    #                                '/EVAL/' + pname + '_pred_results_f1_{0}_auc_{1}'.format(
    #                 np.round(f1, 3),np.round(auc, 3)).replace('.','-') + '.csv')





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Differentiable rule learning for microbiome')
    args, parser = parse(parser)
    if args.method =='fc' or args.method=='full_fc':
        from model_fc import ComboMDITRE
    elif args.method=='nam':
        from model_nam import ComboMDITRE
    elif args.method=='basic':
        from models_time import ComboMDITRE
    else:
        print('Warning: accepted method not provided. Choices are: "basic", "fc", "nam", or "full_fc". Default "basic" will be used.')
        from models_time import ComboMDITRE

    seed_everything_custom(args.seed)
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    # if './datasets/' not in args.data:
    #     args.data = './datasets/cdi/' + args.data
    st = time.time()
    run_training_with_folds(args, OUTPUT_PATH=args.out_path + '/' + args.data_name + '/')
    et = time.time() - st
    print(f"TRAINING {args.epochs} TOOK {np.round(et/60,3)} MINUTES")

