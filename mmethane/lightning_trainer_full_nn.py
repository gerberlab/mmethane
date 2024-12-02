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

import warnings

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
# import datetime
# from model_nam import ComboMDITRE
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
    parser.add_argument('--plot_all_seeds', type=int, default=1)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--h_sizes', type=int, nargs='+', default=[12, 6])
    parser.add_argument('--standardize_from_training_data', type=int, default=1)
    parser.add_argument('--num_cpus', type=float, default=0.5)
    parser.add_argument('--num_gpus', type=float, default=0)

    # Main model specific parameters
    parser.add_argument('--lr_fc', default=0.01, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')
    parser.add_argument('--lr_bias', default=0.01, type=float,
                        help='Initial learning rate for linear classifier weights and bias.', nargs='+')


    # Training Parameters
    parser.add_argument('--data_met', metavar='DIR',
                        help='path to metabolite dataset',
                        default='../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl')
    parser.add_argument('--data_otu', metavar='DIR',
                        help='path to otu dataset',
                        default='../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl')
    parser.add_argument('--run_name', type=str,
                        help='Name of the dataset, will be used for log dirname',

                        # default='CDI_test_ISSUEFIXED',
                        # default=datetime.datetime.now().strftime('%m-%d-%Y_%H-%M'),
                        )
    parser.add_argument('--min_epochs', default=100, type=int, metavar='N',
                        help='number of minimum epochs to run')
    parser.add_argument('--epochs', default=5000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', type=int, default=9,
                        help='Set random seed for reproducibility')
    parser.add_argument('--cv_type', type=str, default='kfold',
                        choices=['loo', 'kfold', 'one', 'eval'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--early_stopping', default=0, type=int)
    parser.add_argument('--validate', default=0, type=int)
    parser.add_argument('--test', default=1, type=int)
    # parser.add_argument('--emb_dim', type = float, default=20)
    parser.add_argument('--out_path', type=str, default='/Users/jendawk/logs/mditre-logs/')
    parser.add_argument('--monitor', type=str, default='train_loss')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--dtype', type=str,
                        default=['metabs','otus'],
                        choices=['metabs', 'otus'],
                        help='Choose type of data', nargs='+')
    parser.add_argument('--schedule_lr', type=int, default=0,
                        help='Schedule learning rate')
    parser.add_argument('--parallel', type=int, default=0,
                        help='run in parallel')
    parser.add_argument('--only_mets_w_emb', type=int, default=1, help='whether or not keep only mets with embeddings')
    parser.add_argument('--only_otus_w_emb', type=int, default=1, help='whether or not keep only otus with embeddings')
    parser.add_argument('--learn_emb', type=int, default=0, help='whether or not to learn embeddings')
    # parser.add_argument('--lr_master',type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--p_d', type=float, default=0.1)
    parser.add_argument('--filter_data', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--plot_traces', type=int, default=1)
    parser.add_argument('--otu_tr', type=str, default='clr')
    parser.add_argument('--optimizer', type=str, default='RMSprop', choices=['Adam','RMSprop','AdamW','LBFGS',
                                                                             'NAdam','RAdam','Rprop','ASGD','SGD',
                                                                             'Adamax','Adagrad','Adadelta'])
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

    def parse_args(self):
        if not isinstance(self.args.dtype, list):
            self.args.dtype = [self.args.dtype]
        self.loss_params = []
        self.num_detectors = 0
        self.n_d_per_class = {}

        ### START HERE!!
        # if self.args.method=='basic':
        #     self.detector_ids = [[]*self.args.n_r]
        # else:
        #     self.detector_ids = [[]]
        x_in= {}
        for type in self.args.dtype:
            if type != 'metabs':
                self.learn_embeddings = False
            x_in[type] = self.data_dict[type]['X'].shape[1]

        self.model = ComboMDITRE(self.args, x_in=x_in)


        with open(self.dir + '/model_structure.txt','w') as f:
            for name, parameter in self.model.fc.named_parameters():
                f.write(f'{name}, {parameter.shape}\n')
        self.logging_dict = {b: [] for b, a in self.model.named_parameters()}
            #         self.logging_dict[name + '_' + b].append(a.detach().clone())
            # self.logging_dict.update({type + '_' + key: [] for key,value in self.model.rules[type]})
        self.scores_dict = {'train f1': [], 'test f1': [], 'val f1': [], 'train auc': [], 'test auc': [],
                            'train loss': [], 'test loss': [], 'val loss': [], 'val auc': [], 'total val loss': [],
                            }
        # self.loss_func = mapLoss(self.model, self.args, self.normal_wts, self.bernoulli_rules, self.bernoulli_det)
        self.running_loss_dict = {}
        self.grad_dict = {}
        for b, a in self.model.named_parameters():
            # print(b, a.get_device())
            self.grad_dict[b] = []


    def get_loss(self, logits, labels):
        if labels.size(0) > 1:
            if labels.sum() > 0:
                pos_weight = (labels.size()[0] / (2 * labels.sum()))
            else:
                pos_weight = None
        else:
            pos_weight = None

        loss = F.binary_cross_entropy_with_logits(logits, labels,
                                                  reduction='sum', pos_weight=pos_weight)
        return loss


    def forward(self, x_dict):
        return self.model(x_dict)

    def training_step(self, batch, batch_idx):
        xdict, y = batch

        if self.current_epoch % 1000 == 0:
            print('\nEpoch ' + str(self.current_epoch))
            print(f'{self.current_epoch} epochs took {time.time() - START_TIME} seconds')

        y_hat = self(xdict)

        self.loss = self.get_loss(y_hat, y)
        for b, a in self.model.named_parameters():
            # print(b, a.get_device())
            self.grad_dict[b].append(a)
            # self.logger.experiment.add_histogram(b + '_grad', a, self.current_epoch)
        self.train_preds.extend(y_hat.sigmoid())
        self.train_true.extend(y)

        return self.loss

    def on_train_epoch_end(self):
        for b, a in self.model.named_parameters():
            self.logging_dict[b].append(a.clone())

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
        self.log('train loss', self.loss, logger=False)
        self.log('train f1', f1, logger=False)
        self.log('train auc', auc, logger=False)
        self.scores_dict['train auc'].append(auc)
        self.scores_dict['train f1'].append(f1)
        self.scores_dict['train loss'].append(self.loss)
        self.train_true, self.train_preds = [], []
        return (None)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.val_loss = self.get_loss(y_hat, y)
        self.val_preds.extend(y_hat.sigmoid())
        self.val_true.extend(y)
        return self.val_loss

    def on_validation_epoch_end(self):
        y, y_hat = torch.stack(self.val_true, 0), torch.stack(self.val_preds, 0)
        self.y_preds = y_hat
        self.y_true = y
        f1 = self.F1Score(y_hat > 0.5, y)
        try:
            f1 = self.F1Score(y_hat > 0.5, y)
            auc = self.AUROC(y_hat, y)
            ctrls, case = y == 0, y != 0
            self.log('val auc', auc)
            self.log('val f1', f1)
        except:
            f1, auc, acc_0, acc_1 = 0, 0, 0, 0

        self.log('val loss', self.val_loss)
        #

        if len(self.scores_dict['train loss']) > 0:
            self.scores_dict['val f1'].append(f1)
            self.scores_dict['val auc'].append(auc)
            self.scores_dict['val loss'].append(self.val_loss)

        self.y_hat_val, self.y_val = y_hat, y
        self.val_true, self.val_preds = [], []
        return (None)
        # return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_preds.extend(y_hat.sigmoid())
        self.test_true.extend(y)
        self.test_loss  = self.get_loss(y_hat, y)
        return self.test_loss

    def on_test_epoch_end(self):
        y, y_hat = torch.stack(self.test_true, 0), torch.stack(self.test_preds, 0)
        f1 = self.F1Score(y_hat > 0.5, y)
        ctrls, case = y == 0, y != 0
        # auc = roc_auc_score(y.detach().cpu().numpy(), self.y_hat.sigmoid().detach().cpu().numpy())
        self.scores_dict['test loss'].append(self.test_loss.detach().item())
        self.scores_dict['test f1'].append(f1)
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
        self.y_preds = y_hat
        self.y_true = y

        self.test_true, self.test_preds = [], []
        return (None)

    def configure_optimizers(self):
        # ['Adam','RMSprop','AdamW','LBFGS',
        #  'NAdam','RAdam','Rprop','ASGD','SGD',
        #   'Adamax','Adagrad','Adadelta']
        if isinstance(self.args.lr_fc, list):
            self.args.lr_fc = self.args.lr_fc[0]
        if self.args.optimizer=='Adam':
            self.optimizer_0 = optim.Adam(self.model.parameters(), lr=self.args.lr_fc, weight_decay=self.args.weight_decay)
        elif self.args.optimizer=='AdamW':
            self.optimizer_0 = optim.AdamW(self.model.parameters(), lr=self.args.lr_fc,
                                          weight_decay=self.args.weight_decay)
        elif self.args.optimizer=='RMSprop':
            self.optimizer_0 = optim.RMSprop(self.model.parameters(), lr=self.args.lr_fc, weight_decay=self.args.weight_decay)
        elif self.args.optimizer=='LBFGS':
            self.optimizer_0=optim.LBFGS(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='NAdam':
            self.optimizer_0=optim.NAdam(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='RAdam':
            self.optimizer_0=optim.RAdam(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='Rprop':
            self.optimizer_0=optim.Rprop(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='ASGD':
            self.optimizer_0=optim.ASGD(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='SGD':
            self.optimizer_0=optim.SGD(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='Adamax':
            self.optimizer_0=optim.Adamax(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='Adagrad':
            self.optimizer_0=optim.Adagrad(self.model.parameters(), lr = self.args.lr_fc)
        elif self.args.optimizer=='Adadelta':
            self.optimizer_0=optim.Adadelta(self.model.parameters(), lr = self.args.lr_fc)
        else:
            ValueError("Please provide correct argument for optimizer. Options are Adam or RMSprop.\n"
                       "To add additional options, see lightning_trainer.py, lines 617-620")
        if args.schedule_lr == 1:
            self.scheduler_0 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_0,
                                                                    int(self.args.epochs))
            # eta_min=1e-5)
            return [self.optimizer_0], [self.scheduler_0]
        else:
            return [self.optimizer_0]


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

@conditional_decorator(ray.remote, global_args.use_ray, num_gpus=0)
class CVTrainer():
    def __init__(self, args, OUTPUT_PATH, y):
        self.args = args
        self.outpath = OUTPUT_PATH
        self.Y = y
        self.y = y.values

    def check_if_fold_already_finished(self, fold, log_dir):
        print(f"CHECKING IF FOLD FINISHED IN DIR: {log_dir}")
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
                            print(f"FOLD FINISHED IN DIR: {log_dir + '/' + file_folder + '/' + file}")
                            print('Fold {0} testing already finished'.format(fold))
                            return pred_ls
                else:
                    if 'pred_results' in file_folder:
                        pred_ls = pd.read_csv(log_dir + '/' + file_folder, index_col=0)
                        print('Fold {0} testing already finished'.format(fold))
                        print(f"FOLD FINISHED IN DIR: {log_dir + '/' + file_folder + '/' + file}")
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

        with open(self.outpath + f"seed_{self.args.seed}/time.txt", "w") as f:
            f.write(f"OUTER TIME LOG\n")

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
        pred_ls = self.check_if_fold_already_finished(fold, tb_logger.log_dir)
        if pred_ls is not None:
            return pred_ls

        monitor = args.monitor.replace('_', ' ')

        callbacks = [ModelCheckpoint(save_last=False,
                                     dirpath=tb_logger.log_dir,
                                     save_top_k=self.args.validate,
                                     verbose=False,
                                     monitor=monitor,
                                     every_n_epochs=None, every_n_train_steps=1, train_time_interval=None,
                                     mode='min' if 'loss' in monitor else 'max',
                                     filename='{epoch}' + '-{' + monitor + ':.2f}',
                                     )
                         ]
        if args.early_stopping == 1:
            callbacks.extend([EarlyStopping(monitor=monitor, patience=100)])
        # else:
        # callbacks.append(LearningRateMonitor())
        trainer = pl.Trainer(logger=tb_logger, max_epochs=self.args.epochs, min_epochs=self.args.min_epochs,
                             callbacks=callbacks,
                             enable_progress_bar=False, gradient_clip_val=0.5
                             )

        if self.args.filter_data:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=True,
                                                                                 sqrt_transform = self.args.otu_tr=='sqrt',
                                                                                 clr_transform_otus=self.args.otu_tr=='clr',
                                                                                 standardize_otus=self.args.otu_tr=='std',

                                                                                 standardize_from_training_data=self.args.standardize_from_training_data,
                                                                                 logdir=tb_logger.log_dir)
        else:
            train_dataset_dict, test_dataset_dict = split_and_preprocess_dataset(dataset_dict, train_ixs,
                                                                                 test_ixs, preprocess=False,
                                                                                 sqrt_transform=self.args.otu_tr == 'sqrt',
                                                                                 clr_transform_otus=self.args.otu_tr == 'clr',
                                                                                 standardize_otus=self.args.otu_tr == 'std',
                                                                                 standardize_from_training_data=self.args.standardize_from_training_data,
                                                                                 logdir=tb_logger.log_dir)
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
            if fold == "EVAL":
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
            test_loader = DataLoader(mditreDataset(test_dataset_dict), batch_size=self.batch_size, shuffle=False)
            trainer.fit(self.lit_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            if fold=="EVAL":
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
        if fold=='EVAL':
            save_input_data(self.lit_model, train_dataset_dict, test_dataset_dict, self.args,
                            self.outpath + f'seed_{self.args.seed}/')
        with open(tb_logger.log_dir + '/running_loss_dict.pkl', 'wb') as f:
            pkl.dump(self.lit_model.running_loss_dict, f)

        # if self.args.remote == 0 and (self.args.seed == 0 or self.args.plot_all_seeds == 1 or self.args.seed==1):
        if self.args.seed<10:
            save_and_plot_post_training(self.lit_model, train_ixs, test_ixs, tb_logger.log_dir,
                                        plot_traces=self.args.plot_traces == 1, best_epoch=best_epoch,save_grads=False)



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

    if args.cv_type == 'eval':
        check_inputs_for_eval(args.out_path + '/' + args.run_name + '/', args.__dict__)

    from models_fc import ComboMDITRE
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

