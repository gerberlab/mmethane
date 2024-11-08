import numpy as np
from scipy.special import logit

import torch
import scipy.stats as st
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import sklearn
import pandas as pd

# Lowest possible float
EPSILON = np.finfo(np.float32).tiny

def get_epsilon(data):
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = np.max(vals)
    epsilon = 0.1*np.min(np.abs(vals))
    return epsilon
def run_logreg(x_dict, y_train, inner_folds=5, scorer='f1'):
    x_tr = []
    for k, v in x_dict.items():
        x_tr.append(v.detach().cpu().numpy())
    X_tr = np.concatenate(x_tr, axis=1)
    X_tr = X_tr/np.sum(X_tr,0)
    eps = EPSILON
    X_train = np.log(X_tr + eps)
    mean = np.mean(X_train, 0) + 1e-10
    stdev = np.std(X_train, 0) + 1e-10
    X_train = (X_train-mean) / (stdev)
    lambda_min_ratio = 0.01
    path_len = 300
    bval = True
    lam = 0
    while (bval):
        lam += 0.1
        model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
        try:
            model2.fit(X_train, y_train)
        except:
            print('error')
        if np.sum(np.abs(model2.coef_)) < 1e-8:
            l_max = lam + 1
            bval = False
    # l_max = 14
    l_path = np.logspace(np.log10(l_max * lambda_min_ratio), np.log10(l_max), path_len)
    Cs = [1 / l for l in l_path]
    model = sklearn.linear_model.LogisticRegressionCV(cv=int(np.min([inner_folds,
                                                                     y_train.sum()])),
                                                      penalty='l1', scoring=scorer,
                                                      Cs=Cs, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    return model, mean, stdev

def logreg_test(x_dict, y, model, train_mean, train_stdev):
    x_tr = []
    for k, v in x_dict.items():
        x_tr.append(v.detach().cpu().numpy())
    X_test = np.concatenate(x_tr, axis=1)
    X_test = (X_test - train_mean) / (train_stdev)
    log_prob_predictions = model.predict_log_proba(X_test)
    score = model.score(X_test, y)
    return log_prob_predictions, score



    # score = model.score(X_ts, self.y.iloc[test_ids])
    # coef_df = pd.DataFrame(model.coef_.squeeze(), index=X_tr.columns.values, columns=['Fold ' + str(i)])
    # pred_probs = model.predict_proba(X_ts)

class Standardize():
    def __init__(self, train_mean=None, train_stdev=None):
        self.train_mean=train_mean
        self.train_stdev=train_stdev
        self.eps = torch.tensor(EPSILON)

    def _xdict_to_x(self, x):
        x_tr = []
        for k, v in x.items():
            x_tr.append(v)
        X_tr = torch.cat(x_tr, 1)
        return X_tr

    def standardize_training_data(self, x):
        if isinstance(x, dict):
            x = self._xdict_to_x(x)

        x = torch.log(x + self.eps)
        self.train_mean = torch.mean(x, 0)
        self.train_std = torch.std(x, 0)
        x = (x - self.train_mean)/(self.train_std + self.eps)
        return x

    def standardize_testing_data(self, x):
        if isinstance(x, dict):
            x = self._xdict_to_x(x)
        x = (x - self.train_mean)/(self.train_std + self.eps)
        return x



class NormLayer():
    def __init__(self, dtype=None):
        self.dtype = dtype

    def standardize(self, x, dtype=None):
        if dtype=='metabs' or self.dtype=='metabs':
            return x
        eps = torch.tensor(EPSILON)
        x = torch.log(x + eps)
        x = (x - torch.mean(x, 0))/(torch.std(x, 0) + eps)
        return x

# binary concrete selector
def binary_concrete(x, k, hard=False, use_noise=False, noise_factor=1.):
    if use_noise:
        u = noise_factor*torch.zeros_like(x.data).uniform_(0, 1) + torch.tensor([EPSILON], device=x.device)
        logs = u.log() - (-u).log1p()
        # import pdb; pdb.set_trace()
        z_soft = torch.sigmoid((x + logs) * k)
    else:
        z_soft = torch.sigmoid(x * k)

    # Straight through estimator
    if hard:
        z = (z_soft > 0.5).float() - z_soft.detach() + z_soft
    else:
        z = z_soft

    return z


# approximate a unit height boxcar function
# using analytic approximations of the Heaviside function
def unitboxcar(x, mu, l, k):
    # parameterize boxcar function by the center and length
    dist = x - mu
    window_half = l / 2.
    y = torch.sigmoid((dist + window_half) * k) - torch.sigmoid((dist - window_half) * k)
    return y


def transf_log(x, u, l):
    return (u - l) * torch.sigmoid(x) + l


def inv_transf_log(x, u, l):
    return logit((x - l) / (u - l))


class CustomBernoulli():
    def __init__(self, p, device='cpu'):
        self.device = device
        if not torch.is_tensor(p):
            p = torch.tensor(p, device=self.device)
        self.p = p

    def pdf(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        return (self.p**x)*(1-self.p)**(1-x)

    def log_prob(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        return x*torch.log(self.p) + (1-x)*torch.log(1-self.p)

    def sample(self, size):
        b = Bernoulli(self.p)
        return b.sample(size)

class TruncatedNormal():
    def __init__(self, mean, var, a, b, device='cpu'):
        self.device = device
        self.mean = torch.tensor(mean, device=self.device)
        self.var = torch.tensor(var, device=self.device)
        self.a = torch.tensor(a, device=self.device)
        self.b = torch.tensor(b, device=self.device)
        self.normal = Normal(torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))


    def pdf(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        pdf = self.normal.log_prob((x - self.mean) / self.var).exp()
        return (1 / self.var) * (pdf / (self.normal.cdf((self.b - self.mean) / self.var) -
                                        self.normal.cdf((self.a - self.mean) / self.var)))

    def log_prob(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        log_prob = self.normal.log_prob((x - self.mean) / self.var)
        return torch.log(1/self.var) + log_prob - torch.log((self.normal.cdf((self.b - self.mean) / self.var) -
                                        self.normal.cdf((self.a - self.mean) / self.var)))

    def sample(self, size=None):
        a_, b_ = (self.a - self.mean) / np.sqrt(self.var), (self.b - self.mean) / np.sqrt(self.var)
        tn = st.truncnorm(a=a_, b=b_, loc=self.mean, scale=np.sqrt(self.var))
        return torch.tensor(tn.rvs(size), device=self.device)


class BinaryConcrete():
    def __init__(self, loc, tau, device='cpu'):
        self.loc = loc
        self.tau = tau
        self.device = device
        if not torch.is_tensor(self.loc):
            self.loc = torch.Tensor([self.loc], device=self.device)
        if not torch.is_tensor(self.tau):
            self.tau = torch.Tensor([self.tau], device=self.device)

    def sample(self, size=[1]):
        L = st.logistic(0,1).rvs(size)
        return 1/(1 + torch.exp(-(torch.log(self.loc) + torch.tensor(L, device=self.device).float())/self.tau))

    def pdf(self, x):
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x, device=self.device)
            except:
                x = torch.Tensor([x], device=self.device)
        top = self.tau*self.loc*(x.pow(-self.tau-1))*((1-x).pow(-self.tau-1))
        bottom = (self.loc*(x.pow(-self.tau)) + (1-x).pow(-self.tau)).pow(2)
        return top / bottom

    def log_prob(self, x):
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x, device=self.device)
            except:
                x = torch.Tensor([x], device=self.device)
        return torch.log(self.tau) + torch.log(self.loc) + (-self.tau - 1)*torch.log(x) + \
               (-self.tau - 1)*torch.log(1-x) - 2*torch.log(self.loc*(x.pow(-self.tau)) + (1-x).pow(-self.tau))
