# Loss function on labels

# CURRENTLY: NO NEGBIN LOSS OR BINCON LOSS
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import itertools


def ce_loss(logits, labels):
    if labels.size(0) > 1:
        if labels.sum() > 0:
            pos_weight = (labels.size()[0] / (2*labels.sum()))
        else:
            pos_weight = None
    else:
        pos_weight = None

    loss = F.binary_cross_entropy_with_logits(logits, labels,
                                              reduction='sum', pos_weight=pos_weight)
    return loss

def negbin_loss(x, mean, var):
    r = mean ** 2 / (var - mean)
    p = (var - mean) / var
    loss_1 = -torch.lgamma(r + x + 1e-5)
    loss_2 = torch.lgamma(x + 1)
    loss_3 = -(x * np.log(p))
    loss = loss_1 + loss_2 + loss_3
    return loss

def binary_concrete_loss(temp, alpha, x):
    try:
        # loss_1 = (temp + 1) * (torch.log(x * (1 - x) + 1e-5))
        # loss_2 = 2 * (torch.log((alpha / ((x ** temp) + 1e-5)) + (1 / ((1 - x) ** temp) + 1e-5) + 1e-5))
        loss_1 = (temp + 1) * (torch.log(x * (1 - x))) - np.log(temp) - np.log(alpha)
        loss_2 = 2 * (torch.log((alpha / ((x ** temp))) + (1 / ((1 - x) ** temp))))
    except Exception as e:
        print(x)
        raise e

    return loss_1 + loss_2

# def binary_concrete_loss_new(temp, alpha, x):
#     return torch.log(temp) - temp*x + torch.log(alpha) - 2*torch.log(1 + torch.exp(-temp*x + torch.log(alpha)))
def diversity_loss(q,z,wts):
    flat_wts = wts.flatten(0,1)
    multiplier = (z.T * q).T
    flat_mult = multiplier.flatten(0,1)
    ixa, ixb = zip(*list(itertools.combinations(range(flat_wts.shape[0]), 2)))
    a = flat_wts[ixa, :]
    b = flat_wts[ixb, :]
    res = (1 - (torch.sum(a * b, 1) / (a.norm(dim=1) * b.norm(dim=1)))) * flat_mult[torch.LongTensor(ixa)] * flat_mult[
        torch.LongTensor(ixb)]
    return -torch.log(torch.nansum(res)/torch.nansum(flat_mult[torch.LongTensor(ixa)]*flat_mult[torch.LongTensor(ixb)]))



class moduleLoss():
    def __init__(self, model, args, normal_kappa, normal_emb, name, device='cpu'):
        self.model = model
        self.args = args
        if hasattr(self.args, 'n_mult'):
          self.n_mult = self.args.n_mult
        else:
          self.n_mult=200
        self.normal_kappa = normal_kappa
        self.normal_emb = normal_emb
        self.name = name
        self.device = device
        self.loss_params = [name+'_emb_normal_loss',name+'_kappa_normal_loss']



    def loss(self, k_alpha, N, Nf):
        det_eps = (1 / k_alpha) / 100
        if self.args.kappa_eta_prior:
            # kappa_normal_loss = -self.normal_kappa.log_prob(self.model.kappa).sum()
            if self.args.adj_kappa_loss:
                term=(N / (Nf))
            else:
                term=1
            kappa_normal_loss = -self.normal_kappa.log_prob(self.model.kappa).sum() * term
            emb_normal_loss = -self.normal_emb.log_prob(self.model.eta).sum()
        else:
            kappa_normal_loss = torch.tensor(0.0, device=self.device)
            emb_normal_loss = torch.tensor(0.0, device = self.device)

        # reg_loss =  emb_normal_loss + kappa_normal_loss
        detectors = self.model.z_d

        loc = self.args.p_d/(1-self.args.p_d)
        if self.args.adj_detector_loss:
            term = (N / (self.n_mult * Nf))
        else:
            term = 1

        det_abun_bc_loss = binary_concrete_loss(1 / k_alpha, loc, (1 - 2 * det_eps) * detectors + det_eps).sum()*term
        num_det_loss = det_abun_bc_loss - binary_concrete_loss(1 / k_alpha, 1, (1 - 2 * det_eps) *
                                                               detectors.detach() + det_eps).sum()*term
        z_loss = torch.tensor(0.0)

        reg_loss = emb_normal_loss + kappa_normal_loss + det_abun_bc_loss + z_loss

        loss_dict = {
                     self.name+'_emb_normal_loss': emb_normal_loss,
                     self.name+'_kappa_normal_loss': kappa_normal_loss,
                    self.name + '_detector_bc_loss': det_abun_bc_loss,
            self.name + 'num_detector_loss': num_det_loss,
                     }

        return reg_loss, loss_dict



class mapLoss():
    def __init__(self, model, args, normal_wts, normal_bias, device = 'cpu', n_d=20):
        self.model = model
        self.args = args
        self.normal_wts = normal_wts
        self.device = device
        self.normal_bias = normal_bias
        self.n_d = n_d

    def loss(self, outputs, labels, k_beta):
        train_loss = ce_loss(outputs, labels)
        N = len(labels)
        # det_eps = (1 / k_alpha) / 100

        # if self.args.add_logreg:
        #     logreg_loss = -self.normal_wts.log_prob(self.model.logreg).sum()
        # else:
        #     logreg_loss = torch.tensor(0.0)
        # det_loss_dict = {m:torch.tensor(0) for m in self.model.module_names}
            
        rules = self.model.z_r

        l2_wts_loss = -self.normal_wts.log_prob(self.model.weight).sum()
        # bias_loss = -self.normal_bias.log_prob(self.model.bias).sum()
        bias_loss = torch.tensor(0., device=self.device)

        rule_eps = (1 / k_beta) / 100

        loc_param = (self.args.p_r / (1 - self.args.p_r))
        # rule_bc_loss = binary_concrete_loss(1 / k_beta, loc_param,
        #                                     (1 - 2 * rule_eps) * rules + rule_eps).sum
        if self.args.adj_rule_loss:
            term = (N / (self.args.n_mult * self.n_d))
        else:
            term = 1
        rule_bc_loss = binary_concrete_loss(1 / k_beta, loc_param, (1 - 2 * rule_eps) * rules + rule_eps).sum()*term
        num_rule_loss = rule_bc_loss - binary_concrete_loss(1 / k_beta, 1,
                                                            (1 - 2 * rule_eps) * rules.detach() + rule_eps).sum()*term
            # rule_bc_loss = torch.tensor(0.0)

        reg_loss = l2_wts_loss + rule_bc_loss
                   # + logreg_loss
        loss_dict = {'train_loss': train_loss,
            'num_rule_loss': num_rule_loss,
                 'l2_wts_loss': l2_wts_loss,
                 'rule_bc_loss': rule_bc_loss,
                 # 'detector_bc_loss': det_abun_bc_loss,

                 # 'num_detector_loss':z_loss,
                     'bias_loss':bias_loss
                     # 'logreg_loss': logreg_loss,
                 }

        # for m in self.model.module_names:
        #     loss_dict[m] = det_loss_dict[m]

        # Backprop for computing grads
        return train_loss, reg_loss, loss_dict
