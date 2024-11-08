import torch.nn as nn
import torch.nn.functional as F
from model_helper import *
from time_agg import *

# Lowest possible float
EPSILON = np.finfo(np.float32).tiny
class featMDITRE(nn.Module):
    def __init__(self, num_otu_centers,dist, emb_dim, dtype, num_emb_to_learn=0, args=None, num_rules = None, num_feats=1, num_time=1):
        super(featMDITRE, self).__init__()
        self.num_otu_centers = num_otu_centers
        self.num_rules = num_rules
        self.emb_dim = emb_dim
        self.register_buffer('dist', torch.from_numpy(dist))
        self.dtype = dtype
        self.num_emb_to_learn = num_emb_to_learn
        self.args = args
        self.time_agg = TimeAgg(num_time)

    def forward(self, x, k_otu=1, k_thresh=1, hard=False, exp_kappa=False, mask=None, k_time=1, k_slope=1):
        if exp_kappa:
            kappa = self.kappa.exp().unsqueeze(-1)
        else:
            kappa = self.kappa.unsqueeze(-1)
        if self.emb_to_learn is not None:
            if self.dist.shape[0] != self.emb_to_learn.shape[0]:
                dist = (self.eta.reshape(self.num_rules,
                                         self.num_otu_centers, 1, self.emb_dim) -
                        torch.cat((self.dist, self.emb_to_learn), 0)).norm(2,dim=-1)
            else:
                dist = (self.eta.reshape(self.num_rules, self.num_otu_centers, 1, self.emb_dim) - self.emb_to_learn).norm(2, dim=-1)
            self.full_dist = dist
        else:
            dist = (self.eta.reshape(self.num_rules, self.num_otu_centers, 1, self.emb_dim) - self.dist).norm(2, dim=-1)

        if hard:
            otu_wts_soft = torch.sigmoid((kappa - dist) * k_otu)
            otu_wts_unnorm = (otu_wts_soft > 0.5).float() - otu_wts_soft.detach() + otu_wts_soft
        else:
            otu_wts_unnorm = torch.sigmoid((kappa - dist) * k_otu)
        self.wts = otu_wts_unnorm
        if len(x.shape) > 3:
            x = x.squeeze()
        if self.dtype == 'metabs':
            x = (torch.einsum('kij,sjt->skit', otu_wts_unnorm, x) + 1e-10) / (torch.sum(otu_wts_unnorm,2).unsqueeze(0).unsqueeze(-1) + 1e-10)
        else:
            x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)
        self.kappas = kappa
        self.emb_dist = dist
        if self.args.time==1:
            x, x_slope = self.time_agg(x, mask=mask, k=k_time)
            if self.slope_thresh is not None:
                x_slope = torch.sigmoid((x_slope - self.slope_thresh) * k_slope)
        else:
            x = x.squeeze(-1)
            x_slope=None
        x = torch.sigmoid((x - self.thresh) * k_thresh)
        self.x_out = x
        return x, x_slope

    def init_params(self, init_args,device = 'cuda'):
        if self.num_emb_to_learn > 0:
            self.emb_to_learn = nn.Parameter(torch.tensor(init_args['emb_init'], device=device, dtype=torch.float))
        else:
            self.emb_to_learn = None

        self.eta = nn.Parameter(torch.tensor(init_args['eta_init'], device=device, dtype=torch.float))
        self.kappa = nn.Parameter(torch.tensor(init_args['kappa_init'], device=device, dtype=torch.float))
        self.thresh = nn.Parameter(torch.tensor(init_args['thresh_init'], device=device, dtype=torch.float))
        if self.args.time==1:
            self.time_agg.init_params(init_args, device)
            if 'slope_init' in init_args.keys():
                self.slope_thresh = nn.Parameter(torch.tensor(init_args['slope_init'], device=device, dtype=torch.float))
            else:
                self.slope_thresh = None


class ComboMDITRE(nn.Module):
    def __init__(self, args, module_dict):
        super(ComboMDITRE, self).__init__()
        # self.met_args = args_met
        self.args = args
        # self.bug_args = args_bug
        # self.names, self.modules = list(module_dict.keys()), list(module_dict.values())
        self.module_names, tmp_modules = zip(*module_dict.items())
        self.module_names = list(self.module_names)
        self.combo_modules = nn.ModuleList(list(tmp_modules))
        # self.module_dict = module_dict
        self.num_rules = self.args.n_r
        self.num_otus = int(sum([module_dict[n].num_otu_centers for n in module_dict.keys()]))
        # self.alpha = nn.Parameter(torch.Tensor(self.num_rules, self.num_otus))
        # self.weight = nn.Parameter(torch.Tensor(self.num_rules, 1))
        # # Logistic regression bias
        # self.bias = nn.Parameter(torch.Tensor(1))
        # # Parameter for selecting active rules
        # self.beta = nn.Parameter(torch.Tensor(self.num_rules))

    def forward(self, x_dict, k_dict, hard_otu=False, hard_bc=False, noise_factor=1):
        x_out, x_out_slope = [], []
        for i, name in enumerate(self.module_names):
            x = x_dict[name]
            if self.args.time==1:
                x, mask = x_dict[name]
                k_time, k_slope = k_dict[f'{name}_k_time'], 1
                if f'{name}_k_slope' in k_dict.keys():
                    k_slope = k_dict[f'{name}_k_slope']
            else:
                k_time, k_slope = 1,1
                mask=None

            x, x_slope = self.combo_modules[i](x.unsqueeze(-1), k_otu=k_dict[name + '_k_otu'],k_thresh=k_dict[name + '_k_thresh'],hard=hard_otu,
                                    exp_kappa = self.args.kappa_prior == 'log-normal', k_time=k_time, k_slope=k_slope, mask=mask)
            x_out.append(x)
            if x_slope is not None:
                x_out_slope.append(x_slope)
        # for name, x in x_dict.items():
        #     x_tmp = self.module_dict[name](x.unsqueeze(-1), k_otu=k_dict[name + '_k_otu'],
        #                                         k_thresh=k_dict[name + '_k_thresh'], hard=hard_otu,
        #                                         exp_kappa = self.args.kappa_prior == 'log-normal')
        #     x_out.append(x_tmp)

        x_out = torch.cat(x_out, -1)
        # x_out = self.fc(x_out, k=k_dict['k_beta'], hard=False, use_noise=True)
        if self.args.use_k_1==1:
            k_dict['k_alpha'], k_dict['k_beta'] = 1,1
        self.z_d = binary_concrete(self.alpha, k=k_dict['k_alpha'], hard=hard_bc,use_noise=noise_factor==1, noise_factor=noise_factor)

        x_out = (1 - self.z_d.mul(1 - x_out)).prod(dim=-1)

        if self.args.time==1 and len(x_out_slope)>0:
            self.z_d_slope = binary_concrete(self.alpha_slope, k=k_dict['k_alpha'], hard=hard_bc,use_noise=noise_factor==1, noise_factor=noise_factor)
            x_out_slope = torch.cat(x_out_slope, -1)
            x_out_slope = (1 - self.z_d_slope.mul(1-x_out_slope)).prod(dim=-1)
            x_out = x_out * x_out_slope

        self.z_r = binary_concrete(self.beta, k=k_dict['k_beta'], hard=hard_bc, use_noise=noise_factor==1, noise_factor=noise_factor)
        x_out = F.linear(x_out, self.weight * self.z_r.unsqueeze(0), self.bias)
        self.log_odds = x_out.squeeze(-1)
        return x_out.squeeze(-1)

    def init_params(self, init_args, device='cuda'):
        self.alpha = nn.Parameter(torch.tensor(init_args['alpha_init'], device=device, dtype=torch.float))
        self.weight = nn.Parameter(torch.tensor(init_args['w_init'], device=device, dtype=torch.float))
        # Logistic regression bias
        self.bias = nn.Parameter(torch.tensor(init_args['bias_init'], device=device, dtype=torch.float))
        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.tensor(init_args['beta_init'], device=device, dtype=torch.float))

        if self.args.time==1:
            self.alpha_slope = nn.Parameter(torch.tensor(init_args['alpha_init'], device=device, dtype=torch.float))
        else:
            self.alpha_slope = None

        return