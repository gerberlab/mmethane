import torch.nn as nn
import torch.nn.functional as F
from utilities.model_helper import *
# Lowest possible float
EPSILON = np.finfo(np.float32).tiny


class NAM(nn.Module):
    def __init__(self, L, K, h=[18,12], p=0.2, dtype=None):
        super(NAM, self).__init__()
        self.L = L
        self.K = K
        self.NAM = nn.ModuleList()
        # self.norm = NormLayer(dtype)
        hnew=[]
        for i,hh in enumerate(h):
            if hh==0:
                if i == 0:
                    # hnew.append(np.int(3*L/4))
                    hnew.append(int(np.sqrt(3*L)+2*np.sqrt(L/3)))
                else:
                    hnew.append(int(np.sqrt(L/3)))
                    # hnew.append(np.int(np.sqrt(3*L)+2*np.sqrt(L/3)))
            else:
                hnew.append(hh)
        self.hidden_sizes = [1] + hnew + [1]
        for k in range(self.K):
            detector_module = nn.ModuleList()
            for l in range(self.L):
                inner_module = nn.ModuleList()
                for i in range(len(self.hidden_sizes)-1):
                    layer = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], bias=True)
                    inner_module.append(layer)
                    # inner_module.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], bias=True))
                    if i <= len(h)-1:
                        inner_module.append(nn.ReLU())
                        inner_module.append(nn.Dropout(p))
                detector_module.append(inner_module)
            self.NAM.append(detector_module)
        print(self.NAM)

    def forward(self, x):
        out = []
        # x = self.B1d(x)
        for k in np.arange(self.K):
            tmp = []
            x_ = self.norm.standardize(x[:, k, :])
            for l in np.arange(self.L):
                x_in = x_[:, l:l+1]
                for layer in self.NAM[k][l]:
                    x_in = layer(x_in)
                tmp.append(x_in)
            tmp = torch.cat(tmp, -1)
            out.append(tmp)
        out = torch.stack(out, 1)
        # out = torch.cat([self.NAM[l](x[:, l:l + 1])  for l in np.arange(self.L)], 1)
        # out = bias + (out * z).sum(1)
        return out

class featMDITRE(nn.Module):
    def __init__(self, num_otu_centers,dist, emb_dim, dtype, num_emb_to_learn=0, args=None, num_rules = None, num_feats=1):
        super(featMDITRE, self).__init__()
        self.num_otu_centers = num_otu_centers
        self.num_rules = num_rules
        self.emb_dim = emb_dim
        self.register_buffer('dist', torch.from_numpy(dist))
        self.dtype = dtype
        self.num_emb_to_learn = num_emb_to_learn
        self.args = args
        self.num_feats = num_feats

    def forward(self, x, k_otu=1, k_thresh=1, k_alpha=1, hard=False, exp_kappa=False):
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
        if self.dtype == 'metabs':
            x = (torch.einsum('kij,sjt->skit', otu_wts_unnorm, x) + 1e-10) / (torch.sum(otu_wts_unnorm,2).unsqueeze(0).unsqueeze(-1) + 1e-10)
        else:
            x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)
        self.kappas = kappa
        self.emb_dist = dist

        x = x.squeeze(-1)

        x = torch.sigmoid((x - self.thresh) * k_thresh)
        self.x_out = x
        if self.args.use_k_1==1:
            self.z_d = torch.sigmoid(self.alpha)
        else:
            self.z_d = torch.sigmoid(self.alpha*k_alpha)
        return x

    def init_params(self, init_args,device = 'cuda'):
        if self.num_emb_to_learn > 0:
            self.emb_to_learn = nn.Parameter(torch.tensor(init_args['emb_init'], device=device, dtype=torch.float))
        else:
            self.emb_to_learn = None

        self.eta = nn.Parameter(torch.tensor(init_args['eta_init'], device=device, dtype=torch.float))
        self.kappa = nn.Parameter(torch.tensor(init_args['kappa_init'], device=device, dtype=torch.float))
        self.thresh = nn.Parameter(torch.tensor(init_args['thresh_init'], device=device, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(init_args['alpha_init'], device=device, dtype=torch.float))



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
        self.num_feats = int(sum([module_dict[n].num_feats for n in module_dict.keys()]))
        self.norm = NormLayer()
        # self.alpha = nn.Parameter(torch.Tensor(self.num_rules, self.num_otus))
        # self.weight = nn.Parameter(torch.Tensor(self.num_rules, 1))
        # # Logistic regression bias
        # self.bias = nn.Parameter(torch.Tensor(1))
        # # Parameter for selecting active rules
        # self.beta = nn.Parameter(torch.Tensor(self.num_rules))

    def forward(self, x_dict, k_dict, hard_otu=False, hard_bc=False, noise_factor=1):
        x_out = []
        self.order=[]
        z_vec = []
        for i, name in enumerate(self.module_names):
            x = x_dict[name]
            # k_alpha = k_dict[name + '_k_alpha']
            x = self.combo_modules[i](x.unsqueeze(-1), k_otu=k_dict[name + '_k_otu'],k_thresh=k_dict[name + '_k_thresh'],
                                      hard=hard_otu,k_alpha = k_dict[name + '_k_alpha'],
                                    exp_kappa = self.args.kappa_prior == 'log-normal')
            x_out.append(x)
            z_vec.append(self.combo_modules[i].z_d)
            self.order.extend([name]*x.shape[-1])

        self.z_d = torch.cat(z_vec, -1)
        x_out = torch.cat(x_out, -1)
        self.detector_activations = x_out
        # if self.args.use_k_1==1:
        #     self.z_d = torch.sigmoid(self.alpha)
        #     # self.z_d = binary_concrete(self.alpha, k=1, hard=hard_bc,use_noise=noise_factor==1, noise_factor=noise_factor)
        # else:
        #     self.z_d = torch.sigmoid(self.alpha * k_dict['k_alpha'])
        #     # self.z_d = binary_concrete(self.alpha, k=k_dict['k_alpha'], hard=hard_bc, use_noise=noise_factor == 1,
        #     #                            noise_factor=noise_factor)

        x_out = (1 - self.z_d.mul(1 - x_out)).prod(dim=-1)
        self.rule_activations = x_out
        if self.args.use_k_1 == 1:
            self.z_r = torch.sigmoid(self.beta)
        else:
             self.z_r = torch.sigmoid(self.beta*k_dict['k_beta'])
            # self.z_r = binary_concrete(self.beta, k=1, hard=hard_bc, use_noise=noise_factor==1, noise_factor=noise_factor)
        # else:
        #     self.z_r = torch.simoid(self.beta*k_dict['k_beta'])
            # self.z_r = binary_concrete(self.beta, k=k_dict['k_beta'], hard=hard_bc, use_noise=noise_factor == 1,
            #                            noise_factor=noise_factor)
        x_out = F.linear(x_out, self.weight * self.z_r.unsqueeze(0), self.bias)

        # if self.args.add_logreg:
        #     x_stand = torch.cat(x_stand, -1)
        #     x_out = x_out + F.linear(x_stand, self.logreg[:, :-1], self.logreg[:,-1])
        self.log_odds = x_out.squeeze(-1)
        return x_out.squeeze(-1)

    def init_params(self, init_args, device='cuda'):
        # self.alpha = nn.Parameter(torch.tensor(init_args['alpha_init'], device=device, dtype=torch.float))
        self.weight = nn.Parameter(torch.tensor(init_args['w_init'], device=device, dtype=torch.float))
        # Logistic regression bias
        self.bias = nn.Parameter(torch.tensor(init_args['bias_init'], device=device, dtype=torch.float))
        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.tensor(init_args['beta_init'], device=device, dtype=torch.float))
        # if self.args.add_logreg:
        #     self.logreg = nn.Parameter(torch.tensor(np.random.normal(0, 1, (1, self.num_feats + 1)), device=device, dtype=torch.float))

