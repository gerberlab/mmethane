import torch.nn as nn
from utilities.model_helper import *

# Lowest possible float
EPSILON = np.finfo(np.float32).tiny
class FC_NN(nn.Module):
    def __init__(self, L, h=[18, 12], p=0.2):
        super(FC_NN, self).__init__()
        hnew=[]
        for i,hh in enumerate(h):
            if hh==0:
                if i == 0:
                    hnew.append(int(np.sqrt(3*L)+2*np.sqrt(L/3)))
                else:
                    hnew.append(int(np.sqrt(L/3)))
            else:
                hnew.append(hh)
        h=hnew
        self.hidden_sizes = [L] + h + [1]
        self.fc_nn = nn.ModuleList()
        # self.fc_nn.append(nn.BatchNorm1d(L))
        for k in range(len(self.hidden_sizes) - 1):
            self.fc_nn.append(nn.Linear(self.hidden_sizes[k], self.hidden_sizes[k + 1], bias=True))
            if k <= len(h) - 1:
                self.fc_nn.append(nn.GELU())
                self.fc_nn.append(nn.Dropout(p))

        print(self.fc_nn)

    def forward(self, x):
        for layer in self.fc_nn:
            x = layer(x)
        # out = bias + (out * z).sum(1)
        return x
    
class featMDITRE(nn.Module):
    def __init__(self, num_otu_centers,dist, emb_dim, dtype, num_emb_to_learn=0, args=None, num_rules=None, num_feats=1):
        super(featMDITRE, self).__init__()
        self.num_otu_centers = num_otu_centers
        self.emb_dim = emb_dim
        self.register_buffer('dist', torch.from_numpy(dist))
        self.num_emb_to_learn = num_emb_to_learn
        self.dtype = dtype
        # self.eta = nn.Parameter(torch.tensor(num_otu_centers, emb_dim))
        # self.kappa = nn.Parameter(torch.tensor(num_otu_centers))

    def forward(self, x, k=1, hard=False, exp_kappa=False):
        if exp_kappa:
            kappa = self.kappa.exp().unsqueeze(-1)
        else:
            kappa = self.kappa.unsqueeze(-1)
        if self.emb_to_learn is not None:
            if self.dist.shape[0] != self.emb_to_learn.shape[0]:
                dist = (self.eta.unsqueeze(1) -
                        torch.cat((self.dist, self.emb_to_learn), 0)).norm(2,dim=-1)
            else:
                dist = (self.eta.unsqueeze(1) - self.emb_to_learn).norm(2, dim=-1)
            self.full_dist = dist
        else:
            dist = (self.eta.unsqueeze(1) - self.dist).norm(2, dim=-1)

        if hard:
            otu_wts_soft = torch.sigmoid((kappa - dist) * k)
            otu_wts_unnorm = (otu_wts_soft > 0.5).float() - otu_wts_soft.detach() + otu_wts_soft
        else:
            otu_wts_unnorm = torch.sigmoid((kappa - dist) * k)
        self.wts = otu_wts_unnorm
        if self.dtype == 'metabs':
            x = (torch.einsum('kj,sjt->skt', otu_wts_unnorm, x) + 1e-10) / (torch.sum(otu_wts_unnorm,-1).unsqueeze(0).unsqueeze(-1) + 1e-10)
        else:
            x = torch.einsum('kj,sjt->skt', otu_wts_unnorm, x)
        x = x.squeeze(-1)
        self.kappas = kappa
        self.emb_dist = dist
        self.x_out = x
        # if torch.isnan(self.thresh).any():
        #     print(self.thresh)
        #     raise ValueError('Nan in threshold!')
        # self.x_out = x
        return self.x_out

    def init_params(self, init_args,device = 'cuda'):
        if self.num_emb_to_learn > 0:
            self.emb_to_learn = nn.Parameter(torch.tensor(init_args['emb_init'], device=device, dtype=torch.float32))
        else:
            self.emb_to_learn = None
        self.eta = nn.Parameter(torch.tensor(init_args['eta_init'], device=device, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(init_args['kappa_init'], device=device, dtype=torch.float32))

class ComboMDITRE(nn.Module):
    def __init__(self, args, module_dict=None, x_in=None):
        super(ComboMDITRE, self).__init__()
        # self.met_args = args_met
        self.args = args
        # self.bug_args = args_bug
        if module_dict is not None:
            self.module_names, tmp_modules = zip(*module_dict.items())
            self.module_names = list(self.module_names)
            self.combo_modules = nn.ModuleList(list(tmp_modules))
        # self.num_rules = self.args.n_r
        if self.args.method!='full_fc':
            self.num_otus = int(sum([m.num_otu_centers for m in self.combo_modules]))
        else:
            if module_dict is not None:
                self.num_otus = int(sum([m.num_otus for m in self.combo_modules]))
            else:
                self.module_names = list(x_in.keys())
                self.num_otus = int(sum(x_in.values()))
        # self.alpha = nn.Parameter(torch.tensor(self.num_rules, self.num_otus))
        # self.weight = nn.Parameter(torch.tensor(self.num_rules, 1))
        # Logistic regression bias
        # self.bias = nn.Parameter(torch.tensor(1))
        # Parameter for selecting active rules
        # self.beta = nn.Parameter(torch.tensor(self.num_rules))
        self.fc = FC_NN(self.num_otus, h = args.h_sizes, p=self.args.dropout)

    def forward(self, x_dict, k_dict=None, hard_otu=False, hard_bc=False, noise_factor=1):
        # x_in = [x_dict[m] for m in self.module_names]
        
        x_out=[]
        self.order = []
        for i,name in enumerate(self.module_names):
            x = x_dict[name]
            if self.args.method!='full_fc':
                x_out.append(self.combo_modules[i](x.unsqueeze(-1), k=k_dict[name + '_k_otu'],hard=hard_otu,
                                                        exp_kappa = self.args.kappa_prior == 'log-normal'))
            else:
                x_out.append(x)
            self.order.extend([name] * x.shape[-1])
            
        x_out = torch.cat(x_out, -1)
        # x_out = self.fc(x_out, k=k_dict['k_beta'], hard=False, use_noise=True)
        if self.args.method!='full_fc':
            if self.args.use_k_1==1:
                self.z_d = binary_concrete(self.beta, k=1, hard=hard_bc,use_noise=noise_factor==1, noise_factor=noise_factor)
                # self.z_r = binary_concrete(self.beta, k=1, hard=hard_bc, use_noise=noise_factor==1, noise_factor=noise_factor)
            else:
                self.z_d = binary_concrete(self.beta, k=k_dict['k_alpha'], hard=hard_bc,use_noise=noise_factor==1, noise_factor=noise_factor)
                # self.z_r = binary_concrete(self.beta, k=k_dict['k_beta'], hard=hard_bc, use_noise=noise_factor==1, noise_factor=noise_factor)

            # x_out = (1 - self.z_d.mul(1 - x_out)).prod(dim=-1)
            
            x_out = self.fc(x_out*self.z_d.unsqueeze(0))
        else:
            x_out = self.fc(x_out)
        self.log_odds = x_out.squeeze(-1)
        return x_out.squeeze(-1)

    def init_params(self, init_args, device='cuda'):
        if len(init_args)>0:
            self.beta = nn.Parameter(torch.tensor(init_args['beta_init'], device=device, dtype=torch.float32))
        # Logistic regression bias
        # Parameter for selecting active rules
        # self.beta = nn.Parameter(torch.tensor(init_args['beta_init']).to(device))
        return