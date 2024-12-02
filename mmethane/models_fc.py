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
        if module_dict is not None:
            self.num_otus = int(sum([m.num_otus for m in self.combo_modules]))
        else:
            self.module_names = list(x_in.keys())
            self.num_otus = int(sum(x_in.values()))

        self.fc = FC_NN(self.num_otus, h = args.h_sizes, p=self.args.dropout)

    def forward(self, x_dict, k_dict=None, hard_otu=False, hard_bc=False, noise_factor=1):
        # x_in = [x_dict[m] for m in self.module_names]
        
        x_out=[]
        self.order = []
        for i,name in enumerate(self.module_names):
            x = x_dict[name]
            x_out.append(x)
            self.order.extend([name] * x.shape[-1])
            
        x_out = torch.cat(x_out, -1)
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