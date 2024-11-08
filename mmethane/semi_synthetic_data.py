import sys
import os

sys.path.append(os.path.abspath(".."))
from utilities.util import calculate_radii_prior
from plot_synthetic_data import *
from sklearn.model_selection import train_test_split
from utilities.model_helper import TruncatedNormal
import copy
import torch
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multinomial import Multinomial
import matplotlib
from collections import Counter
from utilities.data_utils import get_epsilon
from helper import seed_everything_custom
matplotlib.use('MacOSX')

class GenerateData():
    def __init__(self, met_data_dict, otu_data_dict=None, seq_tree = None, nonzero_filter_perc = None,
                 cov_filter_perc = None, transform_data=True, seed=0, metab_pert_params=(1.2,0.5), metab_ctrl_params=(0,1),
                 otu_pert_params=(-3,0.5), otu_ctrl_params=(-6,0.5), disp_param=36000, meas_var=0.1, met_ctrl_pert = None,
                 otu_ctrl_pert=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        seed_everything_custom(seed)
        self.metab_pert_params=metab_pert_params
        self.metab_ctrl_params = metab_ctrl_params
        self.otu_pert_params=otu_pert_params
        self.otu_ctrl_params = otu_ctrl_params
        self.dispersion=disp_param
        self.met_data = met_data_dict['X'].loc[met_data_dict['y']==1,:]
        keep_feats = list(set(self.met_data.columns.values).intersection(set(met_data_dict['distances'].index.values)))
        self.met_data = self.met_data[keep_feats]
        self.met_distances = met_data_dict['distances'][keep_feats].loc[keep_feats]
        self.meas_var = meas_var
        # d, met_emb_locs, met_dmat, pval = test_d_dimensions([20], self.met_distances, self.seed)
        # print(f'for metabolite embedding, dimension {d} returned, p-value is {pval}')
        # self.met_emb_locs = pd.DataFrame(met_emb_locs, index=self.met_data.columns.values)
        self.met_dmat = self.met_distances
        # self.met_dmat = pd.DataFrame(met_dmat, index = self.met_data.columns.values, columns = self.met_data.columns.values)
        self.metab_radii_params = calculate_radii_prior(met_data_dict, self.met_dmat, 'metabs', 1)
        self.met_ctrl_pert = met_ctrl_pert
        self.otu_ctrl_pert = otu_ctrl_pert

        if otu_data_dict is not None:
            self.otu_data = otu_data_dict['X']
            # .loc[otu_data_dict['y']==0,:]
            self.R = np.mean(self.otu_data.sum(1))
            keep_feats = list(
                set(self.otu_data.columns.values).intersection(set(otu_data_dict['distances'].index.values)))
            self.otu_data = self.otu_data[keep_feats]
            self.otu_distances = otu_data_dict['distances'][keep_feats].loc[keep_feats]
            # d, otu_emb_locs, self.otu_dmat, pval = test_d_dimensions(np.arange(2, 30), self.otu_distances, self.seed)
            # print(f'for otu embedding, dimension {d} returned, p-value is {pval}')
            # self.otu_emb_locs = pd.DataFrame(otu_emb_locs, index=self.otu_data.columns.values)
            self.otu_dmat = self.otu_distances
            self.taxa_tree = seq_tree


            self.sample_ids = list(set(self.met_data.index.values).intersection(set(self.otu_data.index.values)))
            self.otu_data = self.otu_data.loc[self.sample_ids]
            self.met_data = self.met_data.loc[self.sample_ids]
            self.otu_radii_params = calculate_radii_prior(otu_data_dict, self.otu_dmat, 'otus', 1)
            # self.otu_dmat = pd.DataFrame(self.otu_dmat, index=self.otu_data.columns.values, columns=self.otu_data.columns.values)
        else:
            self.sample_ids = list(self.met_data.index.values)


    def ix_sample_from_data(self, N, data):
        choices = data
        if N > len(data):
            ix_a = self.rng.choice(choices, len(data), replace = False)
            ix_b = self.rng.choice(choices, N-len(data), replace = True)
            ix = np.concatenate([ix_a,ix_b])
        else:
            ix = self.rng.choice(choices, N, replace=False)
        return ix


    # def save_as_pickle(self, filename, dataset):
    #     syndata = dict()
    #     syndata['n_variables'] = self.semi_syn_dat.shape[1]
    #     syndata['variable_names'] = self.semi_syn_dat.columns.values
    #     syndata['variable_tree'] = dataset['variable_tree']
    #     syndata['X'] = self.semi_syn_dat
    #     syndata['y'] = self.labels
    #     syndata['distances'] = self.dmat
    #     syndata['met_ids'] = self.perturbed_mets
    #
    #     with open(filename, 'wb') as f:
    #         pkl.dump(syndata, f)


    def perturb_data(self, N, case='and_both',features=None):
        str_cases = ['metabs','and_metabs','or_metabs','and_both','or_both','both', 'both_metabs','otus','both_and','both_or', 'and_otus','both_and_TEST', 'both_and_TESTSUB']
        if isinstance(case, str) and case not in str_cases:
            raise ValueError(f'Please provide valid option for input case. Options are {",".join(str_cases)}')
        # elif isinstance(case, int):
        #     if case not in int_cases:
        #         raise ValueError(f'Please provide valid option for input case. Options are {",".join(int_cases)}')
        #     else:
        #         case = case_map[case]

        self.subject_ixs = self.ix_sample_from_data(N, self.sample_ids)
        new_subj_ixs = list(np.arange(len(set(self.subject_ixs))))
        # self.subj_ixs_to_id = pd.Series(self.subject_ixs, index = new_subj_ixs)
        met_data = pd.DataFrame(self.met_data.loc[list(set(self.subject_ixs))].values, index = new_subj_ixs,
                                columns = self.met_data.columns.values)
        if self.otu_data is not None:
            otu_data = pd.DataFrame(self.otu_data.loc[list(set(self.subject_ixs))].values, index = new_subj_ixs,
                                columns = self.otu_data.columns.values)
        if len(set(self.subject_ixs))<len(self.subject_ixs):
            ct_dict = Counter(self.subject_ixs)
            curr_max = max(new_subj_ixs)
            for ix, val in ct_dict.items():
                if val>1:
                    # print('RESAMPLING DATA')
                    new_sub_vals = np.stack([self.met_data.loc[ix].values]*(val-1), axis=0)
                    # new_sub_resampled = TruncatedNormal(new_sub_vals, 2, a=0, b=new_sub_vals.flatten().max()).sample()
                    # pert_mu, pert_var = convert_to_log_normal(new_sub_vals, self.meas_var*1e-6)
                    # new_sub_resampled = Normal(torch.Tensor(pert_mu), torch.Tensor(np.sqrt(pert_var))
                    #                    ).sample().exp().numpy()
                    new_sub = pd.DataFrame(new_sub_vals, index=[curr_max + 1 + i for i in np.arange(val-1)],
                                           columns=self.met_data.columns.values)
                    met_data = pd.concat([met_data, new_sub])
                    if self.otu_data is not None:
                        new_sub = pd.DataFrame(np.stack([self.otu_data.loc[ix].values] * (val-1), axis=0),
                                               index=[curr_max + 1 + i for i in range(val-1)],
                                               columns=self.otu_data.columns.values)
                        otu_data = pd.concat([otu_data, new_sub])
                        # otus resampled w meas var during perturbation
                    new_subj_ixs.extend([curr_max + 1 + i for i in range(val-1)])
                    curr_max+=val
        self.subj_ixs_to_id = pd.Series(self.subject_ixs, index=new_subj_ixs)
        self.subject_ixs = new_subj_ixs
        np.random.shuffle(self.subject_ixs)
        # met_data = pd.DataFrame(self.met_data.iloc[self.subject_ixs].values, index = self.subject_ixs,
        #                         columns = self.met_data.columns.values)
        met_data_ = met_data.copy()
        met_data_final = met_data_.copy()
        if self.otu_data is not None:
            # otu_data = pd.DataFrame(self.otu_data.iloc[self.subject_ixs].values, index=self.subject_ixs,
            #                         columns = self.otu_data.columns.values)
            otu_data_ = otu_data.copy()
            self.otu_syn_data=copy.deepcopy(otu_data)
            total = self.otu_syn_data.sum(1).median()
            lod=10/total
            print('lod',lod)
            # (otu_data_.T / total_counts.T).T
            otu_ra = ((otu_data).T/(otu_data.sum(1).T)).T
            otu_counts =copy.deepcopy(otu_data)
        else:
            otu_data_=None
            self.otu_syn_data=None

        if features is None:
            features=[]
            if 'both' in case or 'metabs' in case:
                met_clade = self.pick_metab_clade()
                while self.check_metab_clade(met_clade) is False:
                    met_clade = self.pick_metab_clade()
                features.append({'metabs':met_clade})
            if case=='and_metabs':
                second_met_clade = self.pick_metab_clade()
                while self.check_metab_clade(second_met_clade) is False or len(set(second_met_clade).intersection(set(met_clade)))>0:
                    second_met_clade = self.pick_metab_clade()
                features.append({'metabs':second_met_clade})
            if 'otus' in case or 'both' in case:
                otu_clade = self.pick_otu_clade()
                while self.check_otu_clade(otu_clade) is False:
                    otu_clade = self.pick_otu_clade()
                features.append({'otus': otu_clade})
            if case == 'and_otus':
                second_otu_clade = self.pick_otu_clade()
                while self.check_otu_clade(second_otu_clade) is False or len(set(second_otu_clade).intersection(set(otu_clade)))>0:
                    second_otu_clade = self.pick_otu_clade()
                features.append({'otus':second_otu_clade})

        pert_ixs, ctrl_ixs = train_test_split(self.subject_ixs, test_size=0.5)
        self.ctrl_ixs = ctrl_ixs
        y = pd.Series(data=np.zeros((len(self.subject_ixs))), index=self.subject_ixs)
        y[pert_ixs]=1
        if 'and' in case:
            pert_ctrl_1, pert_ctrl_2 = train_test_split(ctrl_ixs, test_size=0.5)
            pert_ixs_1=np.concatenate([pert_ixs, pert_ctrl_1])
            pert_ixs_2 = np.concatenate([pert_ixs, pert_ctrl_2])
            perturbed_per_feat = [pert_ixs_1, pert_ixs_2]
            ctrl_per_feat = [pert_ctrl_2, pert_ctrl_1]

        else:
            perturbed_per_feat=[pert_ixs]
            ctrl_per_feat=[ctrl_ixs]
            if len(features)>1:
                perturbed_per_feat = [pert_ixs, pert_ixs]
                ctrl_per_feat = [ctrl_ixs, ctrl_ixs]

        self.perturbed_per_feat = perturbed_per_feat
        self.ctrl_per_feat = ctrl_per_feat
        feats_pert= {'metabs':[], 'otus':[]}
        otu_perts=[]
        otu_gammas=[]
        otu_feats = []
        otu_pert_sum = pd.DataFrame(np.zeros((len(self.subject_ixs),1)), index=self.subject_ixs)
        if 'metabs' in case or 'both' in case:
            eps = get_epsilon(met_data_.values)
            mu = np.mean(np.log(met_data_.values + eps), 0)
            std = np.std(np.log(met_data_.values + eps), 0)
            z = pd.DataFrame((np.log(met_data_.values + eps) - mu) / std, index=met_data_.index.values,
                             columns=met_data_.columns.values)
        for i, (ixs_to_perturb, ixs_to_ctrl, feats_to_perturb) in enumerate(zip(perturbed_per_feat, ctrl_per_feat, features)):

            assert(len(list(set(ixs_to_ctrl).intersection(set(ixs_to_perturb))))==0)
            # tmp = copy.deepcopy(feats_to_perturb)
            # tmp['ixs_ctrl'] = ixs_to_ctrl
            # tmp['ixs_pert'] = ixs_to_perturb
            if 'metabs' in feats_to_perturb.keys():
                subj_to_perturbation = {ix: self.metab_ctrl_params for ix in ixs_to_ctrl}
                if self.otu_ctrl_pert is not None:
                    ctrl_ixs_to_perturb = [ix for ix in ixs_to_perturb if ix in self.ctrl_ixs]
                    pert_ixs_to_perturb = [ix for ix in ixs_to_perturb if ix not in self.ctrl_ixs]
                    subj_to_perturbation.update({ix:self.metab_pert_params for ix in pert_ixs_to_perturb})
                    subj_to_perturbation.update({ix:self.met_ctrl_pert for ix in ctrl_ixs_to_perturb})
                    print('DOUBLE PERTURBING CTRL')
                else:
                    subj_to_perturbation.update({ix: self.metab_pert_params for ix in ixs_to_perturb})
                feats_pert['metabs'].append(list(feats_to_perturb['metabs']))
                for s in self.subject_ixs:
                    pert = subj_to_perturbation[s]
                    for f in feats_to_perturb['metabs']:
                        z.loc[s,f] = Normal(torch.tensor(pert[0]), torch.tensor(pert[1])).sample().numpy()


            else:
                feats_pert['otus'].append(list(feats_to_perturb['otus']))
                subj_to_perturbation = {ix: self.otu_ctrl_params for ix in ixs_to_ctrl}
                if self.otu_ctrl_pert is not None:
                    ctrl_ixs_to_perturb = [ix for ix in ixs_to_perturb if ix in self.ctrl_ixs]
                    pert_ixs_to_perturb = [ix for ix in ixs_to_perturb if ix not in self.ctrl_ixs]
                    subj_to_perturbation.update({ix:self.otu_pert_params for ix in pert_ixs_to_perturb})
                    subj_to_perturbation.update({ix:self.otu_ctrl_pert for ix in ctrl_ixs_to_perturb})
                    print('DOUBLE PERTURBING CTRL')
                else:
                    subj_to_perturbation.update({ix: self.otu_pert_params for ix in ixs_to_perturb})

                otu_ra = pd.DataFrame(self.sample_otu_trunc_norm(otu_ra.values, 0.3, lod),
                                      index=otu_ra.index.values, columns=otu_ra.columns.values)
                feats = feats_to_perturb['otus']
                per_subj_pert = {}
                per_subj_gamma = {}
                ct = 0
                for s in self.subject_ixs:
                    ra_s = otu_ra.loc[s,:].copy()
                    per_subj_gamma[s] = ra_s[feats]/ra_s[feats].sum()
                    pert = subj_to_perturbation[s]
                    # if s in self.ctrl_ixs:
                    #     print(f'Control {s}', pert)
                    # else:
                    #     print(f'Case {s}', pert)
                    p = np.exp(Normal(torch.tensor(pert[0]), torch.tensor(pert[1])).sample().numpy())
                    if p>=1:
                        p=0.99
                    p_wNoise = self.sample_otu_trunc_norm(p, 0.3, lod)
                    while p_wNoise>=0.99:
                        p = np.exp(Normal(torch.tensor(pert[0]), torch.tensor(pert[1])).sample().numpy())
                        if p>=1:
                            continue
                        p_wNoise = self.sample_otu_trunc_norm(p, 0.3, lod)

                    if len(otu_perts)>0:
                        past_per_subj_perts = otu_perts[-1]
                        ct=0
                        while p_wNoise >= 0.99-past_per_subj_perts[s]:
                            p = np.exp(Normal(torch.tensor(pert[0]), torch.tensor(pert[1])).sample().numpy())
                            if p>=1:
                                continue
                            p_wNoise = self.sample_otu_trunc_norm(p, 0.3, lod)

                    per_subj_pert[s] = p_wNoise
                otu_perts.append(pd.Series(per_subj_pert))
                otu_gammas.append(per_subj_gamma)
                otu_feats.append(feats)


        if len(otu_perts)>0:
            ra_out = otu_ra.copy()
            ra_wNoise = pd.DataFrame(self.sample_otu_trunc_norm(ra_out.values, 0.3, lod),
                                     index = otu_ra.index.values, columns = otu_ra.columns.values)
            for clade in otu_feats:
                ra_wNoise[clade] = 0
            ra_wNoise = (ra_wNoise.T/(ra_wNoise.sum(1).T)).T
            df_perts = pd.concat(otu_perts, axis=1)

            print(df_perts.shape)
            total_pert = df_perts.sum(1)
            ra_renorm = (ra_wNoise.T*(1-total_pert.T)).T
            per_subj_counts = {}
            for s in self.subject_ixs:
                for i,clade in enumerate(otu_feats):
                    ra_renorm.loc[s,clade] = otu_gammas[i][s]*otu_perts[i].loc[s]
                ra_renorm.loc[s,:] = ra_renorm.loc[s,:]/ra_renorm.loc[s,:].sum()
                counts = self.sample_counts(ra_renorm.loc[s,:], s)
                per_subj_counts[s] = counts
            otu_counts = pd.DataFrame.from_dict(per_subj_counts, columns =otu_ra.columns.values, orient='index')
            otu_ra = (otu_counts.T/(otu_counts.sum(1).T)).T

        if 'metabs' in case or 'both' in case:
            meas_error = Normal(torch.tensor(0.0), torch.tensor(self.meas_var)).sample(z.shape).numpy()
            z_ = z + meas_error
            met_data_final = np.exp(z_ * std + mu)



        return met_data_final.loc[self.subject_ixs], (otu_counts.loc[self.subject_ixs], otu_ra.loc[self.subject_ixs]), y.loc[self.subject_ixs], feats_pert

    def sample_counts(self, ra, s):
        tmp = abs(ra.values).astype(np.float64)
        tmp = tmp + 0.01 * tmp[tmp > 0].min()
        total = self.otu_syn_data.loc[s].sum()
        probs = Dirichlet(torch.tensor(tmp*self.dispersion)).sample().numpy()
        counts = total*probs
        # counts = Multinomial(total_count=int(total), probs=torch.tensor(probs)).sample().numpy()
        return counts

    def sample_otu_trunc_norm(self, x, theta, lod):
        # out=x
        out = TruncatedNormal(x, theta*(x+lod), 0,1).sample().numpy()
        return out

    def pick_metab_clade(self):
        met_clade = self.rng.choice(self.met_data.columns.values, 1)
        kappa_max = self.met_dmat.values.flatten().max() + 0.01*self.met_dmat.values.flatten().max()
        trunc_norm = TruncatedNormal(self.metab_radii_params['mean'], self.metab_radii_params['var']*10, 0, kappa_max)
        r=trunc_norm.sample([1]).numpy()
        mets_keep = self.met_dmat.columns.values[(self.met_dmat.loc[met_clade]<=r.item()).squeeze()]
        return mets_keep

    def check_metab_clade(self, metab_clade):
        # if len(metab_clade)>self.met_data.shape[0]*.25 or len(metab_clade)==1:
        tmp = [m.split('__')[-1] for m in metab_clade]
        if len(metab_clade)>30 or len(metab_clade)<10 or 'Butanoic acid' in tmp:
            return False
        avg_vals = np.mean(self.met_data,0)
        low_50 = np.percentile(avg_vals, 25)
        high_50 = np.percentile(avg_vals, 75)
        met_vals = np.mean(np.mean(self.met_data[metab_clade],1))
        if met_vals<low_50 or met_vals>high_50:
            return False
        return True

    def pick_otu_clade(self):
        otu_clade = self.rng.choice(self.otu_data.columns.values, 1)
        if self.taxa_tree is None:
            print('Warning, using provided distances to select OTU group for pertubation. To perturb a clade, provide a phylogenetic tree')
            kappa_max = self.otu_dmat.values.flatten().max() + 0.01*self.otu_dmat.values.flatten().max()
            trunc_norm = TruncatedNormal(self.otu_radii_params['mean'], self.otu_radii_params['var'], 0, kappa_max)
            r=trunc_norm.sample([1]).numpy()
            otus_keep = self.otu_dmat.columns.values[(self.otu_dmat.loc[otu_clade]<=r.item()).squeeze()]
        else:
            node = self.taxa_tree.search_nodes(name=otu_clade)
            parent =node[0].up
            clade = parent.get_leaves()
            while len(clade)<5:
                parent = parent.up
                clade= parent.get_leaves()
            otus_keep = [c.name for c in clade]
        return otus_keep

    def check_otu_clade(self, otu_clade):
        if len(otu_clade)>30 or len(otu_clade)<5:
            return False
        otu_rel_abun = self.otu_data.divide(self.otu_data.sum(1), axis='index')
        avg = np.mean(otu_rel_abun[otu_clade].sum(0).values)
        if avg<0.001 or avg>0.20:
            return False
        return True



def calc_effect_size(gp1, gp2):
    # avg_gp1 = np.mean(gp1, 1)
    # avg_gp2 = np.mean(gp2, 1)
    return np.mean((np.mean(gp1) - np.mean(gp2))/np.sqrt((np.var(gp1) + np.var(gp2))/2))


def get_fold_change(data, targets, features, dtype=None):
    if isinstance(features, list) and dtype=='otus':
        mu0=np.mean(data[features].loc[targets==0].sum(1))
        mu1 = np.mean(data[features].loc[targets == 1].sum(1))
        var0 = np.var(data[features].loc[targets == 0].sum(1))
        var1 = np.var(data[features].loc[targets == 1].sum(1))
    elif isinstance(features, list) and dtype!='otus':
        mu0 = np.mean(data[features].loc[targets == 0].mean(1))
        mu1 = np.mean(data[features].loc[targets == 1].mean(1))
        var0 = np.var(data[features].loc[targets == 0].mean(1))
        var1 = np.var(data[features].loc[targets == 1].mean(1))
    else:
        mu0 = np.mean(data[features].loc[targets == 0])
        mu1 = np.mean(data[features].loc[targets == 1])
        var0 = np.var(data[features].loc[targets == 0])
        var1 = np.var(data[features].loc[targets == 1])
    return (mu0,var0),(mu1,var1)

# def get_log_normal_params(data, targets, features, dtype=None):
# #     data_log = pd.DataFrame(np.log(data + get_epsilon(data)), columns = data.columns.values, index = data.index.values)
#     (mu0,var0),(mu1,var1)=get_fold_change(data,targets,features,dtype=dtype)
#
#     mu0_ln, var0_ln = convert_to_log_normal(mu0, var0)
#     mu1_ln, var1_ln = convert_to_log_normal(mu1, var1)
#     if np.isnan(mu0_ln):
#         mu0_ln = mu1_ln
#         var0_ln = var1_ln
#     if np.isnan(mu1_ln):
#         mu1_ln, var1_ln = mu0_ln, var0_ln
#     return (mu0_ln, var0_ln), (mu1_ln, var1_ln)


def get_new_ixs(N, y, gp_dict = None):
    if gp_dict is not None and len(gp_dict)>0:
        gp_rev_dict={}
        for k,v in gp_dict.items():
            for vi in v:
                gp_rev_dict[vi]=k
    n0, n1 = 0, 0
    ixs = []
    it = 0
    gp_ct_0, gp_ct_1 = 0,0
    while (n0 < (N / 2) or n1 < (N / 2)) and len(ixs)<N:
        # try:
        #     new_val = y.iloc[it]
        # except:
        #     print('debug')
        try:
            new_ix = y.index.values[it]
            new_val = y.loc[new_ix]
        except:
            if len(ixs)<N:
                ixs_unused = list(set(y.index.values)-set(ixs))
                ixs.extend(ixs_unused[:(N-len(ixs))])
            print('len ixs', len(ixs))
            print('len set(ixs)', len(set(ixs)))
            print('N',N)
            print('gp_ct_0', gp_ct_0)
            print('gp_ct_1', gp_ct_1)
            print('n0', n0)
            print('n1', n1)

        if len(gp_dict.keys())>1:
            try:
                if new_val==0:
                    gp_ct_0 += 1-gp_rev_dict[new_ix]
                    gp_ct_1 += gp_rev_dict[new_ix]
                if gp_ct_0>(np.ceil(n0/2)):
                    gp_ct_0 = gp_ct_0-1
                    it += 1
                    continue
                if gp_ct_1> np.ceil(n1/2):
                    gp_ct_1 = gp_ct_1-1
                    it += 1
                    continue
            except:
                it += 1
                if it > len(y):
                    break
                continue

        n0 += 1 - new_val
        n1 += new_val
        if n0 > (N / 2):
            n0 = n0 - 1
            it += 1
            continue
        if n1 > (N / 2):
            n1 = n1 - 1
            it += 1
            continue
        ixs.append(new_ix)
        it += 1
        if it>len(y):
            break
    if gp_dict is not None and len(gp_dict)>0:
        fin_gp_dict = {}
        for ix in ixs:
            if y.loc[ix]==0:
                try:
                    fin_gp_dict[ix] = gp_rev_dict[ix]
                except:
                    import pdb; pdb.set_trace()

        print(Counter(list(fin_gp_dict.values())))
    return ixs

def subsample_generated_data(N_generated=1000, N_subjs=[750],cases = None, seed = None,
                             path = '/Users/jendawk/Dropbox (MIT)/microbes-metabolites/datasets/SEMISYN/processed/', shuffle=False):
    for dir in os.listdir(path):

        if dir == '.DStore':
            continue

        if str(N_generated) not in dir:
            continue

        dir_path = os.path.join(path, dir) + '/'
        if seed is None:
            seed = dir_path.split('_')[-1].replace('/','')
        else:
            if int(dir_path.split('_')[-1].replace('/','')) not in set(seed):
                continue

        # import pdb; pdb.set_trace()
        seed_s = dir_path.split('_')[-1]
        n_subj = dir_path.split('_')[-2]
        assert(n_subj==str(N_generated))
        case = '_'.join(dir_path.split('_')[:-2]).split('/')[-1]
        print(case)
        # import pdb; pdb.set_trace()
        if case not in set(cases):
            continue
        dataset = {}
        dataset['metabs'] = pd.read_pickle(dir_path + 'mets.pkl')
        dataset['otus'] = pd.read_pickle(dir_path + 'seqs.pkl')
        if 'metabs' in dir_path or 'both' in dir_path:
            k0 = 'metabs'
        elif 'otus' in dir_path or 'both' in dir_path:
            k0 = 'otus'
        pert = pd.read_pickle(dir_path + 'perturbed.pkl')
        with open(dir_path + 'subjs_perturbed_per_feat.txt','r') as f:
            t=f.readlines()


        gp_dict = {}
        i = 0
        for line in t:
            if line.startswith('Un-perturbed: '):
                gp_dict[i] = [int(l) for l in line.split('Un-perturbed: ')[-1].replace('\n','').split(', ') if l!='']
                i+=1
        # subjs_pert_per_feat =
        for N in N_subjs:
            new_data = copy.deepcopy(dataset)
            old_y = copy.deepcopy(dataset[k0]['y'])
            if shuffle:
                old_y = old_y.sample(frac=1)
            if case=='both':
                gp_dict = {}
            # import pdb; pdb.set_trace()
            new_ixs = get_new_ixs(N, old_y, gp_dict)
            for k in dataset.keys():
                if len(dataset[k]) != 0:
                    new_data[k]['y'] = dataset[k]['y'].loc[new_ixs]
                    new_data[k]['X'] = dataset[k]['X'].loc[new_ixs]
                    new_data[k]['perturbed'] = pert[k]
                    y = new_data[k]['y']

            subj_ix_to_id = pd.read_csv(dir_path + '/subj_ix_to_id.csv', index_col=[0])

            new_path = os.path.join(path, case + '_' + str(N) + '_' + seed_s)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            else:
                if 'mets.pkl' in os.listdir(new_path) or 'seqs.pkl' in os.listdir(new_path):
                    # pdb.set_trace()
                    continue
                else:
                    print('ERROR')
                    return

            subj_ix_to_id.loc[new_ixs].to_csv(new_path + '/subj_ix_to_id.csv', index=True)
            with open(new_path + '/mets.pkl', 'wb') as f:
                pkl.dump(new_data['metabs'], f)
            with open(new_path + '/seqs.pkl', 'wb') as f:
                pkl.dump(new_data['otus'], f)
            with open(new_path + '/perturbed.pkl', 'wb') as f:
                pkl.dump(pert, f)

            with open(new_path + '/subjs_perturbed_per_feat.txt', 'w') as f:
                f.writelines(t)


            ft_subjs = {}
            ft_key = None
            for ti in t:
                if ti == '\n':
                    continue
                if len(ti.split(':')) == 1:
                    ft_key = ti
                    ft_subjs[ti] = {}
                elif ft_key is not None:
                    subj, slist = ti.split(':')
                    tmp = (slist.strip() + ' ').split(', ')[:-1]
                    sub_list = [int(s) for s in tmp]
                    sub_list_filt = list(set(sub_list).intersection(new_ixs))
                    ft_subjs[ft_key][subj] = sub_list_filt

            with open(new_path + '/subjs_perturbed_per_feat_filtered.txt', 'w') as f:
                for outer_label,v in ft_subjs.items():
                    f.write(outer_label)
                    for inner_label, subj_list in v.items():
                        line = inner_label + ': ' + ', '.join([str(s) for s in subj_list]) + '\n'
                        f.write(line)

            y.to_csv(new_path + '/subj_labels.csv')

            with open(new_path + '/perturbed.txt', 'w') as f:
                for key in pert.keys():
                    f.write(key + '\n')
                    for v in pert[key]:
                        for vi in v:
                            f.write(vi + '\n')
                        f.write('\n\n')
                    f.write('\n')

            data_ls = {}
            if len(pert['metabs']) > 0:
                met_trans = transform_func(new_data['metabs']['X'])
                plot_data_hists(met_trans, pert['metabs'], new_data['metabs']['y'], fig_path=new_path + '/metabs_')
                plot_data_hists_agg(met_trans, pert['metabs'], new_data['metabs']['y'], fig_path=new_path + '/metabs_')
                (fig, ax), pval = dot_plots(new_data['metabs']['X'], pert['metabs'], new_data['metabs']['y'], 'metabs')
                fig.savefig(new_path + '/metabs_dot_plot.pdf')

                data_ls['metabs'] =met_trans[sum(pert['metabs'], [])]
            if len(pert['otus']) > 0:
                otu_rel_abun = (new_data['otus']['X'].T / (new_data['otus']['X'].sum(1).T)).T
                data_ls['otus'] = otu_rel_abun[sum(pert['otus'], [])]
                plot_data_hists(otu_rel_abun, pert['otus'], new_data['otus']['y'], fig_path=new_path + '/otus_')
                plot_data_hists_agg(otu_rel_abun, pert['otus'], new_data['otus']['y'], fig_path=new_path + '/otus_')
                (fig, ax), pval = dot_plots(otu_rel_abun, pert['otus'], new_data['otus']['y'], 'otus')
                if isinstance(ax,np.ndarray):
                    for axi in ax:
                        axi.set_yscale('log')
                else:
                    ax.set_yscale('log')
                fig.savefig(new_path + '/otus_dot_plot.pdf')

            plot_heatmap(data_ls, new_data[k0]['y'], fig_path=new_path + '/')

def add_subjs_to_semisyn(N_to_add,gd,case,seed,base_dir='../datasets/SEMISYN/processed/',N_orig=128):
    folder_path = f'{base_dir}{case}_{N_orig}_{seed}'
    pert = pd.read_pickle(f'{folder_path}/perturbed.pkl')
    pert_ls = []
    if len(pert['metabs'])==0:
        dtype=['otus']
        pert_ls.extend([{'otus': l} for l in pert['otus']])
    elif len(pert['otus'])==0:
        dtype=['metabs']
        pert_ls.extend([{'metabs':l} for l in pert['metabs']])
    else:
        dtype=['metabs','otus']
        pert_ls.extend([{'otus': l} for l in pert['otus']])
        pert_ls.extend([{'metabs': l} for l in pert['metabs']])
    dataset = {}
    dataset['metabs']=pd.read_pickle(f'{folder_path}/mets.pkl')
    dataset['otus'] = pd.read_pickle(f'{folder_path}/seqs.pkl')

    met_syn_data, (otu_syn_cts, otu_syn_ra), y, pert_dict = gd.perturb_data(N_to_add, case=case, features=pert_ls)
    assert(pert_dict==pert)
    if 'metabs' in dtype:
        subjs_old = dataset['metabs']['y'].index.values
        y_old = dataset['metabs']['y']
    else:
        subjs_old = dataset['otus']['y'].index.values
        y_old = dataset['otus']['y']
    subjs_new = y.index.values
    new_ixs = np.arange(np.max(subjs_old)+1,np.max(subjs_old)+len(subjs_new)+1)
    assert(len(new_ixs)==len(subjs_new))
    y_new = copy.deepcopy(y)
    y_new.index = new_ixs
    y_new = pd.concat([y_old, y_new])
    mets_total,otu_cts_tot,otu_ra_tot={},{},{}
    if 'metabs' in dtype:
        mets_new = met_syn_data.reset_index(drop=True).set_index([pd.Index(new_ixs)])
        mets_total = pd.concat([dataset['metabs']['X'], mets_new])
    if 'otus' in dtype:
        otu_cts = otu_syn_cts.reset_index(drop=True).set_index([pd.Index(new_ixs)])
        # otu_ra = otu_syn_ra.reset_index(drop=True).set_index([pd.Index(new_ixs)])

        otu_cts_tot = pd.concat([dataset['otus']['X'], otu_cts])
        # (otu_counts.T / (otu_counts.sum(1).T)).T
        otu_ra_tot = (otu_cts_tot.T/(otu_cts_tot.sum(1).T)).T
    return mets_total, (otu_cts_tot, otu_ra_tot),y_new,pert
    # y_new = y.reset_index(drop=True).set_index([pd.Index(new_ixs)])






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N','--Num_Subjs', type=int, default=[1000],nargs='+')
    parser.add_argument('--N_sub', type=int, default=[24], nargs='+')
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    parser.add_argument('--seeds', type=int, default=[0,1,2,3,4,5,6,7,8,9], nargs='+')
    parser.add_argument('--cases', type=str, default=[
                                                      'metabs'], nargs='+',
                        choices=['both_metabs','and_metabs','and_otus','both',
                                 'metabs','otus','both_and','both_and_TEST', 'both_and_TESTSUB']   )
    parser.add_argument('--path', type=str, default='../datasets/SEMISYN/')
    parser.add_argument('--fingerprint', type=str, default='pubchem')
    parser.add_argument('--subsample', action='store_true')
    parser.add_argument('--add_subjs', action='store_true')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--N_add',type=int, default=500)
    args = parser.parse_args()

    # if args.easy:
    #     if args.path[-1]=='/':
    #         args.path = args.path[:-1]
    #     args.path = args.path + '_EASY/'

    if args.subsample:
        print('SUBSAMPLE')
        if isinstance(args.Num_Subjs, list):
            N=args.Num_Subjs[0]
        else:
            N=args.Num_Subjs
        subsample_generated_data(N_generated=N, N_subjs=args.N_sub, seed = args.seeds, cases=args.cases,
                                 path=args.path + 'processed/')
    else:
        # met_pert= (6.7431009448514585, 2.7575136493529713)
        # met_ctrl=(10.19966716880961, 1.5193325971672067)
        # met_ctrl = (-1.2745, 0.33830769399541655)
        # if args.easy:





        # else:
        #     met_ctrl=(-4.599783307872394, 6.240864776334834)
        #     met_pert=(1.2479213845220276, 0.08319314061742872)
        #
        #     otu_ctrl = (-15.795280789339152, 13.996933769242444)
        #     otu_pert = (-2.30272838334923, 0.9554554154588305)

        for seed in args.seeds:
        # seed=2
            # mets_data = pd.read_pickle('../datasets/ERAWIJANTARI/processed/erawijantari_pubchem/mets.pkl')
            # seqs_data = pd.read_pickle('../datasets/ERAWIJANTARI/processed/erawijantari_ra/seqs.pkl')
            # with open('../datasets/cdi/processed/week_1_counts/seqs.pkl','rb') as f:
            #     seqs_data = pkl.load(f)
            for Num_Subjs in args.Num_Subjs:

                path = args.path+ '/' + 'processed' + '/'

                if not os.path.isdir(path):
                    os.mkdir(path)
                for case in args.cases:
                    met_ctrl =(-0.376432, 1)
                    met_pert = (0.483984, 1)
                    met_ctrl_pert = (0.483984*2, 1)
                    # fc = (0.724/0.203)*2
                    # met_ctrl_pert = (0.019510*2, 0.015949)
                    # met_ctrl_pert = (0.234393195118684, 0.6277316084390275)


                    otu_ctrl = (-6.0,1)
                    otu_pert = (-2.0,1)
                    # fc = (-2.7+0.452)/0.452
                    # mnew = -2.7/(fc+1)
                    # otu_pert = (-3.9123546667410976 - 3.9123546667410976*0.5, 0.14431801531943464)
                    otu_ctrl_pert=(-2.0*0.5,1)
                    # otu_ctrl_pert = (-3.9123546667410976 + 3.9123546667410976*0.1, 0.14431801531943464)




                    mets_data = pd.read_pickle(f'../datasets/WANG/processed/wang_{args.fingerprint}/mets.pkl')
                    #
                    seqs_data = pd.read_pickle('../datasets/WANG/processed/wang_cts/seqs.pkl')
                    # dist_mat = mets_data['distances']
                    mets_data['X'] = mets_data['X'][mets_data['distances'].index.values]
                    seqs_data['X'] = seqs_data['X'][seqs_data['distances'].index.values]
                    seqs_data_ra = (seqs_data['X'].T / seqs_data['X'].sum(1).T).T
                    targets = mets_data['y']
                    gd = GenerateData(mets_data, seqs_data, seq_tree = seqs_data['variable_tree'], seed=seed, metab_pert_params=met_pert, metab_ctrl_params=met_ctrl,
                                     otu_pert_params=otu_pert, otu_ctrl_params=otu_ctrl, disp_param=3600,
                                      meas_var=0.024, met_ctrl_pert = met_ctrl_pert, otu_ctrl_pert = otu_ctrl_pert)
                                      # meas_var=1)
                                      # meas_var=20537590)

                # for case in ['both']:
                    print(case)
                    if args.add_subjs:
                        met_syn_data, (otu_syn_cts, otu_syn_ra), y, pert_dict = add_subjs_to_semisyn(args.N_add,gd,case,seed)
                        case_path = f'{path}/{case}_{Num_Subjs+args.N_add}_{seed}/'
                    else:
                        met_syn_data, (otu_syn_cts, otu_syn_ra), y, pert_dict = gd.perturb_data(Num_Subjs, case=case)
                        case_path = f'{path}/{case}_{Num_Subjs}_{seed}/'
                    if not isinstance(met_pert, list):
                        met_pert = [met_pert]
                        met_ctrl = [met_ctrl]
                    # case_path = f'{path}/{case}_{Num_Subjs}_{seed}/'
                    # print('Case ' + case)
                    if not os.path.isdir(case_path):
                        os.mkdir(case_path)
                    with open(case_path + '/perturbed.pkl', 'wb') as f:
                        pkl.dump(pert_dict, f)
                    gd.subj_ixs_to_id.to_csv(case_path + '/subj_ix_to_id.csv', index=True)

                    y.to_csv(case_path + '/subj_labels.csv')

                    with open(case_path + '/perturbation_parameters.txt','w') as f:
                        f.write(f'Met Control: ({met_ctrl})\n')
                        f.write(f'Met Case: ({met_pert})\n')
                        f.write(f'Met Control Pert: ({met_ctrl_pert})\n')
                        f.write('\n')
                        f.write(f'Taxa Control: ({otu_ctrl})\n')
                        f.write(f'Taxa Case: ({otu_pert})\n')
                        f.write(f'Taxa Control Pert: ({otu_ctrl_pert})\n')

                    with open(case_path + '/perturbed.txt', 'w') as f:
                        for key in pert_dict.keys():
                            f.write(key + '\n')
                            for v in pert_dict[key]:
                                for vi in v:
                                    f.write(vi + '\n')
                                f.write('\n\n')
                            f.write('\n')

                    with open(case_path + '/subjs_perturbed_per_feat.txt','w') as f:
                        for i,gp in enumerate(gd.perturbed_per_feat):
                            f.write(f'Feature group {i}\n')
                            f.write(f'Perturbed: ')
                            for p in gp:
                                f.write(str(p) + ', ')
                            gp_ctrl = gd.ctrl_per_feat[i]
                            f.write(f'\nUn-perturbed: ')
                            for c in gp_ctrl:
                                f.write(str(c) + ', ')
                            f.write('\n\n')

                    with open(case_path + '/num_subjs.txt','w') as f:
                        f.write(f'{len(y)}\n')


                    print('')
                    print('{0} recurrers, {1} non-recurrers'.format(np.sum(y.values == 1), np.sum(y.values == 0)))
                    if len(pert_dict['metabs'])>0:
                        met_trans = transform_func(met_syn_data)
                        pert_metabs = pert_dict['metabs']
                        plot_data_hists(met_trans, pert_metabs , y, fig_path=case_path + 'metabs_')
                        plot_data_hists_agg(met_trans, pert_metabs , y, fig_path=case_path + 'metabs_')
                        (fig,ax),pval = dot_plots(met_syn_data, pert_metabs, y, 'metabs',
                                                  gd = gd)
                        fig.savefig(case_path+'metabs_dot_plot.pdf')
                        plt.close(fig)
                        data_ls = {'metabs': met_trans[sum(pert_metabs, [])]}
                        mets_data_new = copy.deepcopy(mets_data)
                        mets_data_new['X'] = met_syn_data
                        mets_data_new['y'] = y
                    else:
                        data_ls={}
                        mets_data_new={}

                    otu_rel_abun = otu_syn_ra
                    pert_otus = pert_dict['otus']
                    if len(pert_otus) > 0:
                        data_ls['otus']=otu_rel_abun[sum(pert_otus, [])]
                        plot_data_hists(otu_rel_abun, pert_otus , y, fig_path=case_path + 'otus_')
                        plot_data_hists_agg(otu_rel_abun, pert_otus, y, fig_path=case_path + 'otus_')

                        otu_sqrt_ = np.sqrt(otu_rel_abun)
                        otu_sqrt = transform_func(otu_sqrt_, log_transform=False)
                        plot_data_hists_new(otu_sqrt, pert_otus, y, fig_path=case_path + 'otus_sqrt_')
                        (fig, ax),pval = dot_plots(otu_rel_abun, pert_otus, y, 'otus',
                                                   gd = gd)
                        if not isinstance(ax, np.ndarray):
                            ax = [ax]
                        for axi in ax:
                            axi.set_yscale('log')
                        fig.savefig(case_path + 'otus_dot_plot.pdf')
                        plt.close(fig)
                        # plot_dist_hists(otu_rel_abun, pert_otus , gd.otu_dmat, fig_path=fig_path + 'otus')
                        # plot_heatmap(data_ls, y, fig_path=case_path)
                        seqs_data_new = copy.deepcopy(seqs_data)
                        seqs_data_new['X'] = otu_syn_cts
                        seqs_data_new['y'] = y
                    else:
                        # plot_heatmap(data_ls, y, fig_path=case_path)
                        seqs_data_new={}
                    if Num_Subjs<=300:

                        plot_heatmap(data_ls, y, fig_path=case_path + '/')



                    with open(f'{case_path}/mets.pkl','wb') as f:
                        pkl.dump(mets_data_new, f)

                    with open(f'{case_path}/seqs.pkl','wb') as f:
                        pkl.dump(seqs_data_new, f)