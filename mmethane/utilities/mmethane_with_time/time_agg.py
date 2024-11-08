import torch
import torch.nn as nn
from scipy.special import logit
from torch.distributions.normal import Normal
import numpy as np


def time_priors_for_featMDITRE(num_time, device, time_prior=0.3):
    normal_window_a = Normal(torch.tensor(logit(time_prior), device=device), torch.tensor(np.sqrt(1e5), device=device))
    normal_center_b = Normal(torch.tensor(logit(0.5), device=device), torch.tensor(np.sqrt(1e5), device=device))
    return normal_window_a, normal_center_b

def time_inits_for_featMDITRE(num_time, num_rules, num_detectors, X_mask, X, y, detector_otuids):
    # Compute minimum time-window lengths for each possible
    # time center
    USE_SLOPE_DETECTORS = True
    window_len = None
    window_len_slope = None
    acc_center_abun = list()
    acc_len_abun = list()
    acc_num_samples_abun = list()
    acc_center_slope = list()
    acc_len_slope = list()
    acc_num_samples_slope = list()
    times = np.arange(num_time)
    for t in times:
        win_len = 1
        invalid_center = False
        window_len_slope = None
        while True:
            window_start = np.floor(t - (win_len / 2.)).astype('int')
            window_end = np.ceil(t + (win_len / 2.)).astype('int')
            if window_start >= 0 and window_end <= num_time:
                win_mask = X_mask[:, window_start:window_end].sum(-1)
                if np.all(win_mask >= 2):
                    window_len_slope = win_len
                    window_len = win_len
                    break
                elif np.all(win_mask >= 1):
                    window_len = win_len
                    break
                else:
                    win_len += 1
            else:
                invalid_center = True
                break
        if not invalid_center:
            acc_center_abun.append(t)
            acc_len_abun.append(window_len)
            acc_num_samples_abun.append(X_mask[:, window_start:window_end].sum())

            if window_len_slope is not None:
                acc_center_slope.append(t)
                acc_len_slope.append(window_len_slope)
                acc_num_samples_slope.append(X_mask[:, window_start:window_end].sum())

    if len(acc_center_slope) == 0:
        USE_SLOPE_DETECTORS = False
    num_subs = X.shape[0]
    thresh_init = np.zeros((num_rules, num_detectors), dtype=np.float32)
    slope_init = np.zeros((num_rules, num_detectors), dtype=np.float32)
    thresh_mean = list()
    slope_mean = list()
    mu_init = np.zeros((num_rules, num_detectors), dtype=np.float32)
    sigma_init = np.zeros((num_rules, num_detectors), dtype=np.float32)

    mu_slope_init = np.zeros((num_rules, num_detectors), dtype=np.float32)
    sigma_slope_init = np.zeros((num_rules, num_detectors), dtype=np.float32)

    for l in range(num_rules):
        for m in range(num_detectors):
            x_t = np.zeros((num_subs, len(acc_center_abun)))
            for n in range(len(acc_center_abun)):
                mu_abun = acc_center_abun[n]
                sigma_abun = acc_len_abun[n]
                window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                x = X[:, detector_otuids[l][m], window_start_abun:window_end_abun]
                x_mask = X_mask[:, window_start_abun:window_end_abun]
                X_out = x.sum(1).sum(-1) / x_mask.sum(-1)
                x_t[:, n] = X_out - X_out.mean()

            x_marg_0 = x_t[y == 0, :].mean(axis=0)
            x_marg_1 = x_t[y == 1, :].mean(axis=0)
            x_marg = np.absolute(x_marg_0 - x_marg_1)
            best_t_id_abun = np.argsort(x_marg)[::-1][0]
            mu_abun = acc_center_abun[best_t_id_abun]
            sigma_abun = acc_len_abun[best_t_id_abun]
            window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
            window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
            x = X[:, detector_otuids[l][m], window_start_abun:window_end_abun]
            x_mask = X_mask[:, window_start_abun:window_end_abun]
            X_out = x.sum(1).sum(-1) / x_mask.sum(-1)
            thresh_init[l, m] = X_out.mean()
            mu_init[l, m] = mu_abun
            sigma_init[l, m] = sigma_abun
            thresh_mean.append(x_marg[best_t_id_abun])


            if USE_SLOPE_DETECTORS:
                x_t = np.zeros((num_subs, len(acc_center_slope)))
                for n in range(len(acc_center_slope)):
                    mu_abun = acc_center_slope[n]
                    sigma_abun = acc_len_slope[n]
                    window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                    window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                    x = X[:, detector_otuids[l][m], window_start_abun:window_end_abun]
                    x_mask = X_mask[:, window_start_abun:window_end_abun]
                    tau = np.arange(window_start_abun, window_end_abun) - mu_abun
                    X_out = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(num_subs)])
                    x_t[:, n] = X_out - X_out.mean()

                x_marg_0 = x_t[y == 0, :].mean(axis=0)
                x_marg_1 = x_t[y == 1, :].mean(axis=0)
                x_marg = np.absolute(x_marg_0 - x_marg_1)
                best_t_id_abun = np.argsort(x_marg)[::-1][0]
                mu_abun = acc_center_slope[best_t_id_abun]
                sigma_abun = acc_len_slope[best_t_id_abun]
                window_start_abun = np.floor(mu_abun - (sigma_abun / 2.)).astype('int')
                window_end_abun = np.ceil(mu_abun + (sigma_abun / 2.)).astype('int')
                x = X[:, detector_otuids[l][m], window_start_abun:window_end_abun]
                x_mask = X_mask[:, window_start_abun:window_end_abun]
                tau = np.arange(window_start_abun, window_end_abun) - mu_abun
                X_out = np.array([np.polyfit(tau, x[s].sum(0), 1, w=x_mask[s])[0] for s in range(num_subs)])
                slope_init[l, m] = X_out.mean()
                mu_slope_init[l, m] = mu_abun
                sigma_slope_init[l, m] = sigma_abun
                slope_mean.append(x_marg[best_t_id_abun])

    abun_a_init = sigma_init / num_time
    abun_a_init = np.clip(abun_a_init, 1e-2, 1 - 1e-2)
    abun_b_init = (mu_init - (num_time * abun_a_init / 2.)) / ((1 - abun_a_init) * num_time)
    abun_b_init = np.clip(abun_b_init, 1e-2, 1 - 1e-2)
    init_dict = {'abun_a_init':abun_a_init, 'abun_b_init':abun_b_init,
                  'thresh_init': thresh_init}

    if USE_SLOPE_DETECTORS:
        slope_a_init = sigma_slope_init / num_time
        slope_a_init = np.clip(slope_a_init, 1e-2, 1 - 1e-2)
        slope_b_init = (mu_slope_init - (num_time * slope_a_init / 2.)) / ((1 - slope_a_init) * num_time)
        slope_b_init = np.clip(slope_b_init, 1e-2, 1 - 1e-2)

        init_dict.update({'slope_a_init':slope_a_init, 'slope_b_init': slope_b_init,
                     'slope_init':slope_init})
    return init_dict


def unitboxcar(x, mu, l, k):
    # parameterize boxcar function by the center and length
    dist = x - mu
    window_half = l / 2.
    y = torch.sigmoid((dist + window_half) * k) - torch.sigmoid((dist - window_half) * k)
    return y
class TimeAgg(nn.Module):
    """
    Aggregate time-series along the time dimension. Select a contiguous
    time window that's important for prediction task.
    We use the heavyside logistic function to calculate the
    importance weights of each time point for a detector and then normalize.
    """
    def __init__(self, num_time):
        super(TimeAgg, self).__init__()
        # Tensor of time points, starting from 0 to num_time - 1 (experiment duration)
        self.num_time = num_time
        self.register_buffer('times', torch.arange(num_time, dtype=torch.float32))

        # # # Time window bandwidth parameter
        # self.abun_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        # self.slope_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        # self.abun_b = nn.Parameter(torch.Tensor(num_rules, num_otus))
        # self.slope_b = nn.Parameter(torch.Tensor(num_rules, num_otus))

    def forward(self, x, mask=None, k=1.):
        # Compute unnormalized importance weights for each time point
        abun_a = torch.sigmoid(self.abun_a).unsqueeze(-1)
        abun_b = torch.sigmoid(self.abun_b).unsqueeze(-1)
        sigma = self.num_time * abun_a
        mu = (self.num_time * abun_a / 2.) + (1 - abun_a) * self.num_time * abun_b
        time_wts_unnorm = unitboxcar(self.times,
            mu,
            sigma, k)
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))
        self.wts = time_wts_unnorm
        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(time_wts_unnorm.sum(dim=-1, keepdims=True) + 1e-8)

        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(-1))
            raise ValueError('Nan in time aggregation!')

        # Aggregation over time dimension
        # Essentially a convolution over time
        x_abun = x.mul(time_wts).sum(dim=-1)
        self.m = mu
        self.s_abun = sigma

        if self.USE_SLOPE:
            slope_a = torch.sigmoid(self.slope_a).unsqueeze(-1)
            slope_b = torch.sigmoid(self.slope_b).unsqueeze(-1)
            sigma_slope = self.num_time * slope_a
            mu_slope = (self.num_time * slope_a / 2.) + (1 -  slope_a) * self.num_time * slope_b

            time_wts_unnorm_slope = unitboxcar(self.times,
                mu_slope,
                sigma_slope, k)

            # Mask out time points with no samples
            if mask is not None:
                time_wts_unnorm_slope = time_wts_unnorm_slope.mul(
                    mask.unsqueeze(1).unsqueeze(1))

            self.wts_slope = time_wts_unnorm_slope

            # Compute approx. avg. slope over time window
            tau = self.times - mu_slope
            a = (time_wts_unnorm_slope * x).sum(dim=-1)
            b = (time_wts_unnorm_slope * tau).sum(dim=-1)
            c = (time_wts_unnorm_slope).sum(dim=-1)
            d = (time_wts_unnorm_slope * x * tau).sum(dim=-1)
            e = (time_wts_unnorm_slope * (tau ** 2)).sum(dim=-1)
            num = ((a*b) - (c*d))
            den = ((b**2) - (e*c)) + 1e-8
            x_slope = num / den

            if torch.isnan(x_slope).any():
                print(time_wts_unnorm_slope.sum(dim=-1))
                print(x_slope)
                raise ValueError('Nan in time aggregation!')


            self.m_slope = mu_slope
            self.s_slope = sigma_slope
        else:
            x_slope = None

        return x_abun, x_slope

    def init_params(self, init_args, device='cuda'):
        # # Initialize mu and sigma parameter
        self.abun_a = nn.Parameter(torch.tensor(logit(init_args['abun_a_init']), device=device, dtype=torch.float))
        self.abun_b = nn.Parameter(torch.tensor(logit(init_args['abun_b_init']), device=device, dtype=torch.float))
        if 'slope_a_init' in init_args.keys():
            self.slope_a = nn.Parameter(torch.tensor(logit(init_args['slope_a_init']), device=device, dtype=torch.float))
            self.slope_b = nn.Parameter(torch.tensor(logit(init_args['slope_b_init']), device=device, dtype=torch.float))
            self.USE_SLOPE=True
        else:
            self.USE_SLOPE=False

        return