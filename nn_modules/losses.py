# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch

bce_logitloss_func = torch.nn.BCEWithLogitsLoss()
bce_func = torch.nn.BCELoss()
_eps = 1e-24


def neg_snr(x, xhat):
    res_val = -10. * torch.log10(torch.norm(x, 2.).pow(2.) / (torch.norm(x - xhat, 2).pow(2.) + _eps) + _eps)
    return res_val


def si_sdr(x, xhat, scaling=False):
    if scaling:
        a = (x*xhat).sum()/(torch.norm(x, p=2).pow(2.) + _eps)
        res_val = 10. * torch.log10(torch.norm(a*x, 2.).pow(2.) / (torch.norm(a*x - xhat, 2).pow(2.) + _eps) + _eps)
    else:
        res_val = 10. * torch.log10(torch.norm(x, 2.).pow(2.) / (torch.norm(x - xhat, 2).pow(2.) + _eps) + _eps)
    return res_val


def mse(x, xhat):
    res_val = torch.mean((x - xhat).pow(2.))
    return res_val


def just_l1(x, xhat):
    res_val = torch.norm(x - xhat, 1)/(x.size(1) * x.size(2))
    return res_val


def tot_variation_2d(x):
    feat_var = torch.norm(x[:, :, :-1] - x[:, :, 1:], 2.)/x.size(2)
    time_var = torch.norm(x[:, :-1, :] - x[:, 1:, :], 2.)/x.size(1)
    tot_var = feat_var + time_var
    return tot_var


def bce(x_pos, x_neg):
    return 0.5 * (-(x_pos + 1e-24).log().mean() - (1. - x_neg + 1e-24).log().mean())


# EOF
