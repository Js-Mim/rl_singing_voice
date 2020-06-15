# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'
"""
    Based on: https://raw.githubusercontent.com/rythei/PyTorchOT/master/
"""
import torch
from torch.autograd import Variable


def sinkhorn(dist_mat, reg, num_max_iter=1e3, stop_thr=1e-5, cuda=True):

    if cuda:
        a = Variable(torch.ones((dist_mat.size()[0],)) / dist_mat.size()[0]).cuda()
        b = Variable(torch.ones((dist_mat.size()[1],)) / dist_mat.size()[1]).cuda()
    else:
        a = Variable(torch.ones((dist_mat.size()[0],)) / dist_mat.size()[0])
        b = Variable(torch.ones((dist_mat.size()[1],)) / dist_mat.size()[1])

    # init data
    n_ini = len(a)
    n_fin = len(b)

    if cuda:
        u = Variable(torch.ones(n_ini) / n_ini).cuda()
        v = Variable(torch.ones(n_fin) / n_fin).cuda()
    else:
        u = Variable(torch.ones(n_ini) / n_ini)
        v = Variable(torch.ones(n_fin) / n_fin)

    k = torch.exp(- dist_mat / reg)
    k_p = (1 / a).view(-1, 1) * k

    cpt = 0
    err = 1
    while err > stop_thr and cpt < num_max_iter:
        k_transpose_u = k.t().matmul(u)
        v = torch.div(b, k_transpose_u)
        u = 1. / k_p.matmul(v)

        if cpt % 10 == 0:
            transp = u.view(-1, 1) * (k * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()
        cpt += 1

    return torch.sum(u.view((-1, 1)) * k * v.view((1, -1)) * dist_mat)

# EOF
