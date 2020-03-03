# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
        Class for building the embedding module.
        The discriminator is also plugged here.
    """
    def __init__(self, exp_settings={}):
        super(Embedder, self).__init__()

        # Some useful parameters
        self.feat_dim = exp_settings['ft_size_space']
        self.embd_dim = exp_settings['embedding_space_dim']
        self.lat_dim = exp_settings['disc_lat_dim']
        self.only_embd = exp_settings['embedings_only']

        # The building block for the embedding space
        self.relu = torch.nn.ReLU()
        self.embd_a = torch.nn.Conv1d(in_channels=self.feat_dim, out_channels=self.feat_dim*2,
                                      kernel_size=3, stride=1,
                                      padding=1, dilation=1, groups=1, bias=False)
        self.gl_norm = torch.nn.GroupNorm(num_groups=1, num_channels=self.feat_dim)
        self.glu = torch.nn.GLU(dim=1)

        if not self.only_embd:
            # The building blocks of the discriminator
            self.f_reduce = nn.Linear(in_features=self.embd_dim*2, out_features=self.embd_dim, bias=True)
            self.f_a_reduce = nn.Linear(in_features=self.embd_dim, out_features=self.lat_dim, bias=True)
            self.f_cls = nn.Linear(in_features=self.lat_dim, out_features=1, bias=False)

        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.embd_a.weight)
        if not self.only_embd:
            torch.nn.init.xavier_uniform_(self.f_reduce.weight)
            torch.nn.init.xavier_uniform_(self.f_a_reduce.weight)
            torch.nn.init.xavier_uniform_(self.f_cls.weight)
        return None

    def embed(self, x):
        z_embed = self.relu(self.glu(self.embd_a(x)) + x)
        return z_embed

    def forward(self, x_a, x_b):
        bs, ch, tf = x_a.size()
        # Embed the two representations
        z_a = self.gl_norm.forward(self.embed(x_a)).view(bs, tf, ch)
        z_b = self.gl_norm.forward(self.embed(x_b)).view(bs, tf, ch)
        # Concatenation & global average pooling
        out = torch.cat((z_a, z_b), dim=2).mean(dim=1)
        out = self.relu(self.f_reduce(out))
        out = self.relu(self.f_a_reduce(out))
        out = self.f_cls(out)
        return out

# EOF
