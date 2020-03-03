# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from nn_modules.losses import _eps


class SincAnalysisSmooth(nn.Module):
    """
        Re-implementation of SincNet.
    """
    def __init__(self, in_size=1024, out_size=1024, hop_size=384, exp_settings={}):
        super(SincAnalysisSmooth, self).__init__()

        # Analysis Parameters
        self.sort = exp_settings['dict_sorting']
        self.fs = exp_settings['fs']
        self.batch_size = None
        self.time_domain_samples = None
        self.sz_in = in_size
        self.sz_out = out_size
        self.hop = hop_size
        self.f_matrix = np.zeros((self.sz_out, self.sz_in), dtype=np.float32)
        self.input_size = exp_settings['fs'] * exp_settings['d_p_length']
        self.output_size = np.ceil(self.input_size/self.hop)
        self.pad = np.int(-self.input_size/2 + self.sz_in/2 - hop_size/2 + self.output_size*hop_size/2)
        self.p = 1
        self.relu = torch.nn.ReLU()

        # Model parameters to be optimized
        self.conv_a = torch.nn.Conv1d(in_channels=self.sz_out, out_channels=self.sz_out,
                                      kernel_size=5, dilation=10, padding=20, bias=False)

        self.norm_factor = nn.Parameter((torch.ones(1, 1, self.sz_in)).cuda().float())

        # Frequencies
        mel = (torch.linspace(self.to_mel(80), self.to_mel(22050), self.sz_out)).view(self.sz_out, 1, 1)
        self.freq_mat_f1 = nn.Parameter(self.to_normalized_hz(mel).cuda().float())
        self.freq_mat_f2 = self.to_normalized_hz(mel + torch.randn(size=(self.sz_out, 1, 1)))
        self.freq_mat_f2 = nn.Parameter(self.freq_mat_f2.cuda().float())

        # Initialize model parameters
        self.initialize()

    def initialize(self):
        torch.nn.init.kaiming_uniform_(self.conv_a.weight)
        t = np.arange(0, self.sz_in//2 + 1)
        t = np.hstack((-np.flip(t)[:-1], 0.1, t[1:-1]))
        for k in range(self.sz_out):
            self.f_matrix[k, :] = 2.*np.pi*t

        self.f_matrix = torch.autograd.Variable(torch.from_numpy(self.f_matrix[:, None, :]),
                                                requires_grad=True).cuda().float()

    def forward(self, wave_form, return_dictionary=False):
        # Synthesize dictionary
        if self.sort:
            f1_out, s_indices = self.freq_mat_f1.pow(self.p).sort(dim=0)
            f2_out = self.freq_mat_f2.pow(self.p)[s_indices[0, 0, :]]
        else:
            f1_out = self.freq_mat_f1.pow(self.p)
            f2_out = self.freq_mat_f2.pow(self.p)

        dict_f1 = 2. * f1_out * (torch.sin(self.f_matrix * f1_out)/(self.f_matrix * f1_out + _eps))
        dict_f2 = 2. * f2_out * (torch.sin(self.f_matrix * f2_out)/(self.f_matrix * f2_out + _eps))

        sinc_dict = (dict_f1 - dict_f2) * self.norm_factor

        # Resize waveform
        batch_size = wave_form.size(0)
        time_domain_samples = wave_form.size(1)
        # Reshaping
        wave_form = wave_form.view(batch_size, 1, time_domain_samples)

        # Cosine part
        x_coeff = F.conv1d(wave_form, sinc_dict, None, self.hop,
                           padding=self.pad, dilation=1, groups=1)
        x_c_coeff = self.relu(self.conv_a(x_coeff) + x_coeff)

        if return_dictionary:
            return x_c_coeff, sinc_dict
        else:
            return x_c_coeff

    def to_normalized_hz(self, mel):
        return (700 * (10 ** (mel / 2595) - 1))/self.fs

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

# EOF
