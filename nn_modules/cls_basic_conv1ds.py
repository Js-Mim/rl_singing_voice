# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe'), with randomly
        initialized dictionaries.
    """
    def __init__(self, in_size=1024, out_size=1024, hop_size=384, exp_settings={}):
        super(ConvEncoder, self).__init__()

        # Analysis Parameters
        self.fully_modulated = exp_settings['fully_modulated']
        self.batch_size = None
        self.time_domain_samples = None
        self.sz_in = in_size
        self.sz_out = out_size
        self.hop = hop_size
        self.f_matrix = np.zeros((self.sz_out, self.sz_in), dtype=np.float32)
        self.input_size = exp_settings['fs'] * exp_settings['d_p_length']
        self.output_size = np.ceil(self.input_size/self.hop)
        self.pad = np.int(-self.input_size/2 + self.sz_in/2 - hop_size/2 + self.output_size*hop_size/2)

        self.relu = torch.nn.ReLU()

        # Model parameters to be optimized
        self.conv_a1 = torch.nn.Conv1d(in_channels=1, out_channels=self.sz_out,
                                       kernel_size=self.sz_in, stride=self.hop, padding=self.pad, bias=False)
        self.conv_a2 = torch.nn.Conv1d(in_channels=self.sz_out, out_channels=self.sz_out,
                                       kernel_size=5, dilation=10, padding=20, bias=False)

        # Initialize model parameters
        self.initialize()

    def initialize(self):
        torch.nn.init.kaiming_uniform_(self.conv_a1.weight)
        torch.nn.init.kaiming_uniform_(self.conv_a2.weight)

    def forward(self, wave_form):

        # Resize waveform
        batch_size = wave_form.size(0)
        time_domain_samples = wave_form.size(1)
        # Reshaping
        wave_form = wave_form.view(batch_size, 1, time_domain_samples)
        # Cosine part
        x_coeff = self.conv_a1.forward(wave_form)
        x_c_coeff = self.relu(self.conv_a2(x_coeff) + x_coeff)

        return x_c_coeff


class ConvDecoder(nn.Module):
    """
        Class for building the synthesis part
        of the Front-End ('Fe'), with randomly
        initialized dictionaries.
    """
    def __init__(self, ft_size=1024, kernel_size=1024, hop_size=384, exp_settings={}):
        super(ConvDecoder, self).__init__()

        # Synthesis Parameters
        self.fully_modulated = exp_settings['fully_modulated']
        self.batch_size = None
        self.time_domain_samples = None
        self.sz_in = ft_size
        self.kernel_sz = kernel_size
        self.hop = hop_size
        self.output_size = exp_settings['fs'] * exp_settings['d_p_length']
        self.input_size = np.ceil(self.output_size / self.hop)
        self.pad = np.int(((self.input_size-1)*self.hop + self.kernel_sz - self.output_size)/2)
        self.f_matrix = np.zeros((self.sz_in, self.kernel_sz), dtype=np.float32)
        self.tanh = torch.nn.Tanh()

        self.conv_dec = torch.nn.ConvTranspose1d(in_channels=self.sz_in, out_channels=1,
                                                 kernel_size=self.kernel_sz, bias=None, stride=self.hop,
                                                 padding=self.pad,
                                                 dilation=1, groups=1)

    def forward(self, x_coeff, use_sorting):
        # Reshaping
        wave_form = self.tanh(self.conv_dec.forward(x_coeff))

        return wave_form[:, 0, :]

# EOF
