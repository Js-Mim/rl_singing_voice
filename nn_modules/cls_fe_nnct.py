# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def to_normalized_hz(mel, fs):
    return (700 * (10 ** (mel / 2595) - 1)) / fs


def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


class AnalysiSmooth(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, in_size=1024, out_size=1024, hop_size=384, exp_settings={}):
        super(AnalysiSmooth, self).__init__()

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
        self.p = 2

        self.relu = torch.nn.ReLU()

        # Model parameters to be optimized
        self.conv_a = torch.nn.Conv1d(in_channels=self.sz_out, out_channels=self.sz_out,
                                      kernel_size=5, dilation=10, padding=20, bias=False)
        if self.fully_modulated:
            self.norm_factor = nn.Parameter((torch.ones(self.sz_out, 1, self.sz_in)).cuda().float())
        else:
            self.norm_factor = nn.Parameter((torch.ones(1, 1, self.sz_in)).cuda().float())

        mel = (torch.linspace(to_mel(30), to_mel(22050), self.sz_out)).view(self.sz_out, 1, 1)
        self.freq_mat = nn.Parameter(to_normalized_hz(mel, exp_settings['fs']).cuda().float())
        self.freq_mat = nn.Parameter(self.freq_mat.cuda().float())
        self.phi = nn.Parameter(torch.randn(self.sz_out, 1, 1).cuda().float())

        # Initialize model parameters
        self.initialize()

    def initialize(self):
        torch.nn.init.kaiming_uniform_(self.conv_a.weight)
        t = np.arange(0, self.sz_in)
        for k in range(self.sz_out):
            self.f_matrix[k, :] = 2.*np.pi*t

        self.f_matrix = torch.autograd.Variable(torch.from_numpy(self.f_matrix[:, None, :]),
                                                requires_grad=True).cuda().float()

    def forward(self, wave_form, return_dictionary=False):
        # Synthesize dictionary
        f_sort_mat, sorted_indices = self.freq_mat.pow(self.p).sort(dim=0)
        if self.fully_modulated:
            cos_dict = torch.cos(self.f_matrix * f_sort_mat + self.phi[sorted_indices[:, 0, 0]]) *\
                       self.norm_factor[sorted_indices[:, 0, 0]]
        else:
            cos_dict = torch.cos(self.f_matrix * f_sort_mat + self.phi[sorted_indices[:, 0, 0]]) *\
                       self.norm_factor

        # Resize waveform
        batch_size = wave_form.size(0)
        time_domain_samples = wave_form.size(1)
        # Reshaping
        wave_form = wave_form.view(batch_size, 1, time_domain_samples)
        # Cosine part
        x_coeff = F.conv1d(wave_form, cos_dict, None, self.hop,
                           padding=self.pad, dilation=1, groups=1)
        x_c_coeff = self.relu(self.conv_a(x_coeff) + x_coeff)

        if return_dictionary:
            return x_c_coeff, cos_dict
        else:
            return x_c_coeff


class Synthesis(nn.Module):
    """
        Class for building the synthesis part
        of the Front-End ('Fe').
    """
    def __init__(self, ft_size=1024, kernel_size=1024, hop_size=384, exp_settings={}):
        super(Synthesis, self).__init__()

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
        self.p = 2

        # Initialize model parameters
        self.initialize()

        # Model parameters to be optimized
        if self.fully_modulated:
            self.norm_factor = nn.Parameter((torch.ones(self.sz_in, 1, self.kernel_sz) *
                                             1./(self.sz_in + self.kernel_sz)).cuda().float())
        else:
            self.norm_factor = nn.Parameter((torch.ones(1, 1, self.kernel_sz)
                                             * 1. / (self.sz_in + self.kernel_sz)).cuda().float())

        mel = (torch.linspace(to_mel(30), to_mel(22050), self.sz_in)).view(self.sz_in, 1, 1)
        self.freq_mat = nn.Parameter(to_normalized_hz(mel, exp_settings['fs']).cuda().float())
        self.phi = nn.Parameter(torch.randn(self.sz_in, 1, 1).cuda().float())

    def initialize(self):
        t = np.arange(0, self.kernel_sz)
        for k in range(self.sz_in):
            self.f_matrix[k, :] = 2.*np.pi*t

        self.f_matrix = torch.autograd.Variable(torch.from_numpy(self.f_matrix[:, None, :]),
                                                requires_grad=True).cuda().float()

    def forward(self, x_coeff, use_sorting=False):
        # Synthesize dictionary
        if use_sorting:
            f_sort_mat, sorted_indices = self.freq_mat.pow(self.p).sort(dim=0)
            if self.fully_modulated:
                norm_factor = self.norm_factor[sorted_indices[:, 0, 0]]
            else:
                norm_factor = self.norm_factor

            cos_dict = torch.cos(self.f_matrix * f_sort_mat + self.phi[sorted_indices[:, 0, 0]]) * norm_factor
            x_coeff = x_coeff[:, sorted_indices[:, 0, 0], :]

        else:
            cos_dict = torch.cos(self.f_matrix * self.freq_mat.pow(self.p) + self.phi) * self.norm_factor

        # Reshaping
        wave_form = F.conv_transpose1d(x_coeff, cos_dict, bias=None, stride=self.hop,
                                       padding=self.pad,
                                       dilation=1, groups=1)
        return wave_form[:, 0, :]

    def get_cos_dictionary(self, use_sorting=True):
        if use_sorting:
            f_sort_mat, sorted_indices = self.freq_mat.pow(self.p).sort(dim=0)
            if self.fully_modulated:
                norm_factor = self.norm_factor[sorted_indices[:, 0, 0]]
            else:
                norm_factor = self.norm_factor

            cos_dict = torch.cos(self.f_matrix * f_sort_mat + self.phi[sorted_indices[:, 0, 0]]) * norm_factor

        else:
            cos_dict = torch.cos(self.f_matrix * self.freq_mat.pow(self.p) + self.phi) * self.norm_factor

        return cos_dict

    def just_sort(self, x_coeff):
        _, sorted_indices = self.freq_mat.pow(self.p).sort(dim=0)
        x_coeff = x_coeff[:, sorted_indices[:, 0, 0], :]
        return x_coeff


class Synthesis2C2S(nn.Module):
    """
        Class for building the synthesis part
        of the Front-End ('Fe') for two channels and two sources.
    """
    def __init__(self, ft_size=1024, kernel_size=1024, hop_size=384, exp_settings={}):
        super(Synthesis2C2S, self).__init__()
        self.settings = exp_settings
        self.hop = hop_size
        self.ft_size = ft_size
        self.kernel_sz = kernel_size
        self.output_size = exp_settings['fs'] * exp_settings['d_p_length']
        self.input_size = np.ceil(self.output_size / self.hop)
        self.pad = np.int(((self.input_size-1)*self.hop + self.kernel_sz - self.output_size)/2)

        self.pretrained_synthesis = Synthesis(ft_size, kernel_size, hop_size, exp_settings)

        self.conv = torch.nn.ConvTranspose1d(in_channels=self.ft_size, out_channels=2,
                                             kernel_size=self.kernel_sz, bias=False, stride=hop_size,
                                             padding=self.pad,
                                             dilation=1, groups=1)

        self.init_weights()

    def init_weights(self):
        try:
            self.pretrained_synthesis.load_state_dict(
                torch.load('results/synthesis_' + self.settings['exp_id'] + '.pytorch',
                           map_location={'cuda:1': 'cuda:0'}))
            cos_dict = self.pretrained_synthesis.get_cos_dictionary(use_sorting=True)
            self.conv.weight.data = torch.cat((cos_dict, cos_dict), dim=1)
            del self.pretrained_synthesis, cos_dict
        except IOError:
            print('Initialization from previous state, failed!')
            pass

        return None

    def forward(self, x_coeff_mc):
        s1_out = self.conv(x_coeff_mc)
        return s1_out

# EOF
