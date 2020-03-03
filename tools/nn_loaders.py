# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
from nn_modules import cls_fe_nnct, cls_basic_conv1ds, cls_fe_sinc, cls_embedder


def build_frontend_model(flag, device='cpu:0', exp_settings={}):

    if exp_settings['use_sinc']:
        print('--- Building Sinc Model ---')
        analysis = cls_fe_sinc.SincAnalysisSmooth(in_size=exp_settings['ft_size'],
                                                  out_size=exp_settings['ft_size_space'],
                                                  hop_size=exp_settings['hop_size'], exp_settings=exp_settings)
    elif exp_settings['use_rand_enc']:
        print('--- Building Simple Random Conv1D Encoder ---')
        analysis = cls_basic_conv1ds.ConvEncoder(in_size=exp_settings['ft_size'],
                                                 out_size=exp_settings['ft_size_space'],
                                                 hop_size=exp_settings['hop_size'], exp_settings=exp_settings)
    else:
        print('--- Building Cosine Model ---')
        analysis = cls_fe_nnct.AnalysiSmooth(in_size=exp_settings['ft_size'], out_size=exp_settings['ft_size_space'],
                                             hop_size=exp_settings['hop_size'], exp_settings=exp_settings)

    if exp_settings['use_simple_conv_dec']:
        print('--- Building Simple Random Conv1D Decoder  ---')
        synthesis = cls_basic_conv1ds.ConvDecoder(ft_size=exp_settings['ft_size_space'],
                                                  kernel_size=exp_settings['ft_syn_size'],
                                                  hop_size=exp_settings['hop_size'], exp_settings=exp_settings)
    else:
        print('--- Building cosine-based decoder ---')
        synthesis = cls_fe_nnct.Synthesis(ft_size=exp_settings['ft_size_space'],
                                          kernel_size=exp_settings['ft_syn_size'],
                                          hop_size=exp_settings['hop_size'], exp_settings=exp_settings)

    if flag == 'testing':
        print('--- Loading Model ---')
        analysis.load_state_dict(torch.load('results/analysis_' + exp_settings['exp_id'] + '.pytorch',
                                            map_location={'cuda:1': device}))
        synthesis.load_state_dict(torch.load('results/synthesis_' + exp_settings['exp_id'] + '.pytorch',
                                             map_location={'cuda:1': device}))

    tot_params = sum(p.numel() for p in analysis.parameters() if p.requires_grad) +\
        sum(p.numel() for p in synthesis.parameters() if p.requires_grad)

    print('Total Number of Parameters: %i' % tot_params)

    if torch.has_cuda:
        analysis = analysis.cuda()
        synthesis = synthesis.cuda()

    return analysis, synthesis


def build_mc_synthesis(flag, device='cuda:0', exp_settings={}, sep='save_id'):

    synthesis = cls_fe_nnct.Synthesis2C2S(ft_size=exp_settings['ft_size_space'],
                                          kernel_size=exp_settings['ft_syn_size'],
                                          hop_size=exp_settings['hop_size'], exp_settings=exp_settings)

    if flag == 'testing':
        print('--- Loading Model ---')
        synthesis.load_state_dict(torch.load('results/mc_synthesis_' + sep + exp_settings['exp_id'] + '_100_.pytorch',
                                             map_location={'cuda:1': device}))

    tot_params = sum(p.numel() for p in synthesis.parameters() if p.requires_grad)
    print('Total Number of Parameters: %i' % tot_params)

    if torch.has_cuda:
        synthesis = synthesis.cuda()

    return synthesis


def build_discriminator(flag, device='cpu:0', exp_settings={}):

    emd_function = cls_embedder.Embedder(exp_settings=exp_settings)

    if flag == 'testing':
        print('--- Loading Previous State ---')
        emd_function.load_state_dict(torch.load('results/disc_' + exp_settings['exp_id'] + '.pytorch',
                                     map_location={'cuda:1': device}))
    if torch.has_cuda:
        emd_function = emd_function.cuda()

    return emd_function


# EOF

