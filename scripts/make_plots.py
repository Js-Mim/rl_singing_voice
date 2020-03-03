# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tools import nn_loaders
from settings.rl_experiment_settings import exp_settings
_eps = 1e-24
fs = 44100


def _get_freqs_np(torch_dict, p=1):
    frqs = np.sort(torch_dict['freq_mat'][:, 0, 0].pow(p).cpu().numpy() * fs)
    return frqs


def _load_ansyn_dicts(exp_id):
    analysis_dict = torch.load('results/analysis_' + exp_id + '.pytorch')
    synthesis_dict = torch.load('results/synthesis_' + exp_id + '.pytorch')
    return analysis_dict, synthesis_dict


def plot_frq_differences():
    sns.set_context("paper", font_scale=2.6)
    exp_id_list = ['r-mcos_ln', 'r-mcos-8']
    c_list = ['cornflowerblue', 'sandybrown']

    for index, experiment in enumerate(exp_id_list):
        _, synthesis_dict = _load_ansyn_dicts(experiment)

        # Check on decoder model id
        if 'mcos' in experiment.split("-")[1]:
            model = 'Modulated cos.: '
        else:
            model = 'nnCS: '

        # Check on the function used over the frequency vector
        if 'ln' in experiment:
            p = 1
            label = model + ' Linear'
            l_style = 'solid'
        else:
            p = 2
            label = model + ' Squared'
            l_style = 'dashed'
        freqs = _get_freqs_np(synthesis_dict, p=p)
        plt.semilogy(freqs, label=label, linewidth=2.7, markersize=2, c=c_list[index], linestyle=l_style)

    plt.xlabel('Component Index ' + r'$(c)$')
    plt.ylabel('(Carrier) Frequency (Hz)')
    plt.ylim(1, fs//2 + 500)
    plt.xlim(0, 800)
    plt.grid(b=True, which='both')
    plt.legend()
    plt.show()

    return None


def boxplot_models_sdr():
    sns.set_context("paper", font_scale=2.6)
    # Linear case
    cos_ln_sdr = np.load('results/eval_res/r-cos_ln_sdr.npy')
    cos_ln_sdr.shape = (cos_ln_sdr.shape[0], 1)
    mcos_ln_sdr = np.load('results/eval_res/r-mcos_ln_sdr.npy')
    mcos_ln_sdr.shape = (mcos_ln_sdr.shape[0], 1)

    # Squared case
    cos_sdr = np.load('results/eval_res/r-cos_sdr.npy')
    cos_sdr.shape = (cos_sdr.shape[0], 1)
    mcos_sdr = np.load('results/eval_res/r-mcos_sdr.npy')
    mcos_sdr.shape = (mcos_sdr.shape[0], 1)

    # Alternatives case
    # Random case
    rconv_conv = np.load('results/eval_res/r-conv-conv_sdr.npy')
    rconv_conv.shape = (rconv_conv.shape[0], 1)
    rsinc_mcos = np.load('results/eval_res/r-sinc-mcos_sdr.npy')
    rsinc_mcos.shape = (rsinc_mcos.shape[0], 1)

    si_sdr_res = np.hstack((cos_ln_sdr, cos_sdr, mcos_ln_sdr, mcos_sdr, rsinc_mcos, rconv_conv))

    plt.violinplot(si_sdr_res, showmeans=False, showextrema=False, showmedians=True)
    plt.xticks([1, 2, 3, 4, 5, 6], ['nnCS: Linear', 'nnCS: Squared',
                                    'nnMCS: Linear', 'nnMCS: Squared',
                                    'nnMCS: Sinc-Encoder', 'Conv1D'], fontsize=13)
    plt.ylabel('SI-SDR (dB)')
    plt.ylim(4, 37.3)

    med_vals = np.round(np.median(si_sdr_res, axis=0), 1)
    plt.annotate(str(med_vals[0]), xy=(1, med_vals[0] + 1), xytext=(0.95, med_vals[0] + 0.3))
    plt.annotate(str(med_vals[1]), xy=(2, med_vals[1] + 1), xytext=(1.95, med_vals[1] + 0.3))
    plt.annotate(str(med_vals[2]), xy=(3, med_vals[2] + 1), xytext=(2.95, med_vals[2] + 0.3))
    plt.annotate(str(med_vals[3]), xy=(4, med_vals[3] + 1), xytext=(3.95, med_vals[3] + 0.3))
    plt.annotate(str(med_vals[4]), xy=(5, med_vals[4] + 1), xytext=(4.95, med_vals[4] + 0.3))
    plt.annotate(str(med_vals[5]), xy=(6, med_vals[5] + 1), xytext=(5.95, med_vals[5] + 0.3))

    plt.show()


def boxplot_embedding_models():
    sns.set_context("paper", font_scale=2.6)

    # Embedder with Discriminator
    # SDR
    mcos_disc_sdr = np.load('results/eval_res/r-mcos-disc_sdr.npy')
    mcos_disc_sdr.shape = (mcos_disc_sdr.shape[0], 1)

    # Masking SDR
    mcos_disc_mask_sdr = np.load('results/eval_res/r-mcos-disc_mask_sdr.npy')
    mcos_disc_mask_sdr.shape = (mcos_disc_mask_sdr.shape[0], 1)
    # Embedding SDR
    mcos_disc_lat_sdr = np.load('results/eval_res/r-mcos-disc_lat_sdr.npy')
    mcos_disc_lat_sdr.shape = (mcos_disc_lat_sdr.shape[0], 1)
    # ADT
    mcos_disc_adt = np.load('results/eval_res/r-mcos-disc_adt.npy')
    mcos_disc_adt.shape = (mcos_disc_adt.shape[0], 1)
    # EMBD-ADT
    mcos_disc_embd_adt = np.load('results/eval_res/r-mcos-disc_lat_adt.npy')
    mcos_disc_embd_adt.shape = (mcos_disc_embd_adt.shape[0], 1)

    # Embedder with Additivity loss
    # SDR
    mcos_embd_sdr = np.load('results/eval_res/r-mcos-embd_sdr.npy')
    mcos_embd_sdr.shape = (mcos_embd_sdr.shape[0], 1)
    # Masking SDR
    mcos_embd_mask_sdr = np.load('results/eval_res/r-mcos-embd_mask_sdr.npy')
    mcos_embd_mask_sdr.shape = (mcos_embd_mask_sdr.shape[0], 1)
    # Embedding SDR
    mcos_embd_lat_sdr = np.load('results/eval_res/r-mcos-embd_lat_sdr.npy')
    mcos_embd_lat_sdr.shape = (mcos_embd_lat_sdr.shape[0], 1)
    # ADT
    mcos_embd_adt = np.load('results/eval_res/r-mcos-embd_adt.npy')
    mcos_embd_adt.shape = (mcos_embd_adt.shape[0], 1)
    # EMBD-ADT
    mcos_embd_lat_adt = np.load('results/eval_res/r-mcos-embd_lat_adt.npy')
    mcos_embd_lat_adt.shape = (mcos_embd_lat_adt.shape[0], 1)

    # Embedder that makes singing voice louder
    # SDR
    mcos_embd_loud_sdr = np.load('results/eval_res/r-mcos-embd-loud_sdr.npy')
    mcos_embd_loud_sdr.shape = (mcos_embd_loud_sdr.shape[0], 1)
    # Masking SDR
    mcos_embd_loud_mask_sdr = np.load('results/eval_res/r-mcos-embd-loud_mask_sdr.npy')
    mcos_embd_loud_mask_sdr.shape = (mcos_embd_mask_sdr.shape[0], 1)
    # Embedding SDR
    mcos_embd_loud_lat_sdr = np.load('results/eval_res/r-mcos-embd-loud_lat_sdr.npy')
    mcos_embd_loud_lat_sdr.shape = (mcos_embd_loud_lat_sdr.shape[0], 1)
    # ADT
    mcos_embd_loud_adt = np.load('results/eval_res/r-mcos-embd-loud_adt.npy')
    mcos_embd_loud_adt.shape = (mcos_embd_loud_adt.shape[0], 1)
    # EMBD-ADT
    mcos_embd_loud_lat_adt = np.load('results/eval_res/r-mcos-embd-loud_lat_adt.npy')
    mcos_embd_loud_lat_adt.shape = (mcos_embd_loud_lat_adt.shape[0], 1)

    # SDR plot
    si_sdr_res = np.hstack((mcos_disc_sdr, mcos_disc_mask_sdr, mcos_disc_lat_sdr,
                            mcos_embd_sdr, mcos_embd_mask_sdr, mcos_embd_lat_sdr,
                            mcos_embd_loud_sdr, mcos_embd_loud_mask_sdr, mcos_embd_loud_lat_sdr))

    plt.violinplot(si_sdr_res, showmeans=False, showextrema=False, showmedians=True)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ['nnMCS-Disc.', 'nnMCS-Disc:Mask', 'nnMCS-Disc:Lat',
                                             'nnMCS-Embd.', 'nnMCS-Embd:Mask', 'nnMCS-Embd:Lat',
                                             'nnMCS-Loud.', 'nnMCS-Loud:Mask', 'nnMCS-Loud:Lat'], fontsize=9)
    plt.ylabel('SI-SDR (dB)')
    plt.ylim(4, 37.3)

    med_vals = np.round(np.median(si_sdr_res, axis=0), 1)
    plt.annotate(str(med_vals[0]), xy=(1, med_vals[0] + 1), xytext=(0.95, med_vals[0] + 0.3))
    plt.annotate(str(med_vals[1]), xy=(2, med_vals[1] + 1), xytext=(1.95, med_vals[1] + 0.3))
    plt.annotate(str(med_vals[2]), xy=(3, med_vals[2] + 1), xytext=(2.95, med_vals[2] + 0.3))
    plt.annotate(str(med_vals[3]), xy=(4, med_vals[3] + 1), xytext=(3.95, med_vals[3] + 0.3))
    plt.annotate(str(med_vals[4]), xy=(5, med_vals[4] + 1), xytext=(4.95, med_vals[4] + 0.3))
    plt.annotate(str(med_vals[5]), xy=(6, med_vals[5] + 1), xytext=(5.95, med_vals[5] + 0.3))
    plt.annotate(str(med_vals[6]), xy=(7, med_vals[6] + 1), xytext=(6.95, med_vals[6] + 0.3))
    plt.annotate(str(med_vals[7]), xy=(8, med_vals[7] + 1), xytext=(7.95, med_vals[7] + 0.3))
    plt.annotate(str(med_vals[8]), xy=(9, med_vals[8] + 1), xytext=(8.95, med_vals[8] + 0.3))

    # ADT plot
    plt.figure()
    adt_res = np.hstack((mcos_disc_adt, mcos_disc_embd_adt,
                         mcos_embd_adt, mcos_embd_lat_adt,
                         mcos_embd_loud_adt, mcos_embd_loud_lat_adt))

    plt.violinplot(adt_res, showmeans=False, showextrema=False, showmedians=True)
    plt.xticks([1, 2, 3, 4, 5, 6], ['nnMCS-Disc.', 'nnMCS-Disc:Lat',
                                    'nnMCS-Embd.', 'nnMCS-Embd:Lat',
                                    'nnMCS-Loud.', 'nnMCS-Loud:Lat'], fontsize=9)
    plt.ylabel(r'$L_1$ Error')
    plt.ylim(0, 0.5)

    med_vals = np.round(np.median(adt_res, axis=0), 2)
    plt.annotate(str(med_vals[0]), xy=(1, med_vals[0] + 0.001), xytext=(0.95, med_vals[0] + 0.005))
    plt.annotate(str(med_vals[1]), xy=(2, med_vals[1] + 0.001), xytext=(1.95, med_vals[1] + 0.005))
    plt.annotate(str(med_vals[2]), xy=(3, med_vals[2] + 0.001), xytext=(2.95, med_vals[2] + 0.005))
    plt.annotate(str(med_vals[3]), xy=(4, med_vals[3] + 0.001), xytext=(3.95, med_vals[3] + 0.005))
    plt.annotate(str(med_vals[4]), xy=(5, med_vals[4] + 0.001), xytext=(4.95, med_vals[4] + 0.005))
    plt.annotate(str(med_vals[5]), xy=(6, med_vals[5] + 0.001), xytext=(5.95, med_vals[5] + 0.005))
    plt.show()


def plot_mcos_dictionary():
    sns.set_context("paper", font_scale=2.6)
    _, synthesis = nn_loaders.build_frontend_model(flag='testing',
                                                   device='cuda:0', exp_settings=exp_settings)

    if exp_settings['dict_sorting']:
        frqs, sorted_indices = synthesis.freq_mat.pow(2).sort(dim=0)
        mods = synthesis.norm_factor[sorted_indices[:, 0, 0]]
        phi = synthesis.phi[sorted_indices[:, 0, 0]]
    else:
        frqs, sorted_indices = synthesis.freq_mat.pow(2)
        mods = synthesis.norm_factor
        phi = synthesis.phi

    cosine_functions = torch.cos(synthesis.f_matrix * frqs + phi)
    mod_cosine_functions = cosine_functions * mods
    fft_size = cosine_functions.size(2)
    x_frq_axis = (np.arange(0, fft_size//2 + 1)/fft_size) * fs
    n_f = 1./np.sqrt(fft_size)
    num_of_components = cosine_functions.size(0)
    for index in range(0, num_of_components, 5):
        print(frqs[index, 0, 0].detach().item() * fs)
        cosine_function = cosine_functions.detach().cpu().numpy()[index, 0, :]
        mod_function = mods.detach().cpu().numpy()[index, 0, :]
        resulting_signal = mod_cosine_functions.detach().cpu().numpy()[index, 0, :]

        # Compute the magnitude using DFT
        cos_mag = np.abs(n_f * np.fft.rfft(cosine_function))
        mod_mag = np.abs(np.fft.rfft(np.blackman(fft_size) * mod_function))
        res_basis_mag = np.abs(np.fft.rfft(resulting_signal))
        # Max-norm for clearer plots
        mod_mag *= np.max(cos_mag)/np.max(mod_mag)
        res_basis_mag *= np.max(cos_mag)/np.max(res_basis_mag)

        # Normalize for convenience (notice the modulation signals are already normalized)
        plt.figure()
        plt.plot(x_frq_axis, 10. * np.log10(cos_mag + _eps), label='Carrier Signal',
                 linestyle='solid', linewidth=2., markersize=1.5, c='cornflowerblue')
        plt.plot(x_frq_axis,  10. * np.log10(mod_mag + _eps), label='Modulating Signal',
                 linestyle='solid', linewidth=2., markersize=1.5, c='sandybrown')
        plt.legend()
        plt.grid(b=True, which='both')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Magnitude (dB)')
        plt.ylim(-60, +3)
        plt.xlim(0, 22000)

        plt.figure()
        plt.plot(x_frq_axis, 10. * np.log10(res_basis_mag + _eps), label='Resulting Signal',
                 linestyle='solid', linewidth=2., markersize=1.5, c='cornflowerblue')

        plt.legend()
        plt.grid(b=True, which='both')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Magnitude (dB)')
        plt.ylim(-60, +3)
        plt.xlim(0, 22000)

        plt.show()

    return None


def main():
    #plot_frq_differences()
    #boxplot_models_sdr()
    #boxplot_embedding_models()
    plot_mcos_dictionary()


if __name__ == "__main__":
    main()


# EOF
