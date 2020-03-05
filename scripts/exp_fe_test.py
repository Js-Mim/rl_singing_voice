# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from nn_modules import losses
from tools import helpers, io_methods, nn_loaders
from settings.rl_experiment_settings import exp_settings as settings
from settings.rl_disc_experiment_settings import exp_settings as settings_disc
_eps = 1e-24


def perform_frontend_testing(make_plots=False, write_files=False,
                             compute_embeddings=False, stft_comparison=False, exp_settings={}):
    print('ID: ' + exp_settings['exp_id'])
    # Make sure no overlapping batches take place
    exp_settings.update({'batch_overlap': 0})
    exp_settings.update({'batch_size': 1})
    # Instantiating data handler
    io_dealer = helpers.DataIO(exp_settings=exp_settings)
    exp_test_set = 1
    exp_test_multitracks = 50

    # Number of file sets
    num_of_sets = exp_test_multitracks//exp_test_set

    # Build NN modules
    analysis, synthesis = nn_loaders.build_frontend_model(flag='testing', exp_settings=exp_settings)
    if compute_embeddings:
        disc_module = nn_loaders.build_discriminator(flag='testing', exp_settings=exp_settings)
        additivity_loss_embd = []
        loss_from_embd = []

    # Initializing two lists for storing loss values
    loss = []
    loss_from_masking = []
    additivity_loss = []
    loss_from_masking_stft = []
    additivity_loss_stft = []
    for file_set in tqdm(range(1, num_of_sets + 1)):
        # Load a sub-set of the recordings
        mix, vox, bkg = io_dealer.get_data(file_set, exp_test_set,
                                           monaural=exp_settings['monaural'], dev=False)

        # Create batches
        vox = io_dealer.gimme_batches(vox)
        mix = io_dealer.gimme_batches(mix)
        bkg = io_dealer.gimme_batches(bkg)

        # Compute the total number of batches contained in this sub-set
        num_batches = vox.shape[0] // exp_settings['batch_size']
        voxhat = np.zeros(vox.shape)

        for batch in range(num_batches):
            p_start = batch * exp_settings['batch_size']
            p_end = (batch + 1) * exp_settings['batch_size']
            vox_tr_batch = io_dealer.batches_from_numpy(vox[p_start:p_end, :])
            mix_tr_batch = io_dealer.batches_from_numpy(mix[p_start:p_end, :])
            bkg_tr_batch = io_dealer.batches_from_numpy(bkg[p_start:p_end, :])

            # Analysis
            mix_coeff = analysis.forward(mix_tr_batch)
            vox_coeff = analysis.forward(vox_tr_batch)
            bkg_coeff = analysis.forward(bkg_tr_batch)

            if 'disc_module' in locals():
                mix_embeddings = disc_module.embed(mix_coeff)
                vox_embeddings = disc_module.embed(vox_coeff)
                bkg_embeddings = disc_module.embed(bkg_coeff)

            # Synthesis
            waveform = synthesis.forward(vox_coeff, use_sorting=exp_settings['dict_sorting'])

            # Waveform storing
            voxhat[p_start:p_end, :] = waveform.data.cpu().numpy()

            # Remove silent frames
            loud_x = (10. * (vox_tr_batch.norm(2., dim=1, keepdim=True).log10())).data.cpu().numpy()
            if loud_x >= exp_settings['loudness_threshold']:
                loss.append(losses.si_sdr(vox_tr_batch, waveform, scaling=True).item())
                additivity_loss.append((torch.norm(mix_coeff - vox_coeff - bkg_coeff, 1.) /
                                       (torch.norm(mix_coeff, 1.) + _eps)).item())

                vox_hat = synthesis.forward(mix_coeff * (vox_coeff/bkg_coeff).gt(0.5).float(),
                                            use_sorting=exp_settings['dict_sorting'])

                loss_from_masking.append(losses.si_sdr(vox_tr_batch, vox_hat, scaling=True).item())

                if compute_embeddings:
                    vox_hat_from_embd = synthesis.forward(vox_embeddings, use_sorting=exp_settings['dict_sorting'])
                    additivity_loss_embd.append((torch.norm(mix_embeddings - vox_coeff - bkg_coeff, 1.) /
                                                 (torch.norm(mix_embeddings, 1.) + _eps)).item())
                    loss_from_embd.append(losses.si_sdr(vox_tr_batch, vox_hat_from_embd, scaling=True).item())

                if stft_comparison:
                    vox_stft_mag = np.abs(librosa.stft(vox_tr_batch.cpu().numpy()[0],
                                                       n_fft=exp_settings['ft_size'],
                                                       win_length=exp_settings['ft_size'],
                                                       hop_length=exp_settings['hop_size']
                                                       )/np.sqrt(exp_settings['ft_size']))
                    bkg_stft_mag = np.abs(librosa.stft(bkg_tr_batch.cpu().numpy()[0],
                                                       n_fft=exp_settings['ft_size'],
                                                       win_length=exp_settings['ft_size'],
                                                       hop_length=exp_settings['hop_size']
                                                       )/np.sqrt(exp_settings['ft_size']))
                    mix_stft = librosa.stft(mix_tr_batch.cpu().numpy()[0],
                                            n_fft=exp_settings['ft_size'],
                                            win_length=exp_settings['ft_size'],
                                            hop_length=exp_settings['hop_size'])

                    mix_stft_mag = np.abs(mix_stft)/np.sqrt(exp_settings['ft_size'])

                    mask = np.greater(vox_stft_mag/(bkg_stft_mag + 1e-24), 0.5)*1.
                    vox_tr_batch_hat = librosa.istft(mix_stft * mask,
                                                     length=exp_settings['d_p_length'] * exp_settings['fs'],
                                                     hop_length=exp_settings['hop_size'],
                                                     win_length=exp_settings['ft_size'])
                    vox_tr_batch_hat = torch.autograd.Variable(torch.from_numpy(vox_tr_batch_hat).cuda()).float()

                    additivity_loss_stft.append((np.sum(np.abs(mix_stft_mag - vox_stft_mag - bkg_stft_mag)) /
                                                 (np.sum(mix_stft_mag) + _eps)))

                    loss_from_masking_stft.append(losses.si_sdr(vox_tr_batch,
                                                                vox_tr_batch_hat.unsqueeze(0), scaling=True).item())

            if make_plots:
                if loud_x >= exp_settings['loudness_threshold']:
                    # Representation
                    coeff_mag = synthesis.just_sort(mix_coeff).data.cpu().numpy()
                    coeff_mag = np.reshape(coeff_mag, (coeff_mag.shape[0] * coeff_mag.shape[1],
                                                       coeff_mag.shape[2]))
                    coeff_vox_mag = synthesis.just_sort(vox_coeff).data.cpu().numpy()
                    coeff_vox_mag = np.reshape(coeff_vox_mag, (coeff_vox_mag.shape[0] * coeff_vox_mag.shape[1],
                                                               coeff_vox_mag.shape[2]))

                    plt.imshow(coeff_vox_mag, aspect='auto', origin='lower')
                    plt.figure()
                    plt.imshow(coeff_mag, aspect='auto', origin='lower')
                    plt.show()

                    # Embeddings
                    if 'mix_embeddings' in locals() and 'vox_embeddings' in locals():
                        print('Computing Embeddings')
                        coeff_emb_mag = mix_embeddings.data.cpu().numpy()
                        coeff_emb_mag = np.reshape(coeff_emb_mag, (coeff_emb_mag.shape[0] * coeff_emb_mag.shape[1],
                                                   coeff_emb_mag.shape[2]))

                        coeff_emb_vox = vox_embeddings.data.cpu().numpy()
                        coeff_emb_vox = np.reshape(coeff_emb_vox, (coeff_emb_vox.shape[0] * coeff_emb_vox.shape[1],
                                                                   coeff_emb_vox.shape[2]))

                        plt.imshow(coeff_emb_vox, aspect='auto', origin='lower')
                        plt.figure()
                        plt.imshow(coeff_emb_mag, aspect='auto', origin='lower')
                        plt.show()

        if write_files:
            voxhat = np.reshape(voxhat, voxhat.shape[0] * voxhat.shape[1])
            io_methods.wav_write(voxhat, 44100, 16, 'vox_rec_'+str(file_set)+'.wav')

    # Store evaluation results
    np.save('results/eval_res/' + exp_settings['exp_id'] + '_sdr.npy', np.asarray(loss))
    np.save('results/eval_res/' + exp_settings['exp_id'] + '_mask_sdr.npy', np.asarray(loss_from_masking))
    np.save('results/eval_res/' + exp_settings['exp_id'] + '_adt.npy', np.asarray(additivity_loss))
    if compute_embeddings:
        np.save('results/eval_res/' + exp_settings['exp_id'] + '_lat_sdr.npy', np.asarray(loss_from_embd))
        np.save('results/eval_res/' + exp_settings['exp_id'] + '_lat_adt.npy', np.asarray(additivity_loss_embd))

    if additivity_loss_stft:
        np.save('results/eval_res/stft_adt.npy', np.asarray(additivity_loss_stft))
        np.save('results/eval_res/stft_mask_sdr.npy', np.asarray(loss_from_masking_stft))

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    c_set = [settings, settings_disc]

    # Testing
    perform_frontend_testing(make_plots=False,
                             write_files=False,
                             compute_embeddings=False,
                             stft_comparison=False, exp_settings=c_set[0])

# EOF
