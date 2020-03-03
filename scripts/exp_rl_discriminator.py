# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import numpy as np
import torch
from nn_modules import losses
from tools import helpers, visualize, nn_loaders
from settings.rl_disc_experiment_settings import exp_settings
from torch.distributions import Normal


def perform_frontend_training():
    print('ID: ' + exp_settings['exp_id'])
    # Instantiating data handler
    io_dealer = helpers.DataIO(exp_settings=exp_settings)

    # Number of file sets
    num_of_sets = exp_settings['num_of_multitracks']//exp_settings['set_size']

    # Initialize modules
    # Initialize modules
    if exp_settings['visualize']:
        win_viz, win_viz_b = visualize.init_visdom()    # Web loss plotting
    analysis, synthesis = nn_loaders.build_frontend_model(flag='training', exp_settings=exp_settings)
    disc = nn_loaders.build_discriminator(flag='training', exp_settings=exp_settings)
    sigmoid = torch.nn.Sigmoid()

    # Expected shapes
    data_shape = (exp_settings['batch_size'], exp_settings['d_p_length'] * exp_settings['fs'])
    noise_sampler = Normal(torch.zeros(data_shape), torch.ones(data_shape)*exp_settings['noise_scalar'])

    # Initialize optimizer and add the parameters that will be updated
    parameters_list = list(analysis.parameters()) + list(synthesis.parameters()) + list(disc.parameters())

    optimizer = torch.optim.Adam(parameters_list, lr=exp_settings['learning_rate'])

    # Start of the training
    batch_indx = 0
    for epoch in range(1, exp_settings['epochs'] + 1):
        for file_set in range(1, num_of_sets + 1):
            # Load a sub-set of the recordings
            _, vox, bkg = io_dealer.get_data(file_set, exp_settings['set_size'], monaural=exp_settings['monaural'])

            # Create batches
            vox = io_dealer.gimme_batches(vox)
            bkg = io_dealer.gimme_batches(bkg)

            # Compute the total number of batches contained in this sub-set
            num_batches = vox.shape[0] // exp_settings['batch_size']

            # Compute permutations for random shuffling
            perm_in_vox = np.random.permutation(vox.shape[0])
            perm_in_bkg = np.random.permutation(bkg.shape[0])
            for batch in range(num_batches):
                shuf_ind_vox = perm_in_vox[batch * exp_settings['batch_size']: (batch + 1) * exp_settings['batch_size']]
                shuf_ind_bkg = perm_in_bkg[batch * exp_settings['batch_size']: (batch + 1) * exp_settings['batch_size']]
                vox_tr_batch = io_dealer.batches_from_numpy(vox[shuf_ind_vox, :])
                bkg_tr_batch = io_dealer.batches_from_numpy(vox[shuf_ind_bkg, :])

                vox_var = torch.autograd.Variable(vox_tr_batch, requires_grad=False)
                bkg_var = torch.autograd.Variable(bkg_tr_batch, requires_grad=False)
                mix_var = torch.autograd.Variable(vox_tr_batch + bkg_tr_batch, requires_grad=False)

                # Sample noise
                noise = torch.autograd.Variable(noise_sampler.sample().cuda().float(), requires_grad=False)

                # 0 Mean
                vox_var -= vox_var.mean()
                bkg_var -= bkg_tr_batch.mean()
                mix_var -= mix_var.mean()

                # Target source forward pass
                vox_coeff = analysis.forward(vox_var + noise)
                waveform = synthesis.forward(vox_coeff, use_sorting=exp_settings['dict_sorting'])

                # Mixture and Background signals forward pass
                mix_coeff = analysis.forward(mix_var)
                bkg_coeff = analysis.forward(bkg_var)

                # Loss functions
                rec_loss = losses.neg_snr(vox_var, waveform)
                smt_loss = exp_settings['lambda_reg'] * losses.tot_variation_2d(mix_coeff)

                loss = rec_loss + smt_loss

                # Optimize for reconstruction & smoothness
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # Optimize with discriminator
                # Remove silent frames
                c_loud_x = (10. * (vox_tr_batch.norm(2., dim=1, keepdim=True).log10())).data.cpu().numpy()
                # Which segments are below the threshold?
                loud_locs = np.where(c_loud_x > exp_settings['loudness_threshold'])[0]
                vox_coeff = vox_coeff[loud_locs]
                if vox_coeff.size(0) > 2:
                    # Make sure we are getting unmatched pairs
                    bkg_coeff = bkg_coeff[loud_locs]
                    vox_coeff_shf = vox_coeff[np.random.permutation(vox_coeff.size(0))]

                    # Sample from discriminator
                    y_neg = sigmoid(disc.forward(vox_coeff, bkg_coeff))
                    y_pos = sigmoid(disc.forward(vox_coeff, vox_coeff_shf))

                    # Compute discriminator loss
                    disc_loss = losses.bce(y_pos, y_neg)

                    # Optimize the discriminator
                    optimizer.zero_grad()
                    disc_loss.backward()
                    optimizer.step()

                else:
                    pass

                if exp_settings['visualize']:
                    # Visualization
                    win_viz = visualize.viz.line(X=np.arange(batch_indx, batch_indx + 1),
                                                 Y=np.reshape(rec_loss.item(), (1,)),
                                                 win=win_viz, update='append')
                    win_viz_b = visualize.viz.line(X=np.arange(batch_indx, batch_indx + 1),
                                                   Y=np.reshape(disc_loss.item(), (1,)),
                                                   win=win_viz_b, update='append')
                    batch_indx += 1

        if not torch.isnan(loss) and not torch.isnan(disc_loss):
            print('--- Saving Model ---')
            torch.save(analysis.state_dict(), 'results/analysis_' + exp_settings['exp_id'] + '.pytorch')
            torch.save(synthesis.state_dict(), 'results/synthesis_' + exp_settings['exp_id'] + '.pytorch')
            torch.save(disc.state_dict(), 'results/disc_' + exp_settings['exp_id'] + '.pytorch')
        else:
            break

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Training
    perform_frontend_training()


# EOF
