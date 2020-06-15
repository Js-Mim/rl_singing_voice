# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import numpy as np
import torch
from nn_modules import losses
from tools import helpers, visualize, nn_loaders
from settings.rl_experiment_settings import exp_settings
from torch.distributions import Normal


def perform_frontend_training(p, reg):
    print('ID: ' + exp_settings['exp_id'])
    # Instantiating data handler
    io_dealer = helpers.DataIO(exp_settings)

    # Number of file sets
    num_of_sets = exp_settings['num_of_multitracks']//exp_settings['set_size']

    # Initialize modules
    if exp_settings['visualize']:
        win_viz, win_viz_b = visualize.init_visdom()    # Web loss plotting
    analysis, synthesis = nn_loaders.build_frontend_model(flag='training', exp_settings=exp_settings)

    # Expected shapes
    data_shape = (exp_settings['batch_size'], exp_settings['d_p_length'] * exp_settings['fs'])
    noise_sampler = Normal(torch.zeros(data_shape), torch.ones(data_shape)*exp_settings['noise_scalar'])

    # Initialize optimizer and add the parameters that will be updated
    parameters_list = list(analysis.parameters()) + list(synthesis.parameters())
    optimizer = torch.optim.Adam(parameters_list, lr=exp_settings['learning_rate'])
    # Start of the training
    batch_indx = 0
    reg_loss = []
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
                bkg_tr_batch = io_dealer.batches_from_numpy(bkg[shuf_ind_bkg, :])

                vox_var = torch.autograd.Variable(vox_tr_batch, requires_grad=False)
                mix_var = torch.autograd.Variable(vox_tr_batch + bkg_tr_batch, requires_grad=False)

                # Sample noise
                noise = torch.autograd.Variable(noise_sampler.sample().cuda().float(),
                                                requires_grad=False)
                # 0 Mean
                vox_var -= vox_var.mean()
                mix_var -= mix_var.mean()

                # Target source forward pass
                vox_coeff = analysis.forward(vox_var + noise)
                waveform = synthesis.forward(vox_coeff, use_sorting=exp_settings['dict_sorting'])

                # Mixture signal forward pass
                mix_coeff = analysis.forward(mix_var)

                # Loss functions
                rec_loss = losses.neg_snr(vox_var, waveform)
                rep_loss = losses.sinkhorn_dist(mix_coeff, p=p, reg=reg)
                loss = rec_loss + exp_settings['lambda_reg'] * rep_loss

                # Optimize for reconstruction & smoothness
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if exp_settings['visualize']:
                    # Visualization
                    win_viz = visualize.viz.line(X=np.arange(batch_indx, batch_indx + 1),
                                                 Y=np.reshape(rec_loss.item(), (1,)),
                                                 win=win_viz, update='append')
                    win_viz_b = visualize.viz.line(X=np.arange(batch_indx, batch_indx + 1),
                                                   Y=np.reshape(smt_loss.item(), (1,)),
                                                   win=win_viz_b, update='append')
                    batch_indx += 1

        print('--- Saving Model ---')
        torch.save(analysis.state_dict(), 'results/analysis_' + exp_settings['exp_id'] + '.pytorch')
        torch.save(synthesis.state_dict(), 'results/synthesis_' + exp_settings['exp_id'] + '.pytorch')

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    p_val = 1
    lambda_val = 1.3 
    perform_frontend_training(p=p_val, reg=lambda_val)

# EOF
