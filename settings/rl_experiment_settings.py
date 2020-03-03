# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

exp_settings = {
    # Experiment
    'exp_id': 'r-mcos-8',                       # An id to store specific experiment results
    'visualize': False,                         # Turn off for speed!

    # Signal settings
    'fs': 44100,                                # Sampling frequency
    'monaural': True,                           # Mono/Stereo
    'd_p_length': 1,                            # Length of each data point in seconds
    'num_of_multitracks': 100,                  # Number of multitracks used for training
    'set_size': 4,                              # Number of files for reading
    'noise_scalar': 1e-4,                       # Deviation of additive Gaussian noise
    'loudness_threshold': -10,                  # Silent segments threshold for optimizing the embedder

    # Encoder-decoder settings
    'use_sinc': False,                          # Whether to instantiate Sinc-Net or not
    'use_rand_enc': True,                       # Whether to instantiate a random encoder
    'fully_modulated': True,                    # All of the analysis bases are modulated
    'use_simple_conv_dec': False,               # Using a typical conv-transpose 1d for decoder
    'ft_size': 2048,                            # Length of the analysis filters in samples
    'ft_size_space': 800,                       # Number of components
    'hop_size': 256,                            # Analysis/Synthesis hop-size in samples
    'ft_syn_size': 2048,                        # Length of the synthesis filters in samples
    'learning_rate': 1e-4,                      # Learning rate
    'dict_sorting': True,                       # Sorting out wrt frequencies of the analysis/synthesis dictionary

    # Optimization settings
    'batch_size': 8,                            # Batch size
    'batch_overlap': 22050,                     # Overlap of time-domain waveforms between batches (in samples)
    'epochs': 8,                                # Number of iterations
    'lambda_reg': 0.5,                          # Lambda for smoothness regularization
                }
# EOF
