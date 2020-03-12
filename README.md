# Unsupervised Interpretable Representation Learning for Singing Voice Separation

This repository contains the PyTorch (1.4) implementation of our method for representation learning. Our method is based on (convolutional) neural networks, to learn representations from music signals that could be used for singing voice separation. The proposed method is employing a decoder that relies on cosine functions. The resulting representation is non-negative and real-valued, and it could employed, fairly easily, by current supervised models for music source separation. The proposed method is inspired by [Sinc-Net](https://github.com/mravanelli/SincNet/) and [dDSP](https://github.com/magenta/ddsp).

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/method_overview.png" width="560" height="260" /> </p>

### Authors

[S.I. Mimilakis](https://github.com/Js-Mim), [K. Drossos](https://www.tuni.fi/en/konstantinos-drossos), [G. Schuller](https://www.tu-ilmenau.de/mt-ams/personen/schuller-gerald/)

# What's inside?

* Code for the neural architectures used in our study and their corresponding minimization objectives (`nn_modules/`)
* Code for performing the unsupervised training (`scripts/exp_rl_*`)
* Code for reconstructing the signal(s) (`scripts/exp_fe_test.py`)
* Code for inspecting the outcome(s) of the training (`scripts/make_plots.py`)
* Code for visualizing loss functions, reading/writing audio files, and creating batches (`tools/`)
* Perks (unreported implementations/routines)
  * The discriminator-like objective, as a proxy to mutual information, reported [here](https://arxiv.org/pdf/1812.00271.pdf)

# What's not inside!

* Our [paper](https://arxiv.org/pdf/2003.01567.pdf)
* Additional [results](https://js-mim.github.io/rl_singing_voice/) that didn't fit in the paper
* The optimized models &rarr; [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3707885.svg)](https://doi.org/10.5281/zenodo.3707885)
* The used dataset &rarr; [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3338373.svg)](https://doi.org/10.5281/zenodo.3338373)

# How to use
### Training
1. Download the dataset and declare the path of the downloaded dataset in `tools/helpers.py`
2. Apply any desired changes to the model by tweeking the parameters in `settings/rl_experiment_settings.py`
3. Execute `scripts/exp_rl_vanilla.py`

### Testing
1. Download the dataset and declare the path of the downloaded dataset in `tools/helpers.py`
2. Download the results and place them under the `results` folder
3. Load up the desired model by declaring the experiment id in `settings/rl_experiment_settings.py` (e.g. `r-mcos8`)
4. Execute `scripts/exp_fe_test.py` (some arguments for plotting and file writing are necesary)

# Reference
If you find this code useful for your research, cite our paper:

```latex
  @misc{mim20_uirl,  
  author={S. I. Mimilakis and K. Drossos and G. Schuller},  
  title={Unsupervised Interpretable Representation Learning for Singing Voice Separation},  
  year={2020},
  eprint={2003.01567},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
  }
  ```
  
# Acknowledgements

Stylianos Ioannis Mimilakis is supported in part by the German ResearchFoundation (AB 675/2-1, MU 2686/11-1). 

# License

MIT


