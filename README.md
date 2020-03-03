# Unsupervised Interpretable Representation Learning for Singing Voice Separation

This repository contains the PyTorch (1.4) implementation of a method, based on (convolutional) neural networks, to learn representations from music signals that could be used for singing voice separation. The proposed method is employing a decoder to reconstruct the analysed signal. The decoder relies on cosine functions, that could be sorted, leading to interpretable representations. Furthermore, the representation is non-negative and real-valued, and it could employed, fairly easily, by current spervised models for music source separation. The proposed method is inspired by [Sinc-Net](https://github.com/mravanelli/SincNet/) and [dDSP](https://github.com/magenta/ddsp).

### Authors

[S.I. Mimilakis](https://github.com/Js-Mim), [K. Drossos](https://www.tuni.fi/en/konstantinos-drossos), [G. Schuller](https://www.tu-ilmenau.de/mt-ams/personen/schuller-gerald/)

# What's inside?

* Code for the neural architectures used in our study and their corresponding minimization objectives (`nn_modules/`)
* Code for performing the unsupervised training (`scripts/exp_rl_*`)
* Code for reconstructing the signal(s) (`scripts/exp_fe_test.py`)
* Code for inspecting the outcome(s) of the training (`scripts/make_plots.py`)
* Code for visualizing loss functions, reading/writing audio files, and creating batches (`tools/`)
* Perks (unreported implementations/routines)
** The discriminator-like objective, as a proxy to mutual information, reported [here](https://arxiv.org/pdf/1812.00271.pdf)

# Beyond code

* Download the optimized models &rarr; [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3695332.svg)](https://doi.org/10.5281/zenodo.3695332)
* Demo [page](tba)

## Reference


<details><summary>If you use this code for your research – Cite this work/summary>
```latex
@misc{mim20_uirl,  
  author={S. I. Mimilakis and K. Drossos and G. Schuller},  
  title={Unsupervised Interpretable Representation Learning for Singing Voice Separation},  
  year={2020},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```
  </details>
  
### License

MIT
