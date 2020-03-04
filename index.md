---
layout: default
---
We present a method for learning interpretable music signal representations directly from waveform signals. Our method can be trained using unsupervised objectives and relies on the denoising auto-encoder model that uses a simple sinusoidal model as decoding functions to reconstruct the singing voice. 
The supplementary material focuses on three "qualitative" aspects:
* Audio examples of reconstructed music signals
* Visual examples of the representation
* Visual examples of frequency responses

# Audio examples
| **Example** | **Input** | **Reconstructed** |
|:-----------:|:---------:|:-----------------:|
| Mix Ex. 1 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_1_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_1_6_.wav"></audio> |
| Vox Ex. 1 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_1_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_1_6_.wav"></audio> |
| Mix Ex. 2 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_2_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_2_3_.wav"></audio> |
| Vox Ex. 2 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_2_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_2_3_.wav"></audio> |
 


# Visual examples

# Frequency responses
We used the best performing model, reported in our paper, and compute the discrete Fourier transform (DFT) of the
resulting basis signals, that the decoder is is using.