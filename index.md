---
layout: default
---
We present a method for learning interpretable music signal representations directly from waveform signals. Our method can be trained using unsupervised objectives and relies on the denoising auto-encoder model that uses a simple sinusoidal model as decoding functions to reconstruct the singing voice. 
The supplementary material focuses on three "qualitative" aspects:
* Audio examples of reconstructed music signals
* Visual examples of the representation
* Visual examples of frequency responses

# Audio examples
Randomly sampled mixture (`mix`) and singing voice (`vox`) segments of 3 seconds, from the test sub-set.

|**Example**|**Input**|**Reconstructed**|
|:---------:|:-------:|:---------------:|
| Mix Ex. 1 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_1_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_1_6_.wav"></audio> |
| Vox Ex. 1 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_1_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_1_6_.wav"></audio> |
| Mix Ex. 2 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_2_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_2_3_.wav"></audio> |
| Vox Ex. 2 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_2_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_2_3_.wav"></audio> |
| Mix Ex. 3 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_3_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_3_6_.wav"></audio> | 
| Vox Ex. 3 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_3_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_3_6_.wav"></audio> | 
| Mix Ex. 4 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_4_7_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_4_7_.wav"></audio> | 
| Vox Ex. 4 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_4_7_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_4_7_.wav"></audio> | 
| Mix Ex. 5 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_10_10_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_10_10_.wav"></audio> | 
| Vox Ex. 5 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_10_10_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_10_10_.wav"></audio> | 
| Mix Ex. 6 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_12_15_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_12_15_.wav"></audio> | 
| Vox Ex. 6 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_12_15_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_12_15_.wav"></audio> | 
| Mix Ex. 7 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_13_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_13_6_.wav"></audio> | 
| Vox Ex. 7 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_13_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_13_6_.wav"></audio> | 
| Mix Ex. 8 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_18_9_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_18_9_.wav"></audio> | 
| Vox Ex. 8 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_18_9_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_18_9_.wav"></audio> | 
| Mix Ex. 9 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_20_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_20_6_.wav"></audio> | 
| Vox Ex. 9 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_20_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_20_6_.wav"></audio> | 
| Mix Ex. 10 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_39_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_39_3_.wav"></audio> | 
| Vox Ex. 10 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_39_3_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_39_3_.wav"></audio> | 
| Mix Ex. 11 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_true_47_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_47_6_.wav"></audio> | 
| Vox Ex. 11 | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_true_47_6_.wav"></audio> | <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_47_6_.wav"></audio> | 


# Representation examples
<p align="center"> <b>Example 1</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_1_6.png" width="470" height="360" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_1_6.png" width="470" height="360" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_1_6_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_1_6_.wav"></audio>
</p>

<p align="center"> <b>Example 2</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_2_3.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_2_3.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_2_3_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_2_3_.wav"></audio>
</p>

<p align="center"> <b>Example 3</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_3_6.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_3_6.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_3_6_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_3_6_.wav"></audio>
</p>

<p align="center"> <b>Example 4</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_4_7.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_4_7.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_4_7_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_4_7_.wav"></audio>
</p>

<p align="center"> <b>Example 5</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_10_10.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_10_10.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_10_10_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_10_10_.wav"></audio>
</p>

<p align="center"> <b>Example 6</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_12_15.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_12_15.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_12_15_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_12_15_.wav"></audio>
</p>

<p align="center"> <b>Example 7</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_13_6.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_13_6.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_13_6_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_13_6_.wav"></audio>
</p>

<p align="center"> <b>Example 8</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_18_9.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_18_9.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_18_9_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_18_9_.wav"></audio>
</p>

<p align="center"> <b>Example 9</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_20_6.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_20_6.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_20_6_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_20_6_.wav"></audio>
</p>

<p align="center"> <b>Example 10</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_39_3.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_39_3.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_39_3_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_39_3_.wav"></audio>
</p>

<p align="center"> <b>Example 11</b></p>

<p align="center"> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_m_47_6.png" width="460" height="380" /> <img class="center" src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/figures/A_v_47_6.png" width="460" height="380" /> 

<audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/mix_rec_47_6_.wav"></audio> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <audio controls="1"><source src="https://raw.githubusercontent.com/Js-Mim/rl_singing_voice/gh-pages/audio_files/vox_rec_47_6_.wav"></audio>
</p>


# Frequency responses
We used the best performing model, reported in our paper, and compute the discrete Fourier transform (DFT) of the
resulting basis signals, that the decoder is is using.