# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import os
import torch
import numpy as np
from tools import io_methods as io


class DataIO:
    """ Class for data
        input-output passing.
    """
    def __init__(self, exp_settings={}):
        super(DataIO, self).__init__()

        # definitions
        self.dataset_path = '/some/path/Datasets/musdb18/'
        self.keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
        self.foldersList = ['train', 'test']
        self.d_p_length = exp_settings['d_p_length']
        self.fs = exp_settings['fs']
        self.batch_overlap = exp_settings['batch_overlap']
        self.batch_size = exp_settings['batch_size']
        self.loudness_threshold = exp_settings['loudness_threshold']
        try:
            self.channel_augmentation = exp_settings['ch_augment']
            self.gain_augmentation = exp_settings['gain_augment']
        except KeyError:
            self.channel_augmentation = False
            self.gain_augmentation = False

    def get_data(self, current_set, set_size, monaural=True, dev=True):
        """
            Method to load training data.
                current_set      : (int)          An integer denoting the current training set (Starting from "1").
                set_size         : (int)          The amount of files a set has.
                monaural         : (bool)         Return monaural audio files or not.

            Returns:
                mix_out          : (numpy array)  The mixture signal waveform     (samples x channels)
                vox_out          : (numpy array)  The vocal signal waveform       (samples x channels)
                bkg_out          : (numpy array)  The background signal waveform  (samples x channels)
        """
        if dev:
            folders_list = self.foldersList[0]
        else:
            folders_list = self.foldersList[1]

        # Generate full paths for dev and test
        dev_list = sorted(os.listdir(self.dataset_path + folders_list))
        dev_list = [self.dataset_path + folders_list + '/' + i for i in dev_list]

        # Current lists for training
        c_train_mlist = dev_list[(current_set - 1) * set_size: current_set * set_size]

        mix_out = np.array([])
        vox_out = np.array([])
        bkg_out = np.array([])
        if not monaural:
            mix_out.shape = (0, 2)
            vox_out.shape = (0, 2)
            bkg_out.shape = (0, 2)

        for index in range(len(c_train_mlist)):
            # Reading
            vocal_signal, or_fs = io.wav_read(os.path.join(c_train_mlist[index], self.keywords[3]), mono=False)
            mix_signal, or_fs = io.wav_read(os.path.join(c_train_mlist[index], self.keywords[4]), mono=False)

            if self.channel_augmentation:
                fl_channels = np.random.permutation(2)
                vocal_signal = vocal_signal[:, fl_channels]
                mix_signal = mix_signal[:, fl_channels]
            bkg_signal = mix_signal - vocal_signal

            if self.gain_augmentation:
                gain = np.random.uniform(0.7, 1.05)
                vocal_signal *= gain
                mix_signal = bkg_signal + vocal_signal

            if monaural and len(mix_signal.shape) == 2:
                vocal_signal = np.mean(vocal_signal, axis=-1)
                mix_signal = np.mean(mix_signal, axis=-1)
                bkg_signal = np.mean(bkg_signal, axis=-1)

            mix_out = np.concatenate([mix_out, mix_signal], axis=0)
            vox_out = np.concatenate([vox_out, vocal_signal], axis=0)
            bkg_out = np.concatenate([bkg_out, bkg_signal], axis=0)

        return mix_out, vox_out, bkg_out

    def gimme_batches(self, wav_in):
        d_p_length_samples = self.d_p_length * self.fs
        resize_factor = d_p_length_samples - self.batch_overlap
        trim_frame = wav_in.shape[0] % resize_factor
        trim_frame -= resize_factor
        trim_frame = np.abs(trim_frame)

        # Zero-padding
        if trim_frame != 0:
            wav_in = np.pad(wav_in, (0, trim_frame), 'constant', constant_values=0)

        # Reshaping with overlap
        strides = (resize_factor * wav_in.itemsize, wav_in.itemsize)
        shape = (1 + int((wav_in.nbytes - d_p_length_samples * wav_in.itemsize) / strides[0]),
                 d_p_length_samples)
        wav_in = np.lib.stride_tricks.as_strided(wav_in, shape=shape, strides=strides)

        b_trim_frame = wav_in.shape[0] % self.batch_size
        b_trim_frame -= self.batch_size
        b_trim_frame = np.abs(b_trim_frame)
        # Zero-padding
        if b_trim_frame != 0:
            wav_in = np.pad(wav_in, (0, b_trim_frame), 'constant', constant_values=(0, 0))

        return wav_in[:, :d_p_length_samples]

    def gimme_batches_stereo(self, st_wav_in):
        wav_l = self.gimme_batches(st_wav_in[:, 0])
        wav_r = self.gimme_batches(st_wav_in[:, 1])
        return np.stack((wav_l, wav_r), axis=1)

    @staticmethod
    def batches_from_numpy(st_batch_in):
        if torch.has_cuda:
            return torch.from_numpy(st_batch_in).cuda().float()
        else:
            return torch.from_numpy(st_batch_in).float()


if __name__ == '__main__':
    io_dealer = DataIO()
    mix, vox, bkg = io_dealer.get_data(42, 4, monaural=True)
    b_mix = io_dealer.gimme_batches(mix)


# EOF
