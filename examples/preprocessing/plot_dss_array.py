# -*- coding: utf-8 -*-

"""Denoising source separation"""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

from __future__ import division
import numpy as np
from mne_sandbox.preprocessing import dss
from matplotlib import pyplot as plt


def rms(data):
    return np.sqrt(np.mean(data ** 2, axis=-1, keepdims=True))

snr = 0.1
noise_dims = 20
rand = np.random.RandomState(123)

# create synthetic data
n_trials = 100
n_times = 1000
n_channels = 32
pad = np.zeros(n_times // 3)
signal_nsamps = n_times - 2 * pad.size
signal = np.sin(2 * np.pi * np.arange(signal_nsamps) / signal_nsamps)
signal = np.r_[pad, signal, pad]
signal_chans = rand.randn(n_channels, 1) * signal[np.newaxis, :]
channel_noise = rand.randn(n_channels, noise_dims)
trial_noise = rand.randn(n_trials, n_times, noise_dims)
noise = np.einsum('ijk,lk->ilj', trial_noise, channel_noise)
data = 4e-6 * (noise / rms(noise) + snr * signal_chans / rms(signal_chans))

# perform DSS
dss_mat, dss_data = dss.dss(data, data_thresh=1e-3, bias_thresh=1e-3)

# plot
fig, axs = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
plotdata = [signal_chans.T, data[0].T, dss_data[:, 0].T]
linewidths = (1, 0.3, 0.4)
titles = ('synthetic signal with random weights for each channel',
          'one trial, all channels, after noise addition (SNR=0.1)',
          'First DSS component from each trial')
for ax, dat, lw, ti in zip(axs, plotdata, linewidths, titles):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    _ = ax.plot(dat, linewidth=lw)
    _ = ax.set_title(ti)
ax.set_xlabel('samples')
plt.tight_layout()
