# -*- coding: utf-8 -*-

"""Denoising source separation applied to an Epochs object"""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import mne
from os import path as op
from mne_sandbox.preprocessing import dss
from mne.datasets import sample
from matplotlib import pyplot as plt


# file paths
data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
evokeds_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
# import sample data
raw = mne.io.Raw(raw_fname, preload=True)
events = mne.read_events(events_fname)
evokeds = mne.read_evokeds(evokeds_fname)
# pick channels, filter, epoch
picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.filter(0.3, 30, method='iir', picks=picks)
epochs = mne.Epochs(raw, events, event_id=1, preload=True, picks=picks)
evoked = evokeds[0]  # left auditory only
evoked = evoked.pick_types(meg=False, eeg=True)

# perform DSS
dss_mat, dss_data = dss(epochs, data_thresh=1e-9, bias_thresh=1e-9)

# plot
fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
plotdata = [evoked.data.T, dss_data[:, 0].T]
linewidths = (1, 0.6)
titles = ('evoked data (EEG only)',
          'first DSS component from each epoch (EEG only)')
for ax, dat, lw, ti in zip(axs, plotdata, linewidths, titles):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(dat, linewidth=lw)
    ax.set_title(ti)
ax.set_xlabel('samples')
plt.tight_layout()
