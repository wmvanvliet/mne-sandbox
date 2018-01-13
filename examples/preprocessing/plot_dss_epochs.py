# -*- coding: utf-8 -*-

"""Denoising source separation applied to an Epochs object"""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

from mne_sandbox.preprocessing import dss


# file paths
data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
# import sample data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(events_fname)
# pick channels, filter, epoch
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
# reject = dict(eeg=180e-6, eog=150e-6)
reject = None
raw.filter(0.3, 30, method='iir', picks=picks)
epochs = mne.Epochs(raw, events, event_id=1, preload=True, picks=picks,
                    reject=reject)
epochs.pick_types(eeg=True)
evoked = epochs.average()

# perform DSS
dss_mat, dss_data = dss(epochs, data_thresh=1e-9, bias_thresh=1e-9)

evoked_data_clean = np.dot(dss_mat, evoked.data)
evoked_data_clean[4:] = 0.  # select 4 components
evoked_data_clean = np.dot(np.linalg.pinv(dss_mat), evoked_data_clean)

# plot
fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
plotdata = [evoked.data.T, evoked_data_clean.T, dss_data[:, 0].T]
linewidths = (1, 1, 0.6)
titles = ('evoked data (EEG only)',
          'evoked data after DSS (EEG only)',
          'first DSS component from each epoch (EEG only)')
for ax, dat, lw, ti in zip(axs, plotdata, linewidths, titles):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(1e3 * evoked.times, dat, linewidth=lw)
    ax.set_title(ti)
ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
