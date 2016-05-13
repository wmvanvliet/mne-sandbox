"""
==============================================================
Compute phase-amplitude coupling measures between signals
==============================================================
Simulate phase-amplitude coupling between two signals, and computes
several PAC metrics between them. Calculates PAC for all timepoints, as well
as for time-locked PAC responses.
References
----------
[1] Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT. "High gamma power is phase-locked to
    theta oscillations in human neocortex." Science. 2006.
[2] Tort ABL, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude
    coupling between neuronal oscillations of different frequencies. Journal of
    Neurophysiology. 2010.
"""
# Author: Chris Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne_sandbox.connectivity import (simulate_pac_signal,
                                      phase_amplitude_coupling)

print(__doc__)

###############################################################################
# Phase-amplitude coupling (PAC) is a technique to determine if the
# amplitude of a high-frequency signal is locked to the phase
# of a low-frequency signal. The phase_amplitude_coupling function
# calculates PAC between pairs of signals for one or multiple
# time windows. In this example, we'll simulate two signals. One
# of the signals has an amplitude that is locked to the phase of
# the other. We'll calculate PAC for a number of time points, and
# in both directions to show how PAC responds.

# Define parameters for our simulated signal
sfreq = 1000.
f_phase = 4
f_amp = 40
frac_pac = .99  # This is the fraction of PAC to use
mag_ph = 4
mag_am = 1

# These are the times where PAC is active in our simulated signal
n_secs = 20.
time = np.arange(0, n_secs, 1. / sfreq)
event_times = [1, 5, 9, 13, 17]
event_dur = 2.

# Create a time maks that defines when PAC is active
msk_pac_times = np.zeros_like(time).astype(bool)
for i_time in event_times:
    msk_pac_times += mne.utils._time_mask(time, i_time, i_time + event_dur)

# Now simulate two signals. First, a low-frequency phase
# that modulates high-frequency amplitude
_, lo_pac, hi_pac = simulate_pac_signal(time, f_phase, f_amp, mag_ph, mag_am,
                                        frac_pac=frac_pac,
                                        mask_pac_times=msk_pac_times)

# Now two signals with no relationship between them
_, lo_none, hi_none = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                          mag_am, frac_pac=0,
                                          mask_pac_times=msk_pac_times)

# Finally we'll mix them up.
# The low-frequency phase of signal A...
signal_a = lo_pac + hi_none
# Modulates the high-frequency amplitude of signal B. But not the reverse.
signal_b = lo_none + hi_pac


# We'll visualize these signals. A on the left, B on the right
# The top row is a combination of the middle and bottom row
labels = ['Combined Signal', 'Lo-Freq signal', 'Hi-freq signal']
data = [[signal_a, lo_none, hi_pac],
        [signal_b, lo_pac, hi_none]]
f, axs = plt.subplots(3, 2, figsize=(10, 5))
for axcol, i_data in zip(axs.T, data):
    for ax, i_sig, i_label in zip(axcol, i_data, labels):
        ax.plot(time, i_sig)
        ax.set_title(i_label, fontsize=20)
_ = plt.setp(axs, xlim=[8, 12])
plt.tight_layout()

# Create a raw array from the simulated data
info = mne.create_info(['pac_hi', 'pac_lo'], sfreq, 'eeg')
raw = mne.io.RawArray([signal_a, signal_b], info)

# The PAC function needs a lower and upper bound for each frequency
f_phase_bound = (f_phase-.1, f_phase+.1)
f_amp_bound = (f_amp-2, f_amp+2)

# First we'll calculate PAC for the entire timeseries.
# We'll use a few PAC metrics to compare.
iter_pac_funcs = [['glm', 'ozkurt'], ['plv']]
win_size = 1.  # In seconds
step_size = .1
pac_times = np.array(
    [(i, i + win_size)
     for i in np.arange(0, np.max(time) - win_size, step_size)])

# Here we specify indices to calculate PAC in both directions
ixs = np.array([[0, 1],
                [1, 0]])
f, axs = plt.subplots(2, 1, figsize=(10, 5))
all_pac = []
for pac_funcs in iter_pac_funcs:
    pac = phase_amplitude_coupling(
        raw, (f_phase-.1, f_phase+.1), (f_amp-.5, f_amp+.5), ixs,
        pac_func=pac_funcs, tmin=pac_times[:, 0], tmax=pac_times[:, 1],
        n_cycles=3)
    if isinstance(pac, np.ndarray):
        pac = [pac]
    for i_pac, i_name in zip(pac, pac_funcs):
        for i_pac_ix, ax in zip(i_pac.squeeze(), axs):
            ax.plot(i_pac_ix.squeeze(), label=i_name)
axs[0].legend()
axs[0].set_title('PAC: low-freq A to high-freq B', fontsize=20)
axs[1].set_title('PAC: low-freq B to high-freq A', fontsize=20)
_ = plt.setp(axs, ylim=[0, 1.1])
plt.tight_layout()


# We can also calculate event-locked PAC
# by supplying a list of event indices
ev = np.array(event_times) * sfreq
ev = ev.astype(int)

pac_funcs = ['glm', 'ozkurt']
colors = ['b', 'r']
win_size = 1.
ev_tmin = -2.
ev_tmax = 3.
pac_times = np.array([(i, i + win_size)
                      for i in np.arange(ev_tmin, ev_tmax, step_size)])
pac = phase_amplitude_coupling(
    raw, (f_phase-.1, f_phase+.1), (f_amp-.5, f_amp+.5), ixs,
    pac_func=pac_funcs, tmin=pac_times[:, 0], tmax=pac_times[:, 1],
    ev=ev, concat_epochs=False)

# This allows us to calculate the stability of PAC across epochs
f, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
for ii, (i_pac, i_pac_name, color) in enumerate(zip(pac, pac_funcs, colors)):
    mn_pac = i_pac.mean(0)
    ste_pac = i_pac.std(0) / np.sqrt(i_pac.shape[0])
    for i_pac_mn, i_pac_ste, ax in zip(mn_pac, ste_pac, axs):
        ax.fill_between(pac_times[:, 0], i_pac_mn - i_pac_ste,
                        i_pac_mn + i_pac_ste, color=color, label=i_pac_name)
axs[0].legend()
axs[0].set_title('Time-locked PAC: Signal A to Signal B')
axs[1].set_title('Time-locked PAC: Signal B to Signal A')

plt.tight_layout()
plt.show(block=True)
