import numpy as np
from mne.filter import band_pass_filter
from mne.utils import _time_mask
from mne.time_frequency import cwt_morlet
from mne.preprocessing import peak_finder
from mne.utils import ProgressBar
import mne
import warnings


# Supported PAC functions
_pac_funcs = ['plv', 'glm', 'mi_tort', 'mi_canolty', 'ozkurt', 'otc']
# Calculate the phase of the amplitude signal for these PAC funcs
_hi_phase_funcs = ['plv']


def phase_amplitude_coupling(inst, f_phase, f_amp, ixs, pac_func='ozkurt',
                             ev=None, tmin=None, tmax=None, n_cycles=None,
                             scale_amp_func=None,  return_data=False,
                             concat_epochs=False, n_jobs=1, verbose=None):
    """ Compute phase-amplitude coupling between pairs of signals using pacpy.

    Parameters
    ----------
    inst : an instance of Raw or Epochs
        The data used to calculate PAC
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amp : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, list of strings
        The function for estimating PAC. Corresponds to functions in pacpy.pac.
        Must be one of ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt'].
    ev : array, shape (n_events, 3) | array, shape (n_events,) | None
        MNE events array. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided. If
        ndim == 1, it is assumed to be event indices, and all events will be
        grouped together.
    tmin : float | list of floats, shape (n_pac_windows,) | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index. If a list of floats is given, then PAC is calculated
        for each pair of tmin and tmax.
    tmax : float | list of floats, shape (n_pac_windows,) | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index. If a list of floats is given, then PAC is calculated
        for each pair of tmin and tmax.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., skl.preprocessing.scale
    return_data : bool
        If True, return the phase and amplitude data along with the PAC values.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs, [n_events])
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs.

    References
    ----------
    [1] This function uses the PacPy modulte developed by the Voytek lab.
        https://github.com/voytekresearch/pacpy
    """
    from mne.io.base import _BaseRaw
    if not isinstance(inst, _BaseRaw):
        raise ValueError('Must supply either Epochs or Raw')
    sfreq = inst.info['sfreq']
    data = inst[:, :][0]
    pac = _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                                    pac_func=pac_func, ev=ev,
                                    tmin=tmin, tmax=tmax, n_cycles=n_cycles,
                                    scale_amp_func=scale_amp_func,
                                    return_data=return_data,
                                    concat_epochs=concat_epochs,
                                    n_jobs=n_jobs, verbose=verbose)
    # Collect the data properly
    if return_data is True:
        pac, data_ph, data_am = pac
        return pac, data_ph, data_am
    else:
        return pac


def _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='plv', ev=None, ev_grouping=None,
                              tmin=None, tmax=None, n_cycles=None,
                              scale_amp_func=None, return_data=False,
                              concat_epochs=False, n_jobs=1,
                              verbose=None):
    """ Compute phase-amplitude coupling using pacpy.

    Parameters
    ----------
    data : array, shape ([n_epochs], n_channels, n_times)
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amp : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, list of strings
        The function for estimating PAC. Corresponds to functions in pacpy.pac.
        Must be one of ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt'].
    ev : array, shape (n_events, 3) | array, shape (n_events,) | None
        MNE events array. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided. If
        ndim == 1, it is assumed to be event indices, and all events will be
        grouped together.
    tmin : float | list of floats, shape (n_pac_windows,) | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index. If a list of floats is given, then PAC is calculated
        for each pair of tmin and tmax.
    tmax : float | list of floats, shape (n_pac_windows,) | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index. If a list of floats is given, then PAC is calculated
        for each pair of tmin and tmax.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., skl.preprocessing.scale
    return_data : bool
        If True, return the phase and amplitude data along with the PAC values.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    pac_out : array, list of arrays, dtype float, shape (n_pairs,
        [n_events, n_times]) The computed phase-amplitude coupling between
        each pair of data sources given in ixs.
    """
    from ..externals.pacpy import pac as ppac
    from ..externals.pacpy.filt import firf
    pac_func = np.atleast_1d(pac_func)
    for i_func in pac_func:
        if i_func not in _pac_funcs:
            raise ValueError("PAC function %s is not supported" % i_func)
    ixs = np.array(ixs, ndmin=2)
    tmin = 0 if tmin is None else tmin
    tmin = np.atleast_1d(tmin)
    n_times = len(tmin)
    tmax = (data.shape[-1] - 1) / float(sfreq) if tmax is None else tmax
    tmax = np.atleast_1d(tmax)

    if data.ndim != 2:
        raise ValueError('Data must be shape (n_channels, n_times)')
    if ixs.shape[1] != 2:
        raise ValueError('Indices must have have a 2nd dimension of length 2')
    if len(f_phase) != 2 or len(f_amp) != 2:
        raise ValueError('Frequencies must be specified w/ a low/hi tuple')
    if len(tmin) != len(tmax):
        raise ValueError('tmin and tmax have differing lengths')

    print('Pre-filtering data and extracting phase/amplitude...')
    hi_phase = np.unique([i_func in _hi_phase_funcs for i_func in pac_func])
    if len(hi_phase) != 1:
        raise ValueError("Can't mix pac funcs that use both hi-freq phase/amp")
    hi_phase = bool(hi_phase[0])
    data_ph, data_am, ix_map_ph, ix_map_am = _pre_filter_ph_am(
        data, sfreq, ixs, f_phase, f_amp, hi_phase=hi_phase,
        scale_amp_func=scale_amp_func, n_cycles=n_cycles)

    # Redefine indices to match the new data arrays
    ixs_new = [(ix_map_ph[i], ix_map_am[j]) for i, j in ixs]
    n_ixs_new = len(ixs_new)
    data_info = mne.create_info([str(i) for i in range(data.shape[0])], sfreq)
    data_ph = mne.io.RawArray(data_ph, data_info)
    data_am = mne.io.RawArray(data_am, data_info)
    if ev is not None:
        data_ph = _raw_to_epochs_mne(data_ph, ev, tmin, tmax)
        data_am = _raw_to_epochs_mne(data_am, ev, tmin, tmax)
    # So we know how big the PAC output will be
    if isinstance(data_ph, mne.io._BaseRaw):
        n_ep = 1
    elif concat_epochs is True:
        n_ep = len(data_ph.event_id.keys())
    else:
        n_ep = data_ph._data.shape[0]

    # Data is either Raw or Epochs
    pac_all = []
    for i_pac_func in pac_func:
        func = getattr(ppac, i_pac_func)
        pac = np.zeros([n_ep, n_ixs_new, n_times])
        pbar = ProgressBar(n_ep)
        for itime, (i_tmin, i_tmax) in enumerate(zip(tmin, tmax)):
            # Pull times of interest
            with warnings.catch_warnings():  # To suppress a depracation
                warnings.simplefilter("ignore")
                # Not sure how to do this w/o copying
                i_data_am = data_am.copy().crop(i_tmin, i_tmax)
                i_data_ph = data_ph.copy().crop(i_tmin, i_tmax)

            if concat_epochs is True:
                # Iterate through each event type and hstack
                concat_data_ph = []
                concat_data_am = []
                for i_ev in data_am.event_id.keys():
                    concat_data_ph.append(np.hstack(i_data_ph[i_ev]._data))
                    concat_data_am.append(np.hstack(i_data_am[i_ev]._data))
                i_data_am = np.vstack(concat_data_am)
                i_data_ph = np.vstack(concat_data_ph)
            else:
                # Just pull all epochs separately
                i_data_am = i_data_am._data
                i_data_ph = i_data_ph._data
            # Now make sure that inputs to the loop are ep x chan x time
            if i_data_am.ndim == 2:
                i_data_am = i_data_am[np.newaxis, ...]
                i_data_ph = i_data_ph[np.newaxis, ...]
            # Loop through epochs (or groups of epochs) and each index pair
            for iep, (ep_ph, ep_am) in enumerate(zip(i_data_ph, i_data_am)):
                for iix, (i_ix_ph, i_ix_am) in enumerate(ixs_new):
                    pac[iep, iix, itime] = func(ep_ph[i_ix_ph], ep_am[i_ix_am],
                                                f_phase, f_amp, filterfn=False)
            pbar.update_with_increment_value(1)
        pac_all.append(pac)
    if len(pac_all) == 1:
        pac_all = pac_all[0]
    if return_data:
        return pac_all, data_ph._data, data_am._data
    else:
        return pac_all


def _raw_to_epochs_mne(raw, ev, tmin, tmax):
    """Convert Raw data to Epochs w/ some time checks."""
    ev = np.atleast_1d(ev)
    if ev.ndim == 1:
        ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).T
    if ev.ndim != 2:
        raise ValueError('events have incorrect number of dimensions')
    if ev.shape[-1] != 3:
        raise ValueError('events have incorrect number of columns')
    # Convert to Epochs using the event times
    tmin_all = np.min(tmin)
    tmax_all = np.max(tmax)
    kws_epochs = dict(tmin=tmin_all, tmax=tmax_all, preload=True)

    return mne.Epochs(raw, ev, **kws_epochs)


def _pre_filter_ph_am(data, sfreq, ixs, f_ph, f_am, n_cycles=None,
                      hi_phase=False, scale_amp_func=None, kws_filt=None):
    """Filter for phase/amp only once for each channel."""
    from ..externals.pacpy.pac import _range_sanity
    from mne.filter import band_pass_filter
    from scipy.signal import hilbert

    kws_filt = dict() if kws_filt is None else kws_filt
    n_cycles = 3 if n_cycles is None else n_cycles
    _range_sanity(f_ph, f_am)
    ix_ph = np.atleast_1d(np.unique(ixs[:, 0]))
    ix_am = np.atleast_1d(np.unique(ixs[:, 1]))
    n_times = data.shape[-1]

    # Filter for lo-freq phase
    data_ph = data[ix_ph, :]
    for ii in range(data_ph.shape[0]):
        data_ph[ii] = _band_pass_pac(data_ph[ii], f_ph, sfreq, w=n_cycles)
    N = 2 ** np.ceil(np.log2(data_ph.shape[-1]))
    data_ph = np.angle(hilbert(data_ph, N=N)[..., :n_times])
    ix_map_ph = {ix: i for i, ix in enumerate(ix_ph)}

    # Filter for hi-freq amplitude
    data_am = data[ix_am, :]
    for ii in range(data_am.shape[0]):
        data_am[ii] = _band_pass_pac(data_am[ii], f_am, sfreq, w=n_cycles)
    N = 2 ** np.ceil(np.log2(data_am.shape[-1]))
    data_am = np.abs(hilbert(data_am, N=N)[..., :n_times])

    if hi_phase is True:
        # In case the PAC metric needs high-freq amplitude's phase
        for ii in range(data_am.shape[0]):
            data_am[ii] = _band_pass_pac(data_am[ii], f_ph, sfreq, w=n_cycles)
        N = 2 ** np.ceil(np.log2(data_ph.shape[-1]))
        data_am = np.angle(hilbert(data_am, N=N)[..., :n_times])
    ix_map_am = {ix: i for i, ix in enumerate(ix_am)}

    if scale_amp_func is not None:
        data_am = scale_amp_func(data_am, axis=-1)
    return data_ph, data_am, ix_map_ph, ix_map_am


def _raw_to_epochs_array(x, sfreq, ev, tmin, tmax):
    """Aux function to create epochs from a 2D array"""
    if ev.ndim != 1:
        raise ValueError('ev must be 1D')
    if ev.dtype != int:
        raise ValueError('ev must be of dtype int')

    # Check that events won't be cut off
    n_times = x.shape[-1]
    min_ix = 0 - sfreq * tmin
    max_ix = n_times - sfreq * tmax
    msk_keep = np.logical_and(ev > min_ix, ev < max_ix)

    if not all(msk_keep):
        print('Some events will be cut off!')
        ev = ev[msk_keep]

    # Pull events from the raw data
    epochs = []
    for ix in ev:
        ix_min, ix_max = [ix + int(i_tlim * sfreq)
                          for i_tlim in [tmin, tmax]]
        epochs.append(x[np.newaxis, :, ix_min:ix_max])
    epochs = np.concatenate(epochs, axis=0)
    times = np.arange(epochs.shape[-1]) / float(sfreq) + tmin
    return epochs, times, msk_keep


def phase_locked_amplitude(inst, freqs_phase, freqs_amp,
                           ix_ph, ix_amp, tmin=-.5, tmax=.5,
                           mask_times=None):
    """Calculate the average amplitude of a signal at a phase of another.

    Parameters
    ----------
    inst : instance of mne.Epochs or mne.io.Raw
        The data to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    tmin : float
        The time to include before each phase peak
    tmax : float
        The time to include after each phase peak
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        If inst is an instance of Raw, this will only include times contained
        in mask_times.

    Returns
    -------
    data_amp : np.array
        The mean amplitude values for the frequencies specified in freqs_amp,
        time-locked to peaks of the low-frequency phase.
    data_phase : np.array
        The mean low-frequency signal, phase-locked to low-frequency phase
        peaks.
    times : np.array
        The times before / after each phase peak.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_amp = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_amp, sfreq, freqs_phase, freqs_amp)

    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    band_ph = band_ph.mean(0)

    # Find peaks in the phase for time-locking
    phase_peaks, vals = peak_finder.peak_finder(angle_ph)
    ixmin, ixmax = [t * sfreq for t in [tmin, tmax]]
    # Remove peaks w/o buffer
    phase_peaks = phase_peaks[(phase_peaks > np.abs(ixmin)) *
                              (phase_peaks < len(angle_ph) - ixmax)]

    if mask_times is not None:
        # Set datapoints outside out times to nan so we can drop later
        if len(mask_times) != angle_ph.shape[-1]:
            raise ValueError('mask_times must be == in length to data')
        band_ph[..., ~mask_times] = np.nan

    data_phase, times, msk_window = _raw_to_epochs_array(
        band_ph[np.newaxis, :], sfreq, phase_peaks, tmin, tmax)
    data_amp, times, msk_window = _raw_to_epochs_array(
        amp, sfreq, phase_peaks, tmin, tmax)
    data_phase = data_phase.squeeze()
    data_amp = data_amp.squeeze()

    # Drop any peak events where there was a nan
    keep_rows = np.where(~np.isnan(data_phase).any(-1))[0]
    data_phase = data_phase[keep_rows, ...]
    data_amp = data_amp[keep_rows, ...]

    # Average across phase peak events
    data_amp = data_amp.mean(0)
    data_phase = data_phase.mean(0)
    return data_amp, data_phase, times


def phase_binned_amplitude(inst, freqs_phase, freqs_amp,
                           ix_ph, ix_amp, n_bins=20, mask_times=None):
    """Calculate amplitude of one signal in sub-ranges of phase for another.

    Parameters
    ----------
    inst : instance of mne.Epochs or mne.io.Raw
        The data to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size (2 * np.pi) / n_bins.
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        If inst is an instance of Raw, this will only include times contained
        in mask_times.

    Returns
    -------
    amp_binned : np.array, shape (n_bins,)
        The mean amplitude of freqs_amp at each phase bin
    bins_phase : np.array, shape (n_bins+1)
        The bins used in the calculation. There is one extra bin because
        bins represent the left/right edges of each bin, not the center value.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_amp = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_amp, sfreq, freqs_phase, freqs_amp)
    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    if mask_times is not None:
        # Only keep times we want
        if len(mask_times) != amp.shape[-1]:
            raise ValueError('mask_times must be == in length to data')
        angle_ph, band_ph, amp = [i[..., mask_times]
                                  for i in [angle_ph, band_ph, amp]]

    # Bin our phases and extract amplitudes based on bins
    bins_phase = np.linspace(-np.pi, np.pi, n_bins)
    bins_phase_ixs = np.digitize(angle_ph, bins_phase)
    unique_bins = np.unique(bins_phase_ixs)
    amp_binned = [np.mean(amp[:, bins_phase_ixs == i], 1)
                  for i in unique_bins]
    amp_binned = np.vstack(amp_binned).mean(1)

    return amp_binned, bins_phase


# For the viz functions
def _extract_phase_and_amp(data_phase, data_amp, sfreq, freqs_phase,
                           freqs_amp, scale=True):
    """Extract the phase and amplitude of two signals for PAC viz.
    data should be shape (n_epochs, n_times)"""
    from sklearn.preprocessing import scale

    # Morlet transform to get complex representation
    band_ph = cwt_morlet(data_phase, sfreq, freqs_phase)
    band_amp = cwt_morlet(data_amp, sfreq, freqs_amp)

    # Calculate the phase/amplitude of relevant signals across epochs
    band_ph_stacked = np.hstack(np.real(band_ph))
    angle_ph = np.hstack(np.angle(band_ph))
    amp = np.hstack(np.abs(band_amp) ** 2)

    # Scale the amplitude for viz so low freqs don't dominate highs
    if scale is True:
        amp = scale(amp, axis=1)
    return angle_ph, band_ph_stacked, amp


def _pull_data(inst, ix_ph, ix_amp, ev=None, tmin=None, tmax=None):
    """Pull data from either Base or Epochs instances"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if isinstance(inst, _BaseEpochs):
        data_ph = inst._data[:, ix_ph, :]
        data_amp = inst._data[:, ix_amp, :]
    elif isinstance(inst, _BaseRaw):
        data = inst[[ix_ph, ix_amp], :][0].squeeze()
        data_ph, data_amp = [i[np.newaxis, ...] for i in data]
    return data_ph, data_amp


def _band_pass_pac(x, f_range, fs=1000, w=3):
    """
    Band-pass filter a signal using PacPy for PAC coupling.

    This is a version of the firf function in PacPy, minux edge removal.
    It's docstring is below
    ----
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """
    from ..externals.pacpy.filt import firwin, filtfilt

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = np.float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    return x_filt
