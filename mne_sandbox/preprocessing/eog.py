# encoding: utf-8
"""
Functions concerning regressing out EOG signals.
License: BSD (3-clause)

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from numpy.linalg import lstsq
from mne import pick_types


def eog_regression(raw, blink_epochs, saccade_epochs=None, reog=None,
                   eog_channels=None, picks=None, copy=False):
    """Remove EOG signals from the EEG channels by regression.

    It employes the RAAA (recommended aligned-artifact average) procedure
    described by Croft & Barry [1]_.

    Parameters
    ----------
    raw : Instance of Raw
        The raw data on which the EOG correction produce should be performed.
    blink_epochs : Instance of Epochs
        Epochs cut around blink events. We recommend cutting a window from -0.5
        to 0.5 seconds relative to the onset of the blink.
    saccade_epochs : Instance of Epochs | None
        Epochs cut around saccade events. We recommend cutting a window from -1
        to 1.5 seconds relative to the onset of the saccades, and providing
        separate events for "up", "down", "left" and "right" saccades.
        By default, no saccade information is taken into account.
    reog : str | None
        The name of the rEOG channel, if present. If an rEOG channel is
        available as well as saccade data, the accuracy of the estimation of
        the weights can be improved. By default, no rEOG channel is assumed to
        be present.
    eog_channels : str | list of str | None
        The names of the EOG channels to use. By default, all EOG channels are
        used.
    picks : list of int | None
        Indices of the channels in the Raw instance for which to apply the EOG
        correction procedure. By default, the correction is applied to EEG
        channels only.
    copy : bool
        If True, a copy of the Raw instance will be made before applying the
        EOG correction procedure. Defaults to False, which will perform the
        operation in-place.

    See Also
    --------
    calc_reog

    References
    ----------
    .. [1] Croft, R. J., & Barry, R. J. (2000). Removal of ocular artifact from
       the EEG: a review. Clinical Neurophysiology, 30(1), 5-19.
       http://doi.org/10.1016/S0987-7053(00)00055-1
    """
    if not raw.preload:
        raise RuntimeError('Raw data needs to be preloaded.')

    # Handle defaults for EOG channels parameter
    if eog_channels is None:
        eog_picks = pick_types(raw.info, meg=False, ref_meg=False, eog=True)
        eog_channels = [raw.ch_names[ch] for ch in eog_picks]
    elif isinstance(eog_channels, str):
        eog_channels = [eog_channels]

    # Make sure the REOG channel is part of the EOG channel list
    if reog is not None:
        if reog not in eog_channels:
            eog_channels += [reog]

    # Default picks
    if picks is None:
        picks = pick_types(raw.info, meg=False, ref_meg=False, eeg=True)

    if copy:
        raw = raw.copy()

    # Compute channel indices for the EOG channels
    raw_eog_ind = [raw.ch_names.index(ch) for ch in eog_channels]
    ev_eog_ind = [blink_epochs.ch_names.index(ch) for ch in eog_channels]

    blink_evoked = [
        blink_epochs[cl].average(range(blink_epochs.info['nchan']))
        for cl in blink_epochs.event_id.keys()
    ]
    blink_data = np.hstack([ev.data for ev in blink_evoked])

    if saccade_epochs is None:
        # Calculate EOG weights
        v = np.vstack((
            np.ones(blink_data.shape[1]),
            blink_data[ev_eog_ind]
        )).T
        weights = lstsq(v, blink_data.T)[0][1:]
    else:
        saccade_evoked = [
            saccade_epochs[cl].average(range(saccade_epochs.info['nchan']))
            for cl in saccade_epochs.event_id.keys()
        ]
        saccade_data = np.hstack([ev.data for ev in saccade_evoked])

        if reog is None:
            # If no rEOG data is present, just concatenate the saccade data
            # to the blink data and treat it as one
            blink_sac_data = np.c_[blink_data, saccade_data]
            v = np.vstack((
                np.ones(blink_sac_data.shape[1]),
                blink_sac_data[np.r_[ev_eog_ind]]
            )).T
            weights = lstsq(v, blink_sac_data.T)[0][1:]
        else:
            # If rEOG data is present, use the saccade data to compute the
            # weights for all non-rEOG channels. The blink data will be used
            # for the rEOG channel weight.

            # Isolate the rEOG channel from the other EOG channels
            raw_reog_ind = raw.ch_names.index(reog)
            raw_non_reog_ind = list(raw_eog_ind)
            raw_non_reog_ind.remove(raw_reog_ind)
            ev_reog_ind = blink_epochs.ch_names.index(reog)
            ev_non_reog_ind = list(ev_eog_ind)
            ev_non_reog_ind.remove(ev_reog_ind)

            # Compute non-rEOG weights on the saccade data
            v1 = np.vstack((
                np.ones(saccade_data.shape[1]),
                saccade_data[ev_non_reog_ind, :],
            )).T
            weights_sac = lstsq(v1, saccade_data.T)[0][1:]

            # Remove saccades from blink data
            blink_data -= weights_sac.T.dot(blink_data[ev_non_reog_ind, :])

            # Compute rEOG weights on the blink data
            v2 = np.vstack((
                np.ones(blink_data.shape[1]),
                blink_data[ev_reog_ind, :]
            )).T
            weights_blink = lstsq(v2, blink_data.T)[0][[1]]

            # Remove saccades from rEOG channel in raw data
            raw._data[raw_reog_ind, :] -= np.dot(
                weights_sac[:, ev_reog_ind].T, raw._data[raw_non_reog_ind, :])

            # Compile the EOG weights and make sure to put them in the right
            # order.
            ind = list(range(len(eog_channels)))
            REOG_ind = eog_channels.index('REOG')
            del ind[REOG_ind]
            ind.append(REOG_ind)
            weights = np.vstack((weights_sac, weights_blink))[ind]

    # Create a mapping between the picked channels of the raw instance and the
    # EOG weights
    weight_names = blink_epochs.ch_names
    weight_ch_ind = [weight_names.index(raw.ch_names[ch]) for ch in picks]

    # Remove EOG from raw channels
    raw._data[picks, :] -= np.dot(weights[:, weight_ch_ind].T,
                                  raw._data[raw_eog_ind, :])

    return raw, weights[:, weight_ch_ind]


def calc_reog(raw, ref, eog=None, name='rEOG'):
    """Compute a virtual radial-EOG channel and append it to the data.

    The radial EOG channel is a virtual channel constructed by taking the mean
    of all present EOG channels and referencing it to the EEG reference. It
    aims to capture radial eye movements which occur during blinking [1]_.

    The virtual channel is appended to the raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to which the radial EOG channel should be appended.
    ref : ndarray, shape (n_samples,)
        The EEG reference signal. This is for example returned by the
        :func:`mne.io.set_eeg_reference` function.
    eog : list of str | None
        The EOG channels to use in the computation of the radial EOG. If
        ``None`` (the default), all EOG channels present in the data are used.
    name : str
        A name for the new radial EOG channel. Defaults to 'rEOG'.

    Returns
    -------
    raw : instance of Raw
        The raw data with the radial EOG channel appended. Data is modified
        in-place.

    See Also
    --------
    eog_regression

    References
    ----------
    .. [1] Croft, R. J., & Barry, R. J. (2000). Removal of ocular artifact from
       the EEG: a review. Clinical Neurophysiology, 30(1), 5-19.
       http://doi.org/10.1016/S0987-7053(00)00055-1
    """
    if not raw.preload:
        raise RuntimeError('Raw data needs to be preloaded.')

    if eog is None:
        eog = pick_types(raw.info, meg=False, ref_meg=False, eog=True)
        if len(eog) == 0:
           raise RuntimeError('No EOG channels present. Cannot calculate rEOG')
    else:
        eog = [raw.ch_names.index(ch) for ch in eog]

    reog = raw._data[eog, :].mean(axis=0) - ref

    raw._data = np.r_[raw._data, reog[np.newaxis, :]]

    ch_info = raw.info['chs'][eog[0]].copy()
    ch_info['ch_name'] = name
    ch_info['eeg_loc'] = np.zeros((3,2))
    ch_info['loc'] = np.zeros(12)
    ch_info['logno'] = raw.info['nchan'] + 1
    raw.info['chs'].append(ch_info)
    raw.info['nchan'] += 1
    raw.info['ch_names'] = [ch['ch_name'] for ch in raw.info['chs']]

    if getattr(raw, '_cals', None) is not None:
        raw._cals = np.r_[raw._cals, raw._cals[eog[0]]]

    return raw
