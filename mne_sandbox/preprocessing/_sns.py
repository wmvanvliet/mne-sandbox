# -*- coding: utf-8 -*-

import numpy as np

from mne import compute_raw_covariance
from mne.io.pick import _pick_data_channels
from mne.io import _BaseRaw
from mne.utils import logger, verbose

from ._dss import _pca


@verbose
def sensor_noise_suppression(raw, n_neighbors, reject=None, flat=None,
                             verbose=None):
    """Apply the sensor noise suppression (SNS) algorithm

    This algorithm (from [1]_) will replace the data from each channel by
    its regression on the subspace formed by the other channels.

    .. note:: Bad channels are not modified or reset by this function.

    Parameters
    ----------
    raw : Instance of Raw
        The raw data to process.
    n_neighbors : int
        Number of neighbors (based on correlation) to include in the
        projection.
    reject : dict | str | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done.
        This is only used during the covariance-fitting phase.
    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
        This is only used during the covariance-fitting phase.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of Raw
        The cleaned raw data.
    operator : ndarray, shape (n_meg_ch, n_meg_ch)
        The spatial operator that was applied to the MEG channels.

    References
    ----------
    .. [1] De Cheveigné A, Simon JZ. Sensor noise suppression. Journal of
           Neuroscience Methods 168: 195–202, 2008.
    """
    logger.info('Processing data with sensor noise suppression algorithm')
    logger.info('    Loading raw data')
    if not isinstance(raw, _BaseRaw):
        raise TypeError('raw must be an instance of Raw, got %s' % type(raw))
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be positive')
    good_picks = _pick_data_channels(raw.info, exclude='bads')
    if n_neighbors > len(good_picks) - 1:
        raise ValueError('n_neighbors must be at most len(good_picks) - 1 (%s)'
                         % (len(good_picks) - 1,))
    logger.info('    Loading data')
    raw = raw.copy()
    raw.load_data()
    picks = _pick_data_channels(raw.info, exclude=())
    bad_picks = np.setdiff1d(picks, good_picks)
    # The following few lines are equivalent to this, but require less mem use:
    # data_cov = np.cov(orig_data)
    # data_corrs = np.corrcoef(orig_data) ** 2)
    logger.info('    Computing covariance for %s good channels'
                % len(good_picks))
    data_cov = np.eye(len(picks))
    data_cov[good_picks[:, np.newaxis], good_picks] = compute_raw_covariance(
        raw, picks=good_picks, reject=reject, flat=flat, verbose=False)['data']
    del good_picks
    data_norm = np.diag(data_cov)
    data_corrs = data_cov * data_cov
    data_corrs /= data_norm
    data_corrs /= data_norm[:, np.newaxis]
    del data_norm
    data_cov *= len(raw.times)
    operator = np.zeros((len(picks), len(picks)))
    logger.info('    Assembling spatial operator')
    for ii in range(len(picks)):
        # For each channel, the set of other signals is orthogonalized by
        # applying PCA to obtain an orthogonal basis of the subspace spanned
        # by the other channels.
        if picks[ii] in bad_picks:
            operator[ii, ii] = 1.
            continue
        idx = np.argsort(data_corrs[ii])[::-1][:n_neighbors + 1].tolist()
        idx.pop(idx.index(ii))  # should be in there
        # XXX Eventually we might want to actually threshold here (with
        # rank-deficient data it could matter)
        s, v = _pca(data_cov[idx][:, idx], thresh=None)
        v *= 1. / np.sqrt(s)
        # augment with given channel
        v = np.vstack(([[1] + [0] * n_neighbors],
                       np.hstack((np.zeros((n_neighbors, 1)), v))))
        idx = np.concatenate(([ii], idx))
        corr = np.dot(np.dot(v.T, data_cov[idx][:, idx]), v)
        # The channel is projected on this basis and replaced by its projection
        operator[ii, idx[1:]] = np.dot(corr[0, 1:], v[1:, 1:].T)
    offsets = np.concatenate([np.arange(0, len(raw.times), 10000),
                              [len(raw.times)]])
    logger.info('    Applying operator')
    for start, stop in zip(offsets[:-1], offsets[1:]):
        time_sl = slice(start, stop)
        raw._data[picks, time_sl] = np.dot(operator, raw._data[picks, time_sl])
    logger.info('Done')
    return raw, operator
