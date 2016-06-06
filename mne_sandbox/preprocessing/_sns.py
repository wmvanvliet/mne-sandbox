# -*- coding: utf-8 -*-

import numpy as np

from mne import compute_raw_covariance
from mne.io.pick import _pick_data_channels, pick_channels
from mne.io import _BaseRaw
from mne.utils import logger, verbose

from ._dss import _pca


class SensorNoiseSuppression(object):
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

    @verbose
    def __init__(self, raw, n_neighbors, reject=None, flat=None, verbose=None):
        logger.info('Processing data with sensor noise suppression algorithm')
        logger.info('    Loading raw data')
        if not isinstance(raw, _BaseRaw):
            raise TypeError('raw must be an instance of Raw, got %s'
                            % type(raw))
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be positive')
        good_picks = _pick_data_channels(raw.info, exclude='bads')
        if n_neighbors > len(good_picks) - 1:
            raise ValueError('n_neighbors must be at most len(good_picks) '
                             '- 1 (%s)' % (len(good_picks) - 1,))
        logger.info('    Loading data')
        raw = raw.copy()
        picks = _pick_data_channels(raw.info, exclude=())
        # The following lines are equivalent to this, but require less mem use:
        # data_cov = np.cov(orig_data)
        # data_corrs = np.corrcoef(orig_data) ** 2
        logger.info('    Computing covariance for %s good channels'
                    % len(good_picks))
        data_cov = np.eye(len(picks))
        good_cov = compute_raw_covariance(
            raw, picks=good_picks, reject=reject, flat=flat,
            verbose=False)['data']
        # re-index this
        good_picks = np.searchsorted(picks, good_picks)
        bad_picks = np.setdiff1d(np.arange(len(picks)), good_picks)
        data_cov[np.ix_(good_picks, good_picks)] = good_cov
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
            # applying PCA to obtain an orthogonal basis of the subspace
            # spanned by the other channels.
            if ii in bad_picks:
                operator[ii, ii] = 1.
                continue
            idx = np.argsort(data_corrs[ii])[::-1][:n_neighbors + 1].tolist()
            idx.pop(idx.index(ii))  # should be in there
            # XXX Eventually we might want to actually threshold here (with
            # rank-deficient data it could matter)
            eigval, eigvec = _pca(data_cov[np.ix_(idx, idx)], thresh=None)
            eigvec *= 1. / np.sqrt(eigval)
            del eigval
            # augment with given channel
            eigvec = np.vstack(([[1] + [0] * n_neighbors],
                               np.hstack((np.zeros((n_neighbors, 1)),
                                          eigvec))))
            idx = np.concatenate(([ii], idx))
            corr = np.dot(np.dot(eigvec.T, data_cov[np.ix_(idx, idx)]), eigvec)
            # The channel is projected on this basis and replaced by its
            # projection
            operator[ii, idx[1:]] = np.dot(corr[0, 1:], eigvec[1:, 1:].T)
        logger.info('Done')
        self._operator = operator
        self._used_chs = [raw.ch_names[pick] for pick in picks]

    @property
    def operator(self):
        """The operator matrix"""
        return self._operator.copy()

    def apply(self, inst):
        """Apply the operator

        Parameters
        ----------
        inst : instance of Raw
            The data on which to apply the operator.

        Returns
        -------
        inst : instance of Raw
            The input instance.
        """
        if isinstance(inst, _BaseRaw):
            if not inst.preload:
                raise RuntimeError('raw data must be loaded, use '
                                   'raw.load_data()')
            offsets = np.concatenate([np.arange(0, len(inst.times), 10000),
                                      [len(inst.times)]])
            info = inst.info
            picks = pick_channels(info['ch_names'], self._used_chs)
            data_picks = [info['ch_names'][pick]
                          for pick in _pick_data_channels(info, exclude=())]
            missing = set(data_picks) - set(self._used_chs)
            if len(missing) > 0:
                raise RuntimeError('Not all data channels of inst were used '
                                   'to construct the operator: %s'
                                   % sorted(missing))
            missing = set(self._used_chs) - set(info['ch_names'][pick]
                                                for pick in picks)
            if len(missing) > 0:
                raise RuntimeError('Not all channels originally used to '
                                   'construct the operator are present: %s'
                                   % sorted(missing))
            for start, stop in zip(offsets[:-1], offsets[1:]):
                time_sl = slice(start, stop)
                inst._data[picks, time_sl] = np.dot(self._operator,
                                                    inst._data[picks, time_sl])
        else:
            # XXX Eventually this could support Evoked and Epochs, too
            raise TypeError('Only Raw instances are currently supported, got '
                            '%s' % type(inst))
        return inst
