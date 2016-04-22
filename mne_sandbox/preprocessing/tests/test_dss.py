# -*- coding: utf-8 -*-

import numpy as np
from mne_sandbox.preprocessing import dss
from numpy.testing import assert_allclose, assert_raises


def test_dss_args():
    """Test DSS error handling"""
    data1 = list()
    data2 = np.arange(6).reshape(2, 3)
    data3 = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    assert_raises(TypeError, dss, data1)
    assert_raises(ValueError, dss, data2)  # not enough dimensions
    assert_raises(ValueError, dss, data3)  # too many dimensions


def test_dss():
    """Test DSS computations"""

    def rms(data):
        return np.sqrt(np.mean(data ** 2, axis=-1, keepdims=True))

    rand = np.random.RandomState(123)
    # parameters
    n_trials, n_times, n_channels, noise_dims, snr = [200, 1000, 16, 10, 0.1]
    # 1 Hz sine wave with silence before & after
    pad = np.zeros(n_times // 3)
    signal_nsamps = n_times - 2 * pad.size
    sine = np.sin(2 * np.pi * np.arange(signal_nsamps) / float(signal_nsamps))
    sine = np.r_[pad, sine, pad]
    signal = rand.randn(n_channels, 1) * sine[np.newaxis, :]
    # noise
    noise = np.einsum('hjk,ik->hij',
                      rand.randn(n_trials, n_times, noise_dims),
                      rand.randn(n_channels, noise_dims))
    # signal plus noise, in a reasonable range for EEG
    data = 4e-6 * (noise / rms(noise) + snr * signal / rms(signal))
    # perform DSS
    dss_mat = dss(data, data_thresh=1e-3, bias_thresh=1e-3, return_data=False)
    dss_trial1_comp1 = np.dot(dss_mat, data[0])[0]
    dss_trial1_comp1 = dss_trial1_comp1 / np.abs(dss_trial1_comp1).max()
    try:
        assert_allclose(dss_trial1_comp1, sine, rtol=0, atol=0.2)
    except AssertionError:
        # maybe just 180 degree phase difference
        assert_allclose(-1 * dss_trial1_comp1, sine, rtol=0, atol=0.2)
