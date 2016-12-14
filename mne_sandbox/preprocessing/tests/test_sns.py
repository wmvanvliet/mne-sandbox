# -*- coding: utf-8 -*-

import os.path as op

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal
from nose.tools import assert_true

from mne import create_info, io, pick_types, pick_channels
from mne.utils import run_tests_if_main
from mne.datasets import testing

from mne_sandbox.preprocessing import SensorNoiseSuppression

data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_sns():
    """Test sensor noise suppression"""
    # artificial (IID) data
    data = np.random.RandomState(0).randn(102, 5000)
    info = create_info(len(data), 1000., 'mag')
    raw = io.RawArray(data, info)
    assert_raises(ValueError, SensorNoiseSuppression, 'foo')
    assert_raises(TypeError, SensorNoiseSuppression(10).fit, 'foo')
    assert_raises(ValueError, SensorNoiseSuppression, -1)
    raw.info['bads'] = [raw.ch_names[1]]
    assert_raises(ValueError, SensorNoiseSuppression(101).fit, raw)
    for n_neighbors, bounds in ((2, (17, 20)),
                                (5, (11, 15),),
                                (10, (9, 12)),
                                (20, (7, 10)),
                                (50, (5, 9)),
                                (100, (5, 8)),
                                ):
        sns = SensorNoiseSuppression(n_neighbors)
        sns.fit(raw)
        raw_sns = sns.apply(raw.copy())
        operator = sns.operator
        assert_allclose(raw[1][0], raw_sns[1][0])  # bad channel not modified
        assert_allclose(operator[1], np.array([0] + [1] +
                                              [0] * (len(data) - 2)))
        assert_equal(operator[0].astype(bool).sum(), n_neighbors)
        assert_equal(operator[0, 0], 0.)
        picks = pick_types(raw.info)
        orig_power = np.linalg.norm(raw[picks][0])
        # Test the suppression factor
        factor = orig_power / np.linalg.norm(raw_sns[picks][0])
        assert_true(bounds[0] < factor < bounds[1],
                    msg='%s: %s < %s < %s'
                    % (n_neighbors, bounds[0], factor, bounds[1]))
    # degenerate conditions
    assert_raises(TypeError, sns.apply, 'foo')
    sub_raw = raw.copy().pick_channels(raw.ch_names[:-1])
    assert_raises(RuntimeError, sns.apply, sub_raw)  # not all orig chs
    sub_sns = SensorNoiseSuppression(8)
    sub_sns.fit(sub_raw)
    assert_raises(RuntimeError, sub_sns.apply, raw)  # not all new chs
    # sample data
    raw = io.read_raw_fif(raw_fname)
    n_neighbors = 8
    sns = SensorNoiseSuppression(n_neighbors=n_neighbors)
    sns.fit(raw)
    raw_sns = sns.apply(raw.copy().load_data())
    operator = sns.operator
    # bad channels not modified
    assert_equal(len(raw.info['bads']), 2)
    for pick in pick_channels(raw.ch_names, raw.info['bads']):
        expected = np.zeros(operator.shape[0])
        sub_pick = sns._used_chs.index(raw.ch_names[pick])
        expected[sub_pick] = 1.
        assert_allclose(operator[sub_pick], expected)
        assert_allclose(raw[pick][0], raw_sns[pick][0])
    assert_equal(operator[0].astype(bool).sum(), n_neighbors)
    assert_equal(operator[0, 0], 0.)
    picks = pick_types(raw.info)
    orig_power = np.linalg.norm(raw[picks][0])
    # Test the suppression factor
    factor = orig_power / np.linalg.norm(raw_sns[picks][0])
    bounds = (1.3, 1.7)
    assert_true(bounds[0] < factor < bounds[1],
                msg='%s: %s < %s < %s'
                % (n_neighbors, bounds[0], factor, bounds[1]))
    # degenerate conditions
    assert_raises(RuntimeError, sns.apply, raw)  # not preloaded


run_tests_if_main()
