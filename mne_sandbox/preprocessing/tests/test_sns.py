# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal
from nose.tools import assert_true

from mne import create_info, io, pick_types
from mne.utils import run_tests_if_main
from mne_sandbox.preprocessing import sensor_noise_suppression


def test_sns():
    """Test sensor noise suppression"""
    data = np.random.RandomState(0).randn(102, 5000)
    info = create_info(len(data), 1000., 'mag')
    raw = io.RawArray(data, info)
    assert_raises(TypeError, sensor_noise_suppression, 'foo', 10)
    assert_raises(ValueError, sensor_noise_suppression, raw, 'foo')
    assert_raises(ValueError, sensor_noise_suppression, raw, -1)
    raw.info['bads'] = [raw.ch_names[1]]
    assert_raises(ValueError, sensor_noise_suppression, raw, 101)
    for n_neighbors, bounds in ((2, (17, 20)),
                                (5, (11, 15),),
                                (10, (9, 12)),
                                (20, (7, 10)),
                                (50, (5, 9)),
                                (100, (5, 8)),
                                ):
        raw_sns, operator = sensor_noise_suppression(
            raw, n_neighbors=n_neighbors)
        assert_allclose(raw[1][0], raw_sns[1][0])  # bad channel not modified
        assert_allclose(operator[1], np.array([0] + [1] +
                                              [0] * (len(data) - 2)))
        assert_equal(operator[0].astype(bool).sum(), n_neighbors)
        assert_equal(operator[0, 0], 0.)
        picks = pick_types(raw.info)
        orig_power = np.linalg.norm(data[picks])
        # Test the suppression factor
        factor = orig_power / np.linalg.norm(raw_sns[picks][0])
        assert_true(bounds[0] < factor < bounds[1],
                    msg='%s: %s < %s < %s'
                    % (n_neighbors, bounds[0], factor, bounds[1]))

run_tests_if_main()
