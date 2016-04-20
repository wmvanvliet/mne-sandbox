# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:47:10 2016
@author: drmccloy
"""

import numpy as np


def dss(epochs, data_n_keep=None, bias_n_keep=None, data_thresh=None,
        bias_thresh=None):
    """Preprocess physiological data with denoising source separation (DSS)

    Implementation follows the procedure described in S\"arel\"a & Valpola [1]
    and de Cheveign\'e & Simon [2].

    Parameters
    ----------
    data : instance of Epochs
        Data to be denoised.
    data_n_keep, bias_n_keep : int
        Number of components to keep during PCA decomposition of the data and
        bias function. ``None`` (the default) means keep all suprathreshold
        components (see ``data_thresh``).
   data_thresh, bias_thresh : float
       Threshold (relative to the largest component) below which components
       will be discarded during decomposition of the data and bias function.
       For example, if the largest eigenvalue is 3 and ``thresh=0.1``,
       components with eigenvalues below 0.3 will be discarded.

    References
    ----------
    [1] S\"arel\"a, Jaakko, and Valpola, Harri (2005). Denoising source
    separation.  Journal of Machine Learning Research 6: 233–72.

    [2] de Cheveign\'e, Alain, and Simon, Jonathan Z. (2008). Denoising based
    on spatial filtering. Journal of Neuroscience Methods, 171(2): 331-339.
    """
    # TODO: allow Raw as well as Epochs?
    # TODO: implement mean-subtraction? optional?
    # TODO: apply picks before DSS?
    evoked = epochs.average()
    bias_cov = np.dot(evoked, evoked.T)
    data_cov = np.sum([np.dot(trial, trial.T) for trial in epochs], axis=0)
    dss_mat = _dss(data_cov, bias_cov, data_n_keep, bias_n_keep, data_thresh,
                   bias_thresh)
    data = epochs.get_data()
    n_trials, n_chans, n_times = data.shape
    data = data.transpose((1, 2, 0)).reshape(n_chans, n_times * n_trials)
    epochs._data = np.dot(dss_mat.T, data).reshape(dss_mat.shape[1], n_times,
                                                   n_trials).transpose(2, 0, 1)
    return epochs


def _dss(data_cov, bias_cov, data_n_keep=None, bias_n_keep=None,
         data_thresh=None, bias_thresh=None):
    """Preprocess physiological data with denoising source separation (DSS)

    Acts on covariance matrices; allows specification of arbitrary bias
    functions (as compared to the public ``dss`` function, which forces the
    bias function to be the evoked response).
    """
    data_eigval, data_eigvec = _pca(data_cov, data_n_keep, data_thresh)
    W = np.diag(np.sqrt(1 / data_eigval))  # whitening matrix
    # bias covariance projected into whitened PCA space of data channels
    bias_white = W.T.dot(data_eigvec.T).dot(bias_cov).dot(data_eigvec).dot(W)
    # proj. matrix from whitened data space to a space maximizing bias fxn
    bias_eigval, bias_eigvec = _pca(bias_white, bias_n_keep, bias_thresh)
    # proj. matrix from data to bias-maximizing space (DSS space)
    dss_mat = data_eigvec.dot(W).dot(bias_eigvec)
    # matrix to normalize DSS dimensions
    N = np.diag(np.sqrt(1 / np.diag(dss_mat.T.dot(data_cov).dot(dss_mat))))
    return dss_mat.dot(N)


def _pca(cov, n_keep=None, thresh=None):
    """Perform PCA decomposition

    Parameters
    ----------
    cov : array-like
        Covariance matrix
    n_keep : int | None
        Number of components to retain after decomposition.
    thresh : float
        Threshold (relative to the largest component) below which components
        will be discarded.

    Returns
    -------
    eigval : array
        1-dimensional array of eigenvalues.
    eigvec : array
        2-dimensional array of eigenvectors.
    """
    eigval, eigvec = np.linalg.eig(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if n_keep is not None:
        eigval = eigval[:n_keep]
        eigvec = eigvec[:, :n_keep]
    if thresh is not None:
        suprathresh = np.where(eigval / eigval.max() >= thresh)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec
