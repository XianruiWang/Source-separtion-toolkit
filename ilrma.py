# Author: wangxianrui
# Contact: wangxianrui@mail.nwpu.edu.cn

from basetool import *


def ilrma(X_FTM, n_src=None, n_iter=20, W0=None, n_components=2):
    """
    ILRMA
    ======

    Blind Source Separation using independent low-rank matrix analysis.
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    n_components: number of NMF basis

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    -------
    [1].Kitamura D, Ono N, Sawada H, et al. Determined blind source separation unifying
    independent vector analysis and nonnegative matrix factorization.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016, 24(9): 1626-1641.

    [2].Kitamura D, Ono N, Sawada H, et al. Determined blind source separation with independent low-rank matrix analysis
    Audio source separation. Springer, Cham, 2018: 125-155.

    [3]. Kitamura D. Algorithms for independent low-rank matrix analysis. 2018.

    ======
    """

    eps = 1e-12
    X_TFM = X_FTM.transpose([1, 0, 2])
    n_frames, n_freq, n_chan = X_TFM.shape

    # default to determined case
    if n_src is None:
        n_src = X_TFM.shape[2]

    # Only supports determined case
    assert n_chan == n_src, "There should be as many microphones as sources"

    # initialize the demixing matrices
    # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
    if W0 is None:
        W_FMM = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X_TFM.dtype)
    else:
        W_FMM = W0.copy()

    # initialize the nonnegative matrixes with random values
    T = np.random.rand(n_src, n_freq, n_components).astype(np.float64) + eps
    V = np.random.rand(n_src, n_components, n_frames).astype(np.float64) + eps
    R = np.matmul(T, V)
    iR = 1 / R
    lambda_aux = np.zeros(n_src)
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))

    # Things are more efficient when the frequencies are over the first axis
    X_FMT = X_TFM.transpose([1, 2, 0]).copy()

    # Compute the demixed output
    def separate():
        Y = np.matmul(W_FMM, X_FMT)
        return Y

    # P.shape == R.shape == (n_src, n_freq, n_frames)
    YHat_FMT = separate()
    P = abs(YHat_FMT.transpose([1, 0, 2])) ** 2.0

    for epoch in range(n_iter):

        # NMF updates
        T *= np.sqrt(
            np.matmul(P * iR ** 2, V.transpose([0, 2, 1]))
            / np.matmul(iR, V.transpose([0, 2, 1]))
        )

        T[T < eps] = eps

        R = np.matmul(T, V)
        iR = 1 / R
        V *= np.sqrt(
            np.matmul(T.transpose([0, 2, 1]), P * iR ** 2)
            / np.matmul(T.transpose([0, 2, 1]), iR)
        )
        V[V < eps] = eps

        R = np.matmul(T, V)
        iR = 1 / R

        for s in range(n_src):
            ## IVA
            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            C = np.matmul((X_FMT * iR[s, :, None, :]), np.conj(X_FMT.swapaxes(1, 2)) / n_frames)

            WV = np.matmul(W_FMM, C)
            W_FMM[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            # normalize
            denom = np.matmul(
                np.matmul(W_FMM[:, None, s, :], C[:, :, :]), np.conj(W_FMM[:, s, :, None]) + eps
            )
            W_FMM[:, s, :] /= np.sqrt(denom[:, :, 0])
        P = abs(YHat_FMT.transpose([1, 0, 2])) ** 2.0

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[s, :, :]))
            W_FMM[:, s, :] *= lambda_aux[s]
            P[s, :, :] *= lambda_aux[s] ** 2
            R[s, :, :] *= lambda_aux[s] ** 2
            T[s, :, :] *= lambda_aux[s] ** 2
        # re-update first demixing vector is not really necessary, but seems to improve performance
        lambda_aux[0] = 1 / np.sqrt(np.mean(P[0, :, :]))
        W_FMM[:, 0, :] *= lambda_aux
        P[0, :, :] *= lambda_aux[0] ** 2
        R[0, :, :] *= lambda_aux[0] ** 2
        T[0, :, :] *= lambda_aux[0] ** 2
        YHat_FMT = separate()

    YHat_TFN = YHat_FMT.transpose([2, 0, 1])
    spec_F0T = X_FMT[:, 0, :]
    X_T0F = spec_F0T.T
    Z = projection_back(YHat_TFN, X_T0F)
    YHat_TFN *= np.conj(Z[None, :, :])
    YHat_FTN = YHat_TFN.transpose([1, 0, 2])
    return YHat_FTN, W_FMM
