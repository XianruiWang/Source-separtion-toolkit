# Author: wangxianrui
# Contact: wangxianrui@mail.nwpu.edu.cn

import numpy as np
from pyroomacoustics.bss.common import projection_back, sparir


def sparseILRMA(spec_FTM, S=None, n_src=2, W0=None, model="laplace",
                n_iter=30, return_flters=False, callback=None):
    print("using sparse ILRMA")
    eps = 1e-10
    n_freq, n_frame, n_chan = spec_FTM.shape

    assert (
        n_src == 2
    ), "only for two sources"

    assert (
        n_chan == 2
    ), "only for determined case"

    if model not in ["laplace", "gauss"]:
        raise ValueError("only laplace and gauss are supported")

    # choosing frequency bins, if indexes are not given, all frequency bins will be choosed
    if S is None:
        k_freq = n_freq
        S = np.arange(n_freq)
    else:
        k_freq = S.shape[0]

    # extract spectrogram
    r = np.zeros((n_src, n_frame))
    I_KMM = np.tile(np.eye(n_chan), (k_freq, 1, 1))
    spec_KTM = spec_FTM[S]
    spec_KMT = spec_KTM.transpose([0, 2, 1])
    spec_FMT = spec_FTM.transpose([0, 2, 1])
    # YHat_KMT = np.zeros(spec_KTM.shape, dtype=spec_KTM.dtype)
    # initial the demixing matrix
    if W0 is None:
        W_KMM = np.tile(np.eye(n_src,  dtype=spec_FTM.dtype), (k_freq, 1, 1))
    else:
        W_KMM = W0

    W_FMM = np.tile(np.eye(n_src, dtype=spec_FTM.dtype), (n_freq, 1, 1))

    def sparse_separate():
        """
        separate spectrogram in sparse domain
        return:
        separated_spec_KTM: separated spectrogram in sparse domain
        """
        separated_spec_KMT = W_KMM @ spec_KMT
        return separated_spec_KMT

    def cal_r_inv():
        """
        calculate r in eq. 34
        """
        if model == "laplace":
            r[:, :] = 2.0 * np.linalg.norm(YHat_KMT, axis=0)
        elif model == "gauss":
            r[:, :] = (np.linalg.norm(YHat_KMT, axis=0) ** 2) / n_freq
        r[r < eps] = eps
        r_inv = 1.0 / r
        return r_inv

    for epoch in range(n_iter):
        YHat_KMT = sparse_separate()
        r_inv_MT = cal_r_inv()
        for n in range(n_src):
            V_KMM = np.matmul(spec_KMT * r_inv_MT[None, n, None, :], np.conj(spec_KTM)) / n_frame
            WV_KMM = np.matmul(W_KMM, V_KMM)
            u = np.linalg.solve(WV_KMM, I_KMM[:, :, n])
            print(u.shape)
            u_temp1 = np.sqrt(np.matmul(np.matmul(np.conj(u[:, None, :]), V_KMM), u[:, :, None]))[:, :, 0]
            u_temp2 = (u[:, :] / u_temp1)
            print(u_temp2.shape)
            W_KMM[:, n, :] = np.conj(u_temp2)

    # LASSO regularization to reconstruct the complete transfer function
    Z = np.zeros((n_src, k_freq), dtype=W_KMM.dtype)
    G = np.zeros((n_src, n_freq, 1), dtype=W_KMM.dtype)
    hrtf = np.zeros((n_freq, n_src), dtype=W_KMM.dtype)

    W = W_KMM.transpose([0, 2, 1])
    for i in range(n_src):
        # calculate sparse relative transfer function from demixing matrix
        Z[i, :] = np.conj(-W[:, 0, i] / W[:, 1, i]).T
        print(Z.shape)

        # copy selected active frequencies in Z to sparse G
        G[i, S] = np.expand_dims(Z[i, :], axis=1)
        print(G.shape)

        # apply fast proximal algorithm to reconstruct the complete real-valued relative transfer function
        hrtf[:, i] = sparir(G[i, :], S)

        # recover relative transfer function back to the frequency domain
        hrtf[:, i] = np.fft.fft(hrtf[:, i])
        print(hrtf.shape)
        # assemble back the complete demixing matrix
        W_FMM[:, :, i] = np.conj(np.insert(hrtf[:, i, None], 1, -1, axis=1))

    W_FMM = W_FMM.transpose([0, 2, 1])
    YHat_FMT = W_FMM @ spec_FMT
    YHat_TFM = YHat_FMT.transpose([2, 0, 1])
    spec_T0F = (spec_FMT[:, 0, :]).T
    Z = projection_back(YHat_TFM, spec_T0F)
    YHat_TFM *= np.conj(Z[None, :, :])
    YHat_MFT = YHat_TFM.transpose([2, 1, 0])
    return YHat_MFT










