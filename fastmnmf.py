# Author： wangxianrui
# Contact: wangxianrui@mail.nwpu.edu.cn

import numpy as np


def fastmnmf(X_FTM, n_src=2, n_iter=50, n_components=4, mic_index=0, seed=0, Q_0=None):
    """
    FastMNMF
    ======

    Blind Source Separation using fast multichannel non-negative matrix factorization
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    n_components: number of NMF basis
    mic_index: recover signal from microphone mix_index
    seed: random seed
    Q_0: initial value of Q

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    ----------
    [1] Ito N, Nakatani T. FastMNMF: Joint diagonalization based accelerated algorithms for
    multichannel nonnegative matrix factorization, in Proc. ICASSP, pp. 371-375,  2019

    [2] Sekiguchi K, Bando Y, Nugraha A A, et al. Fast multichannel nonnegative matrix factorization
    with directivity-aware jointly-diagonalizable spatial covariance matrices for blind source separation,
    IEEE/ACM TASLP, vol. 28, pp. 2610-2625, 2020.

    [3] Ito N, Ikeshita R, Sawada H, et al. A Joint Diagonalization Based Efficient Approach to Underdetermined
    Blind Audio Source Separation Using the Multichannel Wiener Filter.
    IEEE/ACM TASLP, 2021.

    ======
    """

    ## initialize parameters
    eps = 1e-8
    # random seed
    np.random.seed(seed)
    # load spectrogram
    n_freq, n_frame, n_mic = X_FTM.shape
    # construct observed spatial corvariance matrices(SCMs)
    # F*T*M*1 @ F*T*1*M = F*T*M*M
    X_FTMM = X_FTM[..., None] @ X_FTM[..., None, :].conj()
    # initial PSD eq. 3
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(np.float64)
    H_NKT = np.random.rand(n_src, n_components, n_frame).astype(np.float64)
    # initial diagonal covariance matrix and diagonalizer, eq. 41
    if Q_0 is None:
        Q_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1]).astype(np.complex128)
    else:
        Q_FMM = Q_0.copy()
    G_NFM = np.ones([n_src, n_freq, n_mic], dtype=np.float64) * 1e-2
    for m in range(n_mic):
        G_NFM[m % n_src, :, m] = 1

    # normalize
    # eq. 26
    mu_FM = np.sum(Q_FMM * Q_FMM.conj(), axis=2).real
    Q_FMM /= np.sqrt(mu_FM[:, :, None])
    G_NFM /= mu_FM[None, :, :]
    # eq. 27
    phi_NF = G_NFM.sum(axis=2).real
    G_NFM /= phi_NF[..., None]
    W_NFK *= phi_NF[..., None]
    # eq.28
    kappa_NK = W_NFK.sum(axis=1)
    W_NFK /= kappa_NK[:, None]
    H_NKT *= kappa_NK[:, :, None]
    lambda_NFT = W_NFK @ H_NKT + eps

    # transform signal into diagonal domain
    # F*1*M*M @ F*T*M*1 = F*T*M*1
    Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
    Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

    # separate signal with estimated parameters, eq.19
    def separate():
        # F*1*M*M @ F*T*M*1 = F*T*M*1
        Qx_FTM = (Q_FMM[:, None] @ X_FTM[..., None])[..., 0]
        Q_inv_FMM = np.linalg.inv(Q_FMM)
        tmp_NFTM = lambda_NFT[..., None] * G_NFM[:, :, None]
        for n in range(n_src):
            tmp = (
                np.matmul(
                    Q_inv_FMM[:, None],
                    (Qx_FTM * (tmp_NFTM[n] / tmp_NFTM.sum(axis=0)))[..., None],
                )
            )[:, :, mic_index, 0]
            if n == 0:
                separated_spec_MFT = np.zeros([n_src, n_freq, n_frame]).astype(np.complex128)
            separated_spec_MFT[n] = tmp
        separated_spec_FTM = separated_spec_MFT.transpose([1, 2, 0])
        return separated_spec_FTM

    # update
    for epoch in range(n_iter):
        # update NMF 20-21
        # sum_m (N*F*1*M )*(1*F*T*M)
        tmp_yb1_NFT = (G_NFM[:, :, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=3)
        tmp_yb2_NFT = (G_NFM[:, :, None] / Y_FTM[None]).sum(axis=3)
        # sum_t (N*1*K*T)*(N*F*1*T)
        a_1_NFK = (H_NKT[:, None] * tmp_yb1_NFT[:, :, None]).sum(axis=3)
        b_1_NFK = (H_NKT[:, None] * tmp_yb2_NFT[:, :, None]).sum(axis=3)
        # sum_f (N*F*K*1)*(N*F*1*T)
        a_2_NKT = (W_NFK[..., None] * tmp_yb1_NFT[:, :, None]).sum(axis=1)
        b_2_NKT = (W_NFK[..., None] * tmp_yb2_NFT[:, :, None]).sum(axis=1)
        W_NFK *= np.sqrt(a_1_NFK / b_1_NFK)
        H_NKT *= np.sqrt(a_2_NKT / b_2_NKT)

        # update power spectrogram
        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

        # update diagonal matrix
        # sum_t (N*F*T*1)*(1*F*T*M)
        a_3_NFM = (lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=2)
        b_3_NFM = (lambda_NFT[..., None] / Y_FTM[None]).sum(axis=2)
        G_NFM *= np.sqrt(a_3_NFM / b_3_NFM)
        G_NFM += eps
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

        # update diagonalizer
        for m in range(n_mic):
            V_FMM = (X_FTMM / Y_FTM[:, :, m, None, None]).mean(axis=1)
            tmp_FM = np.linalg.inv(Q_FMM @ V_FMM)[:, :, m]
            Q_FMM[:, m] = (tmp_FM / np.sqrt(((tmp_FM[:, None].conj() @ V_FMM) @ tmp_FM[..., None])[:, :, 0])).conj()

        # normalize
        # eq. 26
        mu_FM = np.sum(Q_FMM * Q_FMM.conj(), axis=2).real
        Q_FMM /= np.sqrt(mu_FM[:, :, None])
        G_NFM /= mu_FM[None, :, :]
        # eq. 27
        phi_NF = G_NFM.sum(axis=2).real
        G_NFM /= phi_NF[..., None]
        W_NFK *= phi_NF[..., None]
        # eq.28
        kappa_NK = W_NFK.sum(axis=1)
        W_NFK /= kappa_NK[:, None]
        H_NKT *= kappa_NK[:, :, None]
        lambda_NFT = W_NFK @ H_NKT + eps
        # reset variables
        Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

    return separate()


def fastmnmf2(X_FTM, n_src=2, n_iter=50, n_components=4, mic_index=0, seed=0, Q_0=None):
    """
    FastMNMF2
    ======

    Blind Source Separation using fast multichannel non-negative matrix factorization
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    n_components: number of NMF basis
    mic_index: recover signal from microphone mix_index
    seed: random seed
    Q_0: initial value of Q

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    ----------
    [1] Ito N, Nakatani T. FastMNMF: Joint diagonalization based accelerated algorithms for
    multichannel nonnegative matrix factorization, in Proc. ICASSP, pp. 371-375,  2019

    [2] Sekiguchi K, Bando Y, Nugraha A A, et al. Fast multichannel nonnegative matrix factorization
    with directivity-aware jointly-diagonalizable spatial covariance matrices for blind source separation,
    IEEE/ACM TASLP, vol. 28, pp. 2610-2625, 2020.

    [3] Ito N, Ikeshita R, Sawada H, et al. A Joint Diagonalization Based Efficient Approach to Underdetermined
    Blind Audio Source Separation Using the Multichannel Wiener Filter.
    IEEE/ACM TASLP, 2021.

    ======
    """
    ## initialize parameters

    eps = 1e-8

    # random seed
    np.random.seed(seed)

    # load spectrogram
    n_freq, n_frame, n_mic = X_FTM.shape
    # construct observed spatial corvariance matrices(SCMs)
    # F*T*M*1 @ F*T*1*M = F*T*M*M
    X_FTMM = X_FTM[..., None] @ X_FTM[..., None, :].conj()
    # initial PSD eq. 3
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(np.float64)
    H_NKT = np.random.rand(n_src, n_components, n_frame).astype(np.float64)

    # initial diagonal covariance matrix and diagonalizer, eq. 41
    if Q_0 == None:
        Q_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1]).astype(np.complex128)
    else:
        Q_FMM = Q_0.copy()
    G_NM = np.ones([n_src, n_mic], dtype=np.float64) * 1e-2
    mu_F = np.zeros(n_freq, dtype=np.float64)
    for m in range(n_mic):
        G_NM[m % n_src, m] = 1

    # normalize
    # eq. 37
    for f in range(n_freq):
        mu_F[f] = (np.trace(Q_FMM[f]*(Q_FMM[f].T.conj())) / n_mic).real
    Q_FMM /= np.sqrt(mu_F[:, None, None])
    W_NFK /= mu_F[None, :, None]

    # eq. 38
    phi_N = G_NM.sum(axis=1)
    G_NM /= phi_N[:, None]
    W_NFK *= phi_N[:, None, None]
    lambda_NFT = W_NFK @ H_NKT + eps

    # transform signal into diagonal domain
    # F*1*M*M @ F*T*M*1 = F*T*M*1
    Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
    Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

    # separate signal with estimated parameters, eq.19
    def separate():
        # F*1*M*M @ F*T*M*1 = F*T*M*1
        Qx_FTM = (Q_FMM[:, None] @ X_FTM[..., None])[..., 0]
        Q_inv_FMM = np.linalg.inv(Q_FMM)
        tmp_NFTM = lambda_NFT[..., None] * G_NM[:, None, None]
        for n in range(n_src):
            tmp = (
                np.matmul(
                    Q_inv_FMM[:, None],
                    (Qx_FTM * (tmp_NFTM[n] / tmp_NFTM.sum(axis=0)))[..., None],
                )
            )[:, :, mic_index, 0]
            if n == 0:
                separated_spec_MFT = np.zeros([n_src, n_freq, n_frame]).astype(np.complex128)
            separated_spec_MFT[n] = tmp
        separated_spec_FTM = separated_spec_MFT.transpose([1, 2, 0])
        return separated_spec_FTM

    # update
    for epoch in range(n_iter):
        # update NMF 20-21
        # sum_m (N*F*1*M )*(1*F*T*M)
        tmp_yb1_NFT = (G_NM[:, None, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=3)
        tmp_yb2_NFT = (G_NM[:, None, None] / Y_FTM[None]).sum(axis=3)
        # sum_t (N*1*K*T)*(N*F*1*T)
        a_1_NFK = (H_NKT[:, None] * tmp_yb1_NFT[:, :, None]).sum(axis=3)
        b_1_NFK = (H_NKT[:, None] * tmp_yb2_NFT[:, :, None]).sum(axis=3)
        # sum_f (N*F*K*1)*(N*F*1*T)
        a_2_NKT = (W_NFK[..., None] * tmp_yb1_NFT[:, :, None]).sum(axis=1)
        b_2_NKT = (W_NFK[..., None] * tmp_yb2_NFT[:, :, None]).sum(axis=1)
        W_NFK *= np.sqrt(a_1_NFK / b_1_NFK)
        H_NKT *= np.sqrt(a_2_NKT / b_2_NKT)

        # update power spectrogram
        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

        # update diagonal matrix
        # sum_t (N*F*T*1)*(1*F*T*M)
        a_3_NM = (lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=(1, 2))
        b_3_NM = (lambda_NFT[..., None] / Y_FTM[None]).sum(axis=(1, 2))
        G_NM *= np.sqrt(a_3_NM / b_3_NM)
        G_NM += eps
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

        # update diagonalizer
        for m in range(n_mic):
            V_FMM = (X_FTMM / Y_FTM[:, :, m, None, None]).mean(axis=1)
            tmp_FM = np.linalg.inv(Q_FMM @ V_FMM)[:, :, m]
            Q_FMM[:, m] = (tmp_FM / np.sqrt(((tmp_FM[:, None].conj() @ V_FMM) @ tmp_FM[..., None])[:, :, 0])).conj()

        # normalize
        # eq. 37
        for f in range(n_freq):
            mu_F[f] = (np.trace(Q_FMM[f]*(Q_FMM[f].T.conj())) / n_mic).real
        Q_FMM /= np.sqrt(mu_F[:, None, None])
        W_NFK /= mu_F[None, :, None]

        # eq.38
        phi_N = G_NM.sum(axis=1)
        G_NM /= phi_N[:, None]
        W_NFK *= phi_N[:, None, None]
        lambda_NFT = W_NFK @ H_NKT + eps
        # reset variables
        Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

    return separate()


def fastmnmf_VCD(X_FTM, n_src=2, n_iter=50, n_components=4, mic_index=0, seed=0, Q_prior_FMM=None):
    """
    FastMNMF with Vector-wise Coordinate Descent(VCD) Regularization
    ======

    Blind Source Separation using fast multichannel non-negative matrix factorization with
    Vector-wise Coordinate Descent(VCD) Regularization
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    n_components: number of NMF basis
    mic_index: recover signal from microphone mix_index
    seed: random seed
    Q_prior_FMM: prior information of Q

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    ----------
    [1] Ito N, Nakatani T. FastMNMF: Joint diagonalization based accelerated algorithms for
    multichannel nonnegative matrix factorization, in Proc. ICASSP, pp. 371-375,  2019

    [2] Sekiguchi K, Bando Y, Nugraha A A, et al. Fast multichannel nonnegative matrix factorization
    with directivity-aware jointly-diagonalizable spatial covariance matrices for blind source separation,
    IEEE/ACM TASLP, vol. 28, pp. 2610-2625, 2020.

    [3] Ito N, Ikeshita R, Sawada H, et al. A Joint Diagonalization Based Efficient Approach to Underdetermined
    Blind Audio Source Separation Using the Multichannel Wiener Filter.
    IEEE/ACM TASLP, 2021.

    [4] Kamo K, Kubo Y, Takamune N, et al. Regularized fast multichannel nonnegative matrix factorization with
    ILRMA-based prior distribution of joint-diagonalization process, in Proc. ICASSP, pp, 606-610, 2020.

    ======
    """

    ## initialize parameters
    eps = 1e-8
    # threshold of regularizer eq. 26
    threshold = 1e-12
    # random seed
    np.random.seed(seed)
    # load spectrogram
    n_freq, n_frame, n_mic = X_FTM.shape
    # regularizer of V_FMM
    E_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1])
    step_init = 1e-6
    step_end = 1e-13
    # construct observed spatial corvariance matrices(SCMs)
    # F*T*M*1 @ F*T*1*M = F*T*M*M
    X_FTMM = X_FTM[..., None] @ X_FTM[..., None, :].conj()
    # initial PSD eq. 3
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(np.float64)
    H_NKT = np.random.rand(n_src, n_components, n_frame).astype(np.float64)

    # initial diagonal covariance matrix and diagonalizer, eq. 41
    Q_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1]).astype(np.complex128)
    G_NFM = np.ones([n_src, n_freq, n_mic], dtype=np.float64) * 1e-2
    for m in range(n_mic):
        G_NFM[m % n_src, :, m] = 1

    if Q_prior_FMM.any() == None:
        raise ValueError("prior information of Q_FMM must be given")

    Q_FMM = Q_prior_FMM

    # normalize
    # eq. 26
    mu_FM = np.sum(Q_FMM * Q_FMM.conj(), axis=2).real
    Q_FMM /= np.sqrt(mu_FM[:, :, None])
    G_NFM /= mu_FM[None, :, :]
    # eq. 27
    phi_NF = G_NFM.sum(axis=2).real
    G_NFM /= phi_NF[..., None]
    W_NFK *= phi_NF[..., None]
    # eq.28
    kappa_NK = W_NFK.sum(axis=1)
    W_NFK /= kappa_NK[:, None]
    H_NKT *= kappa_NK[:, :, None]
    lambda_NFT = W_NFK @ H_NKT + eps

    # transform signal into diagonal domain
    # F*1*M*M @ F*T*M*1 = F*T*M*1
    Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
    Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

    # separate signal with estimated parameters, eq.19
    def separate():
        # F*1*M*M @ F*T*M*1 = F*T*M*1
        Qx_FTM = (Q_FMM[:, None] @ X_FTM[..., None])[..., 0]
        Q_inv_FMM = np.linalg.inv(Q_FMM)
        tmp_NFTM = lambda_NFT[..., None] * G_NFM[:, :, None]
        for n in range(n_src):
            tmp = (
                np.matmul(
                    Q_inv_FMM[:, None],
                    (Qx_FTM * (tmp_NFTM[n] / tmp_NFTM.sum(axis=0)))[..., None],
                )
            )[:, :, mic_index, 0]
            if n == 0:
                separated_spec_MFT = np.zeros([n_src, n_freq, n_frame]).astype(np.complex128)
            separated_spec_MFT[n] = tmp
        separated_spec_FTM = separated_spec_MFT.transpose([1, 2, 0])
        return separated_spec_FTM

    # update
    for epoch in range(n_iter):
        # update NMF 20-21
        # sum_m (N*F*1*M )*(1*F*T*M)
        tmp_yb1_NFT = (G_NFM[:, :, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=3)
        tmp_yb2_NFT = (G_NFM[:, :, None] / Y_FTM[None]).sum(axis=3)
        # sum_t (N*1*K*T)*(N*F*1*T)
        a_1_NFK = (H_NKT[:, None] * tmp_yb1_NFT[:, :, None]).sum(axis=3)
        b_1_NFK = (H_NKT[:, None] * tmp_yb2_NFT[:, :, None]).sum(axis=3)
        # sum_f (N*F*K*1)*(N*F*1*T)
        a_2_NKT = (W_NFK[..., None] * tmp_yb1_NFT[:, :, None]).sum(axis=1)
        b_2_NKT = (W_NFK[..., None] * tmp_yb2_NFT[:, :, None]).sum(axis=1)
        W_NFK *= np.sqrt(a_1_NFK / b_1_NFK)
        H_NKT *= np.sqrt(a_2_NKT / b_2_NKT)

        # update power spectrogram
        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

        # update diagonal matrix
        # sum_t (N*F*T*1)*(1*F*T*M)
        a_3_NFM = (lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=2)
        b_3_NFM = (lambda_NFT[..., None] / Y_FTM[None]).sum(axis=2)
        G_NFM *= np.sqrt(a_3_NFM / b_3_NFM)
        G_NFM += eps
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

        # update diagonalizer
        # this new update rules come from paper Regularized Fast Multichannel Non-negative
        # Matrix Factorization With ILRMA-Based Prior Distribution Of Joint-diagonalization
        # Process
        step_epoch = step_init * ((step_end / step_init) ** (1 / (epoch + 1)))
        for m in range(n_mic):
            # eq. 19
            D_FMM = (X_FTMM / Y_FTM[:, :, m, None, None]).mean(axis=1) \
                    + step_epoch * E_FMM
            u_FM = np.linalg.inv(Q_FMM @ D_FMM)[:, :, m]
            # eq. 23
            q_hat_FM = np.conj(Q_prior_FMM[:, m])
            u_hat_FM = step_epoch * np.linalg.solve(D_FMM, q_hat_FM[..., None])[:, :, 0]
            # eq. 24
            r_F = ((np.conj(u_FM)[:, None, :] @ D_FMM @ u_FM[:, :, None])[:, 0, 0])
            # eq. 25
            r_hat_F = ((np.conj(u_FM)[:, None, :] @ D_FMM @ u_hat_FM[:, :, None])[:, 0, 0])
            # eq. 26
            binary_mask_up_F = np.zeros(n_freq)
            binary_mask_up_F[np.abs(r_hat_F) < threshold] = 1
            binary_mask_low_F = 1 - binary_mask_up_F
            # eq. 26
            q_up_FM = u_FM / np.sqrt(r_F[:, None]) + u_hat_FM
            tem_low_F = r_hat_F / (2 * r_F) * \
                        ((1 + np.sqrt(4 * r_F / (np.abs(r_hat_F) ** 2))) - 1)
            q_low_FM = tem_low_F[:, None] * u_FM + u_hat_FM
            q_FM = binary_mask_up_F[:, None] * q_up_FM + binary_mask_low_F[:, None] * q_low_FM
            Q_FMM[:, m] = np.conj(q_FM)

        # normalize
        # eq. 26
        mu_FM = np.sum(Q_FMM * Q_FMM.conj(), axis=2).real
        Q_FMM /= np.sqrt(mu_FM[:, :, None])
        G_NFM /= mu_FM[None, :, :]
        # eq. 27
        phi_NF = G_NFM.sum(axis=2).real
        G_NFM /= phi_NF[..., None]
        W_NFK *= phi_NF[..., None]
        # eq.28
        kappa_NK = W_NFK.sum(axis=1)
        W_NFK /= kappa_NK[:, None]
        H_NKT *= kappa_NK[:, :, None]
        lambda_NFT = W_NFK @ H_NKT + eps
        # reset variables
        Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
        Y_FTM = (lambda_NFT[..., None] * G_NFM[:, :, None]).sum(axis=0)

    return separate()


def fastmnmf2_VCD(X_FTM, n_src=2, n_iter=50, n_components=4, mic_index=0, seed=0, Q_prior_FMM=None):
    """
    FastMNMF2 with Vectorwise Coordinate Descent(VCD) Regularization
    ======

    Blind Source Separation using fast multichannel non-negative matrix factorization with
    Vector-wise Coordinate Descent(VCD) Regularization
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    n_components: number of NMF basis
    mic_index: recover signal from microphone mix_index
    seed: random seed
    Q_prior_FMM: prior information of Q

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    ----------
    [1] Ito N, Nakatani T. FastMNMF: Joint diagonalization based accelerated algorithms for
    multichannel nonnegative matrix factorization, in Proc. ICASSP, pp. 371-375,  2019

    [2] Sekiguchi K, Bando Y, Nugraha A A, et al. Fast multichannel nonnegative matrix factorization
    with directivity-aware jointly-diagonalizable spatial covariance matrices for blind source separation,
    IEEE/ACM TASLP, vol. 28, pp. 2610-2625, 2020.

    [3] Ito N, Ikeshita R, Sawada H, et al. A Joint Diagonalization Based Efficient Approach to Underdetermined
    Blind Audio Source Separation Using the Multichannel Wiener Filter.
    IEEE/ACM TASLP, 2021.

    [4] Kamo K, Kubo Y, Takamune N, et al. Regularized fast multichannel nonnegative matrix factorization with
    ILRMA-based prior distribution of joint-diagonalization process, in Proc. ICASSP, pp, 606-610, 2020.

    ======
    """
    ## initialize parameters
    # avoid dominator to zero
    eps = 1e-12
    # threshold of regularizer eq. 26
    threshold = 1e-12

    # random seed
    np.random.seed(seed)

    # load spectrogram
    n_freq, n_frame, n_mic = X_FTM.shape

    # regularizer of V_FMM
    E_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1])
    step_init = 1e-6
    step_end = 1e-13
    # construct observed spatial corvariance matrices(SCMs)
    # F*T*M*1 @ F*T*1*M = F*T*M*M
    X_FTMM = X_FTM[..., None] @ X_FTM[..., None, :].conj()
    # initial PSD eq. 3
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(np.float64)
    H_NKT = np.random.rand(n_src, n_components, n_frame).astype(np.float64)

    # initial diagonal covariance matrix and diagonalizer, eq. 41
    Q_FMM = np.tile(np.eye(n_mic), [n_freq, 1, 1]).astype(np.complex128)
    G_NM = np.ones([n_src, n_mic], dtype=np.float64) * 1e-2
    mu_F = np.zeros(n_freq, dtype=np.float64)
    for m in range(n_mic):
        G_NM[m % n_src, m] = 1

    if Q_prior_FMM.any() == None:
        raise ValueError("prior information of Q_FMM must be given")

    Q_FMM = Q_prior_FMM

    # normalize
    # eq. 37
    for f in range(n_freq):
        mu_F[f] = (np.trace(Q_FMM[f]*(Q_FMM[f].T.conj())) / n_mic).real
    Q_FMM /= np.sqrt(mu_F[:, None, None])
    W_NFK /= mu_F[None, :, None]

    # eq. 38
    phi_N = G_NM.sum(axis=1)
    G_NM /= phi_N[:, None]
    W_NFK *= phi_N[:, None, None]
    lambda_NFT = W_NFK @ H_NKT + eps

    # transform signal into diagonal domain
    # F*1*M*M @ F*T*M*1 = F*T*M*1
    Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
    Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

    # separate signal with estimated parameters, eq.19
    def separate():
        # F*1*M*M @ F*T*M*1 = F*T*M*1
        Qx_FTM = (Q_FMM[:, None] @ X_FTM[..., None])[..., 0]
        Q_inv_FMM = np.linalg.inv(Q_FMM)
        tmp_NFTM = lambda_NFT[..., None] * G_NM[:, None, None]
        for n in range(n_src):
            tmp = (
                np.matmul(
                    Q_inv_FMM[:, None],
                    (Qx_FTM * (tmp_NFTM[n] / tmp_NFTM.sum(axis=0)))[..., None],
                )
            )[:, :, mic_index, 0]
            if n == 0:
                separated_spec_MFT = np.zeros([n_src, n_freq, n_frame]).astype(np.complex128)
            separated_spec_MFT[n] = tmp
        separated_spec_FTM = separated_spec_MFT.transpose([1, 2, 0])
        return separated_spec_FTM

    # update
    for epoch in range(n_iter):
        # update NMF 20-21
        # sum_m (N*F*1*M )*(1*F*T*M)
        tmp_yb1_NFT = (G_NM[:, None, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=3)
        tmp_yb2_NFT = (G_NM[:, None, None] / Y_FTM[None]).sum(axis=3)
        # sum_t (N*1*K*T)*(N*F*1*T)
        a_1_NFK = (H_NKT[:, None] * tmp_yb1_NFT[:, :, None]).sum(axis=3)
        b_1_NFK = (H_NKT[:, None] * tmp_yb2_NFT[:, :, None]).sum(axis=3)
        # sum_f (N*F*K*1)*(N*F*1*T)
        a_2_NKT = (W_NFK[..., None] * tmp_yb1_NFT[:, :, None]).sum(axis=1)
        b_2_NKT = (W_NFK[..., None] * tmp_yb2_NFT[:, :, None]).sum(axis=1)
        W_NFK *= np.sqrt(a_1_NFK / b_1_NFK)
        H_NKT *= np.sqrt(a_2_NKT / b_2_NKT)

        # update power spectrogram
        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

        # update diagonal matrix
        # sum_t (N*F*T*1)*(1*F*T*M)
        a_3_NM = (lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]).sum(axis=(1, 2))
        b_3_NM = (lambda_NFT[..., None] / Y_FTM[None]).sum(axis=(1, 2))
        G_NM *= np.sqrt(a_3_NM / b_3_NM)
        G_NM += eps
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

        # update diagonalizer
        # this new update rules come from paper Regularized Fast Multichannel Non-negative
        # Matrix Factorization With ILRMA-Based Prior Distribution Of Joint-diagonalization
        # Process
        step_epoch = step_init * ((step_end / step_init) ** (1/(epoch+1)))
        for m in range(n_mic):
            # eq. 19
            D_FMM = (X_FTMM / Y_FTM[:, :, m, None, None]).mean(axis=1) \
                    + step_epoch * E_FMM
            u_FM = np.linalg.inv(Q_FMM @ D_FMM)[:, :, m]
            # eq. 23
            q_hat_FM = np.conj(Q_prior_FMM[:, m])
            u_hat_FM = step_epoch * np.linalg.solve(D_FMM, q_hat_FM[..., None])[:, :, 0]
            # eq. 24
            r_F = ((np.conj(u_FM)[:, None, :] @  D_FMM @ u_FM[:, :, None])[:, 0, 0])
            # eq. 25
            r_hat_F = ((np.conj(u_FM)[:, None, :] @  D_FMM @ u_hat_FM[:, :, None])[:, 0, 0])
            # eq. 26
            binary_mask_up_F = np.zeros(n_freq)
            binary_mask_up_F[np.abs(r_hat_F) < threshold] = 1
            binary_mask_low_F = 1 - binary_mask_up_F
            # eq. 26
            q_up_FM = u_FM / np.sqrt(r_F[:, None]) + u_hat_FM
            tem_low_F = r_hat_F / (2 * r_F) * ((1 + np.sqrt(4 * r_F / (np.abs(r_hat_F) ** 2))) - 1)
            q_low_FM = tem_low_F[:, None] * u_FM + u_hat_FM
            q_FM = binary_mask_up_F[:, None] * q_up_FM + binary_mask_low_F[:, None] * q_low_FM
            Q_FMM[:, m] = np.conj(q_FM)

        # normalize
        # eq. 37
        for f in range(n_freq):
            mu_F[f] = (np.trace(Q_FMM[f]*(Q_FMM[f].T.conj())) / n_mic).real
        Q_FMM /= np.sqrt(mu_F[:, None, None])
        W_NFK /= mu_F[None, :, None]

        # eq.38
        phi_N = G_NM.sum(axis=1)
        G_NM /= phi_N[:, None]
        W_NFK *= phi_N[:, None, None]
        lambda_NFT = W_NFK @ H_NKT + eps
        # reset variables
        Qx_power_FTM = np.abs((Q_FMM[:, None] @ X_FTM[..., None])[..., 0]) ** 2
        Y_FTM = (lambda_NFT[..., None] * G_NM[:, None, None]).sum(axis=0)

    return separate()
