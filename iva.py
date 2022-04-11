# Author: wangxianrui
# Author: wangxianrui@mail.nwpu.edu.cn

from basetool import *


def auxiva(X_FTM=None, n_src=None, n_iter=20, distribution="laplace"):
    """
    AuxIVA
    ======

    Blind Source Separation using independent vector analysis based on auxiliary function.
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    distribution: gauss or laplace

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *

    References
    ----------
    .. [1] N. Ono, *Stable and fast update rules for independent vector analysis based
        on auxiliary function technique,* Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.

    ======
    """

    eps = 1e-15
    threshold = 5e-5
    # size of T*F*M
    X_TFM = X_FTM.transpose([1, 0, 2])
    # if M>N, Principle Component Analysis (PCA) should be implemented
    spec_FNT = X_TFM.transpose([1, 2, 0])  # size of F*N*T
    n_freq, n_channel, n_frame = spec_FNT.shape
    # parameters check
    assert (n_src <= n_channel), "The sources cannot be more than microphones"
    if distribution not in ["laplace", "gauss"]:
        raise ValueError("distribution not included, please use laplace or gaussian")

    # memory allocation
    # YHat_FNT = np.zeros([n_freq, n_sr, n_frame])
    eyes = np.tile(np.eye(n_src, n_src), (n_freq, 1, 1))
    W_FNN = np.tile(np.eye(n_src, n_src, dtype=X_TFM.dtype), (n_freq, 1, 1))
    loss_old = np.inf

    # demix signal in stft domain
    def separate():
        """
        separate signal
        """
        separated_spec = W_FNN @ spec_FNT
        return separated_spec  # separated spectrogram F*M*T

    def cal_loss():
        """
        calculate cost function
        """
        G_NT = np.sqrt(np.sum(np.abs(YHat_FNT)**2, axis=0))
        G_N = np.mean(G_NT, axis=1)
        G = np.sum(G_N)
        lod_det = np.sum(np.log(np.abs(np.linalg.det(W_FNN))))
        loss = (G-lod_det)/(n_freq*n_src)
        loss_imp = np.abs(loss-loss_old)/abs(loss)
        loss_last = loss
        return loss, loss_last, loss_imp

    def cal_r_inv():
        """
        calculate r in eq. 34
        """
        if distribution == "laplace":
            r = 2.0*np.linalg.norm(YHat_FNT, axis=0)
        elif distribution == "gauss":
            r = (np.linalg.norm(YHat_FNT, axis=0)**2)/n_freq
        r[r < eps] = eps
        r_inv = 1.0/r
        return r_inv

    # estimate demix matrix with n_iter epoches
    for epoch in range(n_iter):
        YHat_FNT = separate()
        if epoch % 10 == 0:
            loss_current, loss_old, loss_imporve = cal_loss()
            print(loss_imporve)
            if loss_imporve <= threshold:
                break
        r_inv_NT = cal_r_inv()
        for n in range(n_src):
            V_FNN = np.matmul(spec_FNT*r_inv_NT[None, n, None, :],
                              np.conj(spec_FNT.swapaxes(1, 2)))/n_frame
            WV_FNN = np.matmul(W_FNN, V_FNN)
            u = np.linalg.solve(WV_FNN, eyes[:, :, n])  # eq. 36
            u_temp1 = np.sqrt(np.matmul(np.matmul(np.conj(u[:, None, :]), V_FNN),
                                        u[:, :, None]))
            u_temp2 = (u[:, :, None]/u_temp1)[:, :, 0]
            W_FNN[:, n, :] = np.conj(u_temp2)
    YHat_FNT = separate()
    YHat_TFN = YHat_FNT.transpose([2, 0, 1])
    spec_F0T = spec_FNT[:, 0, :]
    X_T0F = spec_F0T.T
    Z = projection_back(YHat_TFN, X_T0F)
    YHat_TFN *= np.conj(Z[None, :, :])
    YHat_FTN = YHat_TFN.transpose([1, 0, 2])
    return YHat_FTN, W_FNN


def gradiva(X_FTM=None, n_src=None, n_iter=500):
    """
    GradIVA
    ======

    Blind Source Separation using independent vector analysis with nature gradient.
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *  n_channels

    References
    -------
    [1]. Kim T, Eltoft T, Lee T W. INDEPENDENT VECTOR ANALYSIS: DEFINITION AND ALGORITHMS

    ======
    """

    eps = 1e-15
    mu = 1e-2
    threshold = 5e-5
    # size of T*F*M
    X_TFM = X_FTM.transpose([1, 0, 2])
    # if M>N, Principle Component Analysis (PCA) should be implemented
    spec_FNT = X_TFM.transpose([1, 2, 0])  # size of F*N*T
    n_freq, n_channel, n_frame = spec_FNT.shape
    # parameters check
    assert (n_src <= n_channel), "The sources cannot be more than microphones"

    # memory allocation
    # YHat_FNT = np.zeros([n_freq, n_sr, n_frame])
    eyes = np.tile(np.eye(n_src, n_src), (n_freq, 1, 1))
    W_FNN = np.tile(np.eye(n_src, n_src, dtype=X_TFM.dtype), (n_freq, 1, 1))
    loss_old = np.inf
    # demix signal in stft domain

    def separate():
        """
        separate signal
        """
        separated_spec = W_FNN @ spec_FNT
        return separated_spec  # separated spectrogram F*M*T

    def cal_loss():
        """
        calculate cost function
        """
        G_NT = np.sqrt(np.sum(np.abs(YHat_FNT)**2, axis=0))
        G_N = np.mean(G_NT, axis=1)
        G = np.sum(G_N)
        lod_det = np.sum(np.log(np.abs(np.linalg.det(W_FNN))))
        loss = (G-lod_det)/(n_freq*n_src)
        loss_imp = np.abs(loss-loss_old)/abs(loss)
        loss_last = loss
        return loss, loss_last, loss_imp

    # estimate demix matrix with n_iter epoches
    for epoch in range(n_iter):
        YHat_FNT = separate()
        YHat_FTN = YHat_FNT.transpose([0, 2, 1])
        if epoch % 100 == 0:
            loss_current, loss_old, loss_imporve = cal_loss()
            print(loss_imporve)
            if loss_imporve <= threshold:
                break
        # eq. 14
        phi_FNT = YHat_FNT / (np.linalg.norm(YHat_FNT, axis=0)[None]+eps)
        phi_FTN = phi_FNT.transpose([0, 2, 1])
        phi_y_FTNN = np.matmul(phi_FTN[:, :, :, None], np.conj(YHat_FTN[:, :, None, :]))
        phi_y_FNN = np.mean(phi_y_FTNN, axis=1)  # F*N*N
        gradient = np.matmul(eyes-phi_y_FNN, W_FNN)
        W_FNN = W_FNN + mu * gradient

    YHat_FNT = separate()
    YHat_TFN = YHat_FNT.transpose([2, 0, 1])
    spec_F0T = spec_FNT[:, 0, :]
    X_T0F = spec_F0T.T
    Z = projection_back(YHat_TFN, X_T0F)
    YHat_TFN *= np.conj(Z[None, :, :])
    YHat_FTN = YHat_TFN.transpose([1, 0, 2])
    return YHat_FTN, W_FNN
