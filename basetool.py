# Author: Xianrui Wang
# Contact: wangxianrui@mail.nwpu.edu.cn

import librosa
import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources, bss_eval_images


def multichannel_stft(mixed_signal=None, nfft=None, hop=None):
    """

    Parameters
    ----------
    data: input data M*T, notice T is time length
    n_fft: block of fft
    hop_length: hop size
    win_length: length of window

    Returns
    -------
    spec_FTM: multichannel spectrogram F*T*M, notice T means stft frames
    """
    M = mixed_signal.shape[0]
    for m in range(M):
        tmp = librosa.core.stft(mixed_signal[m], win_length=nfft, n_fft=nfft, hop_length=hop)
        if m == 0:
            spec_FTM = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec_FTM[:, :, m] = tmp
    return spec_FTM


def multichannel_istft(spec_FTM, nfft=None, hop=None):
    """

    Parameters
    ----------
    spectrogram: input multichannel spectrogram T*F*M, notice T means stft frames
    hop_length: hop size
    win_length: length of window
    ori_length: length of original length

    Returns
    -------
    data: multichannel time domain signal M*T , notice T is time length
    """
    N = spec_FTM.shape[2]
    for n in range(N):
        tmp = librosa.istft(spec_FTM[..., n], win_length=nfft, hop_length=hop)

        if n == 0:
            y = np.zeros([N, tmp.shape[0]])
        y[n] = tmp
    return y


def add_noise(multichannel_signal, SNR):
    """
    Add multichannel noise with given signal-to-noise ratio (SNR)
    """
    channel, siglength = multichannel_signal.shape
    random_values = np.random.rand(channel, siglength)
    Ps = np.mean(multichannel_signal ** 2, axis=1)
    Pn1 = np.mean(random_values ** 2, axis=1)
    print(Ps.shape, Pn1.shape)
    k = (np.sqrt(Ps/(10**(SNR/10)*Pn1)))[:, None]
    random_values_we_need = random_values*k
    outdata = multichannel_signal + random_values_we_need
    return outdata


def performance_measure(s_image, y, sHat):
    sdr_mix, sir_mix, sar_mix, perm_mix = bss_eval_sources(s_image, y, compute_permutation=True)
    sdr, sir, sar, perm = bss_eval_sources(s_image, sHat, compute_permutation=True)
    sdr_improve = sdr-sdr_mix
    sir_improve = sir-sir_mix
    return sdr_mix, sir, sdr_mix, sdr, sdr_improve, sir_improve


def projection_back(Y, ref, clip_up=None, clip_down=None):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Here is the derivation of the projection.
    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)

    c = np.ones(num.shape, dtype=np.complex)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])
    return c



