import numpy as np
import scipy.signal
import scipy.fft
from typing import Iterable
import matplotlib.pyplot as plt


EPSN = 1e-8


def _fft_fft(sequence_data):
    fft_trans = np.abs(scipy.fft.fft(sequence_data))
    dc = fft_trans[0]
    freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
    freq_sum = freq_spectrum.sum()
    return dc, freq_spectrum, freq_sum


def _fft_mean(sequence_data):
    _, freq_spectrum, _freq_sum_ = _fft_fft(sequence_data)
    return np.mean(freq_spectrum)


def _fft_var(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    return np.var(freq_spectrum)


def _fft_std(sequence_data):
    _, freq_spectrum, _freq_sum_ = _fft_fft(sequence_data)
    return np.std(freq_spectrum)


def _fft_entropy(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    pr_freq = freq_spectrum * 1.0 / freq_sum
    entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])
    return entropy


def _fft_energy(sequence_data):
    _, freq_spectrum, _freq_sum_ = _fft_fft(sequence_data)
    return np.sum(freq_spectrum ** 2) / len(freq_spectrum)


def _fft_skew(sequence_data):
    _, freq_spectrum, _freq_sum_ = _fft_fft(sequence_data)
    fft_mean, fft_std = _fft_mean(sequence_data), _fft_std(sequence_data)
    return np.mean([0 if fft_std < EPSN else np.power((x - fft_mean) / fft_std, 3) for x in freq_spectrum])


def _fft_kurt(sequence_data):
    _, freq_spectrum, _freq_sum_ = _fft_fft(sequence_data)
    fft_mean, fft_std = _fft_mean(sequence_data), _fft_std(sequence_data)
    return np.mean([0 if fft_std < EPSN else np.power((x - fft_mean) / fft_std, 4) for x in freq_spectrum])


def _fft_shape_mean(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    shape_sum = np.sum([x * freq_spectrum[x]
                        for x in range(len(freq_spectrum))])
    return 0 if freq_sum < EPSN else shape_sum * 1.0 / freq_sum


def _fft_shape_std(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    shape_mean = _fft_shape_mean(sequence_data)
    var = np.sum([0 if freq_sum < EPSN else np.power((x - shape_mean), 2) * freq_spectrum[x]
                  for x in range(len(freq_spectrum))]) / freq_sum
    return np.sqrt(var)


def _fft_shape_skew(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    shape_mean = _fft_shape_mean(sequence_data)
    return np.sum([np.power((x - shape_mean), 3) * freq_spectrum[x] for x in range(len(freq_spectrum))]) / freq_sum


def _fft_shape_kurt(sequence_data):
    _, freq_spectrum, freq_sum = _fft_fft(sequence_data)
    shape_mean = _fft_shape_mean(sequence_data)
    return np.sum([np.power((x - shape_mean), 4) * freq_spectrum[x] - 3
                   for x in range(len(freq_spectrum))]) / freq_sum


def _extract_segment_freq_features(sequence_data):
    fre_features = []

    result_fft_mean = _fft_mean(sequence_data)
    fre_features.append(result_fft_mean)

    result_fft_var = _fft_var(sequence_data)
    fre_features.append(result_fft_var)

    result_fft_std = _fft_std(sequence_data)
    fre_features.append(result_fft_std)

    result_fft_entropy = _fft_entropy(sequence_data)
    fre_features.append(result_fft_entropy)

    result_fft_energy = _fft_energy(sequence_data)
    fre_features.append(result_fft_energy)

    result_fft_skew = _fft_skew(sequence_data)
    fre_features.append(result_fft_skew)

    result_fft_kurt = _fft_kurt(sequence_data)
    fre_features.append(result_fft_kurt)

    result_fft_shape_mean = _fft_shape_mean(sequence_data)
    fre_features.append(result_fft_shape_mean)

    result_fft_shape_std = _fft_shape_std(sequence_data)
    fre_features.append(result_fft_shape_std)

    # result_fft_shape_skew=fft_shape_skew(sequence_data)
    # fre_features.append(result_fft_shape_skew)

    # result_fft_shape_kurt=fft_shape_kurt(sequence_data)
    # fre_features.append(result_fft_shape_kurt)

    fre_features = np.array(fre_features)
    # print(fre_features)
    return fre_features


def extract_freq_feature(ecg_segments: dict, columns: Iterable[str], sample_num=600):
    segment_num = ecg_segments["I"].shape[0]
    freq_features = []
    for i in range(segment_num):
        # get freq feature_data of all 12 leads in each segment
        segment_freq_features = []
        for column in columns:
            lead_segment = ecg_segments[column][i]
            segment_freq_feature = _extract_segment_freq_features(lead_segment)
            segment_freq_features.append(segment_freq_feature)
        segment_freq_features = np.concatenate(segment_freq_features)
        freq_features.append(segment_freq_features)

    freq_features = np.stack(freq_features)
    return freq_features
