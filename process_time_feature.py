import numpy as np
from typing import Iterable


EPSN = 1e-8


def _mean_fea(a):
    return np.mean(a)


def _rms_fea(a):
    return np.sqrt(np.mean(np.square(a)))


def _sr_fea(a):
    return np.square(np.mean(np.sqrt(np.abs(a))))


def _am_fea(a):
    return np.mean(np.abs(a))


def _skew_fea(a):
    return np.mean((a - _mean_fea(a)) ** 3)


def _kurt_fea(a):
    return np.mean((a - _mean_fea(a)) ** 4)


def _max_fea(a):
    return np.max(a)


def _min_fea(a):
    return np.min(a)


def _pp_fea(a):
    return _max_fea(a) - _min_fea(a)


def _var_fea(a):
    n = len(a)
    return np.sum((a - _mean_fea(a)) ** 2) / (n - 1)


def _waveform_index(a):
    return _rms_fea(a) / (_am_fea(a) + EPSN)


def _peak_index(a):
    return _max_fea(a) / (_rms_fea(a) + EPSN)


def _impluse_factor(a):
    return _max_fea(a) / (_am_fea(a) + EPSN)


def _tolerance_index(a):
    return _max_fea(a) / (_sr_fea(a) + EPSN)


def _skew_index(a):
    n = len(a)
    temp1 = np.sum((a - _mean_fea(a)) ** 3)
    temp2 = (np.sqrt(_var_fea(a))) ** 3
    return temp1 / ((n - 1) * temp2)


def _kurt_index(a):
    n = len(a)
    temp1 = np.sum((a - _mean_fea(a)) ** 4)
    temp2 = (np.sqrt(_var_fea(a))) ** 4
    return temp1 / ((n - 1) * temp2)


def _extract_segment_time_features(sequence_data):
    time_features = []

    result_mean_fea = _mean_fea(sequence_data)
    time_features.append(result_mean_fea)

    result_rms_fea = _rms_fea(sequence_data)
    time_features.append(result_rms_fea)

    result_sr_fea = _sr_fea(sequence_data)
    time_features.append(result_sr_fea)

    result_am_fea = _am_fea(sequence_data)
    time_features.append(result_am_fea)

    result_skew_fea = _skew_fea(sequence_data)
    time_features.append(result_skew_fea)

    result_kurt_fea = _kurt_fea(sequence_data)
    time_features.append(result_kurt_fea)

    result_max_fea = _max_fea(sequence_data)
    time_features.append(result_max_fea)

    result_min_fea = _min_fea(sequence_data)
    time_features.append(result_min_fea)

    result_pp_fea = _pp_fea(sequence_data)
    time_features.append(result_pp_fea)

    result_var_fea = _var_fea(sequence_data)
    time_features.append(result_var_fea)

    result_waveform_index_fea = _waveform_index(sequence_data)
    time_features.append(result_waveform_index_fea)

    result_peak_index_fea = _peak_index(sequence_data)
    time_features.append(result_peak_index_fea)

    result_impluse_factor_fea = _impluse_factor(sequence_data)
    time_features.append(result_impluse_factor_fea)

    # result_tolerance_index_fea = _tolerance_index(sequence_data)
    # feature_data.append(result_tolerance_index_fea)

    # result_skew_index_fea = _skew_index(sequence_data)
    # feature_data.append(result_skew_index_fea)

    # result_kurt_index_fea = _kurt_index(sequence_data)
    # feature_data.append(result_kurt_index_fea)

    time_features = np.stack(time_features)
    return time_features


def extract_time_feature(ecg_segments: dict, columns: Iterable[str], sample_num=600):
    segment_num = ecg_segments["I"].shape[0]
    time_features = []
    for i in range(segment_num):
        # get time feature_data of all 12 leads in each segment
        segment_time_features = []
        for column in columns:
            lead_segment = ecg_segments[column][i]
            segment_time_feature = _extract_segment_time_features(lead_segment)
            segment_time_features.append(segment_time_feature)
        segment_time_features = np.concatenate(segment_time_features)
        time_features.append(segment_time_features)

    time_features = np.stack(time_features)
    return time_features
