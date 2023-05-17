import pandas as pd
import numpy as np
from scipy import signal
from typing import Iterable
import matplotlib.pyplot as plt


def extract_signal_feature(ecg_segments: dict, columns: Iterable[str], sample_num=600):
    segment_num = ecg_segments["I"].shape[0]
    signal_features = []
    for i in range(segment_num):
        # get data of all 12 leads in each segment
        ecg_segment = []
        for column in columns:
            lead_segment = ecg_segments[column][i]
            ecg_segment.append(lead_segment)
        ecg_segment = np.concatenate(ecg_segment)
        # down sample ecg segments
        ecg_segment = signal.resample(ecg_segment, sample_num)

        # normalize data
        ecg_segment = ecg_segment / np.abs(ecg_segment).max()
        signal_features.append(ecg_segment)

    signal_features = np.stack(signal_features)
    return signal_features




