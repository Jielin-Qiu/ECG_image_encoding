import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from typing import Iterable

# from sklearn.cluster import KMeans


LABEL_INDEX = {
    'NORM': 0,
    'MI': 1,
    'STTC': 2,
    'CD': 3,
    'HYP': 4
}



LABEL_COLOR = {
    'NORM': 'green',
    'MI': "red",
    'STTC': 'yellow',
    'CD': 'orange',
    'HYP': "blue"
}


def get_sampling_rate(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        sampling_rate = int(root.find('Waveform').find('SampleBase').text)
    except AttributeError as e:
        print(e)
        sampling_rate = -1
    return sampling_rate


def plot_all_leads(data, sampling_rate, img_path):
    columns = list(data.columns)
    assert len(columns) <= 12
    assert sampling_rate != -1
    # leads data
    fig, axs = plt.subplots(6, 2, figsize=[28, 12])
    for i, column in enumerate(columns):
        y = data[column].to_numpy()

        x = np.arange(0, y.size) / sampling_rate
        plot_row = i % 6
        plot_col = int((i - plot_row) / 6)
        ax = axs[plot_row, plot_col]
        ax.plot(x, y, color='blue')
        ax.set_xlabel("Time (s)", size=12)
        ax.set_ylabel(f"Lead {column}", size=12)
        ax.tick_params(labelsize=9)
        ax.set_xlim(x[0], x[-1])

    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def plot_r_peaks(data: pd.DataFrame, sampling_rate: int, lead_name: str, peak_indexes: list, img_path: str):
    columns = list(data.columns)
    assert len(columns) <= 12
    assert sampling_rate > 0

    fig, ax = plt.subplots(figsize=[16, 4])
    y = data[lead_name].to_numpy()
    x = np.arange(0, y.size) / sampling_rate
    x_peak = np.asarray(peak_indexes) / sampling_rate
    y_peak = np.take(y, peak_indexes)
    # plot data
    ax.plot(x, y, color='blue')
    ax.set_xlabel("Time", size=12)
    ax.set_ylabel(f"Lead {lead_name}", size=12)
    ax.tick_params(labelsize=9)
    # plot Peak data
    ax.plot(x_peak, y_peak, 'o', color="r")
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def plot_segments(data: pd.DataFrame, sampling_rate: int, peaks: Iterable[int], img_path: str, length: float = 1.0, ):
    columns = list(data.columns)
    assert len(columns) <= 12
    assert sampling_rate != -1
    # leads data
    fig, axs = plt.subplots(6, 2, figsize=[28, 12])
    start_shift = int(35 * length)
    end_shift = int(65 * length)

    for i, column in enumerate(columns):
        y = data[column].to_numpy()
        x = np.arange(0, y.size) / sampling_rate
        plot_row = i % 6
        plot_col = int((i - plot_row) / 6)
        ax = axs[plot_row, plot_col]
        # plot smoothed data
        ax.plot(x, y, color='blue')
        ax.set_xlabel("Time (s)", size=12)
        ax.set_ylabel(f"Lead {column}", size=12)
        ax.tick_params(labelsize=9)
        ax.set_xlim(x[0], x[-1])
        # plot data segments
        for i, peak in enumerate(peaks):
            if i % 2 == 0:
                color = "red"
            else:
                color = "blue"

            if start_shift <= peak < y.size - end_shift:
                start_index = (peak - start_shift) / sampling_rate
                end_index = (peak + end_shift) / sampling_rate
                ax.axvspan(start_index, end_index, alpha=0.2, color=color)

    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def plot_segments_prediction(
        data: pd.DataFrame,
        sampling_rate: int,
        peaks: Iterable[int],
        predictions,
        img_path: str,
        length: float = 1.0, ):
    columns = list(data.columns)
    assert len(columns) <= 12
    assert sampling_rate != -1
    peaks = np.asarray(peaks)
    predictions = np.asarray(predictions)

    peaks = peaks[(35 <= peaks) & (peaks < 1000 - 65)]
    assert len(peaks) == len(predictions), f"{len(peaks)}, {len(predictions)}"

    # leads data
    fig, axs = plt.subplots(7, 2, figsize=[30, 12])
    start_shift = int(35 * length)
    end_shift = int(65 * length)

    # get position of the plot labels
    tmp = [peaks[0]]
    last_label = predictions[0, 0]
    text_positions = []
    text_predictions = []
    for i in range(1, len(peaks)):
        label = predictions[i, 0]
        if label == last_label:
            tmp.append(peaks[i])
        else:
            text_positions.append(np.mean(tmp))
            text_predictions.append(last_label)
            tmp = [peaks[i]]
        last_label = label

    if len(tmp) > 0:
        text_positions.append(np.mean(tmp))
        text_predictions.append(last_label)

    for i, column in enumerate(columns):
        y = data[column].to_numpy()
        x = np.arange(0, y.size) / sampling_rate
        plot_row = i % 6
        plot_col = int((i - plot_row) / 6)
        ax = axs[plot_row, plot_col]
        # plot smoothed data
        ax.plot(x, y, color='blue')
        ax.set_xlabel("Time (s)", size=12)
        ax.set_ylabel(f"Lead {column}", size=12)
        ax.set_xlim(0, y.size / sampling_rate)
        # ax.set_ylim(bottom=y.min() - 0.25)
        ax.tick_params(labelsize=9)
        # plot data segments
        for i, peak in enumerate(peaks):
            prediction = predictions[i, 0]
            # get name of the prediction
            diagnosis = list(LABEL_INDEX.keys())[list(LABEL_INDEX.values()).index(prediction)]
            color = LABEL_COLOR[diagnosis]

            if start_shift <= peak < y.size - end_shift:
                start_index = (peak - start_shift) / sampling_rate
                end_index = (peak + end_shift) / sampling_rate
                ax.axvspan(start_index, end_index, alpha=0.2, color=color)

    # plot prediction result for each segment
    for col in range(2):

        plot_row = 6
        plot_col = col
        ax = axs[plot_row, plot_col]
        ax.set_xlim(0, 1000 / sampling_rate)
        ax.tick_params(labelsize=9)
        # plot data segments
        for i, peak in enumerate(peaks):
            prediction = predictions[i, 0]
            # get name of the prediction
            diagnosis = list(LABEL_INDEX.keys())[list(LABEL_INDEX.values()).index(prediction)]
            color = LABEL_COLOR[diagnosis]
            # fill with color
            if start_shift <= peak < 1000 - end_shift:
                start_index = (peak - start_shift) / sampling_rate
                end_index = (peak + end_shift) / sampling_rate
                ax.axvspan(start_index, end_index, alpha=0.2, color=color)
            # plot labels
            y_scale = 0.15
            # ax.text((peak - 15) / sampling_rate, y_scale * 5, f"NORM: {np.round(predictions[i, 1] * 100)}%", size=8)
            ax.text((peak - 15) / sampling_rate, y_scale * 4, f"MI: {np.round(predictions[i, 2] * 100)}%", size=8)
            # ax.text((peak - 15) / sampling_rate, y_scale * 3, f"STTC: {np.round(predictions[i, 3] * 100)}%", size=8)
            # ax.text((peak - 15) / sampling_rate, y_scale * 2, f"CD: {np.round(predictions[i, 4] * 100)}%", size=8)
            # ax.text((peak - 15) / sampling_rate, y_scale * 1, f"HYP: {np.round(predictions[i, 5] * 100)}%", size=8)





        # for position, prediction in zip(text_positions, text_predictions):
        #     diagnosis = list(LABEL_INDEX.keys())[list(LABEL_INDEX.values()).index(prediction)]
        #

    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


# def checkR(lead: np.ndarray, sampling_rate):
#     lead = lead.flatten()
#     max_val = lead.max()
#     min_val = lead.min()
#     threshold_val = (max_val - min_val) * 0.7 + min_val
#     index = []
#     for i in range(1, lead.size - 2):
#         if lead[i] > threshold_val and lead[i - 1: i + 2].argmax() == 1:
#             if len(index) > 0:
#                 # check if the peak is too close or too far to the previous one
#                 lb = 60.0 / 160.0 * sampling_rate
#                 ub = 60.0 / 60.0 * sampling_rate
#                 if lb <= i - index[-1] <= ub:
#                     index.append(i)
#             else:
#                 index.append(i)
#     return np.array(index)


def ecg_fft_ana(ecg_original, sampling_rate):
    fs = sampling_rate
    n = len(ecg_original)
    k = np.arange(n)
    t = n / fs
    frq = k / t
    frq = frq[range(int(n / 2))]
    fft_ecg = np.abs(np.fft.fft(ecg_original))[range(int(n / 2))]
    return frq, fft_ecg


def get_r_peaks(data: pd.DataFrame, split_lead_name: str):

    lead_names = list(data.columns.values)
    lead_split_norm = None
    distances = []
    for lead_name in lead_names:
        lead = data[lead_name].to_numpy()
        max_heart_rate = 140 / 60
        min_heart_rate = 20 / 60
        min_distance = 100 / max_heart_rate

        # get r peaks according to the max heart rate
        kernel_size = 100
        kernel = np.ones((kernel_size,)) / kernel_size
        lead_data_mean = np.convolve(lead, kernel, mode="same")
        lead_magnitude = lead - lead_data_mean * 0

        # find magnitude of the lead
        tmp = lead_magnitude.reshape((-1, 100))
        magnitude_pos = np.abs(np.percentile(tmp.max(axis=1), 50))
        magnitude_neg = np.abs(np.percentile(tmp.min(axis=1), 50))
        if magnitude_pos > magnitude_neg:
            lead_norm = lead / (magnitude_pos + 1e-8)
        else:
            lead_norm = -lead / (magnitude_neg + 1e-8)



        #
        # start_index = int(0.3 * lead_split.size)
        # end_index = int(0.7 * lead_split.size)
        # lead_split_mid = lead_magnitude[start_index: end_index]
        # peak_max = lead_split_mid.max()
        # peak_mean = lead_split_mid.mean()

        peaks, properties = scipy.signal.find_peaks(
            lead_norm,
            distance=0.8 * min_distance,
            prominence=0.6
        )

        # find heart rate based on the peaks
        lead_simple = np.zeros_like(lead_norm)
        # peaks_value = lead_split_norm.take(peaks)
        np.put(lead_simple, peaks, np.ones_like(peaks))
        # np.put(lead_simple, peaks, properties['prominences'])

        # plt.plot(lead_simple)
        # plt.show()

        # compute beat frequency with fourier transform
        f, p = scipy.signal.periodogram(lead_simple, fs=100)
        mask = (f < max_heart_rate) & (f > min_heart_rate)
        f = f[mask]
        p = p[mask] / f
        # plt.plot(p / f)
        # plt.show()
        distance = 100 / f[p.argmax()]
        distances.append(distance)
        if lead_name == split_lead_name:
            lead_split_norm = lead / (magnitude_pos + 1e-8)

    # get r peaks according to the beat frequency
    distance = np.median(distances)
    peaks, properties = scipy.signal.find_peaks(
        lead_split_norm,
        distance=0.6 * distance,
        prominence=0.6
    )
    # print(properties['prominences'].min())

    # estimate heart rate from peaks
    # print(properties)
    # print(peaks)
    # plt.plot(lead_split)
    # plt.plot(lead_split_norm)
    # plt.show()

    e_heart_rate = 100 / np.percentile(np.diff(peaks), 50)
    # e_max_heart_rate = e_heart_rate * 1.2
    # e_min_heart_rate = e_heart_rate * 0.8
    #
    # lead_simple = np.zeros_like(lead_split_norm)
    # peaks_value = lead_split_norm.take(peaks)
    # np.put(lead_simple, peaks, peaks_value)
    #
    # f, p = scipy.signal.periodogram(lead_simple, fs=100)
    #
    # mask = (f < e_max_heart_rate) & (f > e_min_heart_rate)
    # f = f[mask]
    # p = p[mask]
    # heart_rate = f[p.argmax()]

    return peaks, e_heart_rate

    # W = np.fft.fft(lead_split)
    # freq = np.fft.fftfreq(len(lead_split), 1)
    #
    # threshold = 10 ** 4
    # idx = np.where(np.abs(W) > threshold)[0][-1]
    # print(np.where(np.abs(W) > threshold)[0][-1])
    #
    # print()
    # quit()
    #
    #
    #
    #
    # plt.show()
    #
    #
    #
    #
    #
    # # check if we should use positive or negative lead
    # lead_medium = np.percentile(lead_split, 50)
    # gap_max = lead_split.max() - lead_medium
    # gap_min = lead_medium - lead_split.min()
    #
    # if gap_max > gap_min:
    #     lead_split = data[split_lead_name].to_numpy()
    #     # height = np.percentile(lead_split, 85)
    #     peaks, properties = scipy.signal.find_peaks(
    #         lead_split,
    #         # rel_height=height,
    #         distance=30,
    #         prominence=200
    #     )
    # else:
    #     # neg_height = np.percentile(-lead_split, 85)
    #     peaks, properties = scipy.signal.find_peaks(
    #         -lead_split,
    #         # height=neg_height,
    #         distance=30,
    #         prominence=200
    #     )

    # # find heart beat cycle
    # prominences_norm = properties['prominences'] / properties['prominences'].max()
    # prominences_min = np.percentile(prominences_norm, 10)
    # prominences_max = np.percentile(prominences_norm, 90)
    # mask = (prominences_min < prominences_norm) & (prominences_norm < prominences_max)
    # tmp = prominences_norm[mask]
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(tmp.reshape(-1, 1))
    # mask = kmeans.predict(prominences_norm.reshape(-1, 1))
    # group1 = prominences_norm[mask == 1]
    # group2 = prominences_norm[mask == 0]
    #
    # # check difference among groups
    #
    # print(group1, group2)
    # print(mask)
    # if group1.size > 0 and group2.size > 0:
    #     tmp = np.asarray([group1.min(), group1.max(), group2.min(), group2.max()])
    #     tmp.sort()
    #     prominence_val = tmp[1: 3].mean()
    # elif group1.size > 0:
    #     prominence_val = group1.min()
    # elif group2.size > 0:
    #     prominence_val = group2.min()
    # else:
    #     raise ValueError("Can not find peaks.")
    # print(prominence_val)
    # print()
    # # print(group1, group2)
    # # if group1.min() > group2.max():
    # #     prominence_val = (group1.min() + group2.max()) / 2
    # # elif group2.min() > group1.max():
    # #     prominence_val = (group2.min() + group1.max()) / 2
    # # else:
    # #     prominence_val = 0.85 * np.percentile(prominences_norm, 70)
    # # print(111, prominences_norm, prominence_val)
    # peaks = peaks[prominences_norm >= prominence_val]
    # # prominence = np.percentile(properties['prominences'], 70)
    # # peaks = peaks[properties['prominences'] > 0.85 * prominence]
    # distance = 0.9 * np.percentile(np.diff(peaks), 50)
    #
    # # relocate peaks to get accurate distinct
    # peaks, properties = scipy.signal.find_peaks(
    #     lead_split,
    #     distance=distance,
    #     prominence=100
    # )
    # return peaks


def split_data(data: pd.DataFrame, peaks: Iterable[int], segment_length: float = 1.0):
    """
    Split data into small segment according to one lead
    """
    # split data according to peak indexes
    columns = data.columns
    ecg_segments = {}
    ecg_segments_pad = {}
    start_shift_default = 35
    end_shift_default = 100 - start_shift_default
    start_shift_adaptive = int(segment_length * start_shift_default)
    end_shift_adaptive = int(segment_length * end_shift_default)
    start_shift = min(start_shift_default, start_shift_adaptive)
    end_shift = min(end_shift_default, end_shift_adaptive)

    for column in columns:
        lead = data[column].to_numpy()
        lead_segments = []
        lead_segments_pad = []
        for index in peaks:

            if start_shift <= index < lead.size - end_shift:
                lead_segment_pad = np.zeros(100)
                start_index = start_shift_default - start_shift
                end_index = 100 - (end_shift_default - end_shift)
                lead_segment_pad[start_index: end_index] = lead[index - start_shift: index + end_shift]

                lead_segment = lead[index - start_shift: index + end_shift]
                lead_segments.append(lead_segment)
                lead_segments_pad.append(lead_segment_pad)
                # if True:
                #     plt.plot(lead_segment)
                #     plt.plot(lead[index - start_shift_default: index + end_shift_default])
                #     plt.show()

        if len(lead_segments) == 0:
            return None

        lead_segments = np.stack(lead_segments)
        ecg_segments[column] = lead_segments

        lead_segments_pad = np.stack(lead_segments_pad)
        ecg_segments_pad[column] = lead_segments_pad

        # print(lead_segments.shape)

    return ecg_segments, ecg_segments_pad

