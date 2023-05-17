#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import pywt
import multiprocessing as mp
from tqdm import tqdm
from utilities import get_sampling_rate, plot_all_leads, plot_r_peaks, plot_segments, plot_segments_prediction
from utilities import get_r_peaks, split_data
from process_signal import extract_signal_feature
from process_time_feature import extract_time_feature
from process_freq_feature import extract_freq_feature


def butter_highpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = scipy.signal.butter(order, normal_cutoff, btype='high', output='sos')
    columns = data.columns
    filtered_sample = {}
    for column in columns:
        lead_data = data[column].to_numpy()
        lead_highpass = scipy.signal.sosfiltfilt(b, lead_data, padtype='even', padlen=200)
        filtered_sample[column] = lead_highpass
    return pd.DataFrame.from_dict(filtered_sample)


def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = scipy.signal.butter(order, normal_cutoff, btype='low', output='sos')
    columns = data.columns
    filtered_sample = {}
    for column in columns:
        lead_data = data[column].to_numpy()
        lead_highpass = scipy.signal.sosfiltfilt(b, lead_data, padtype='even', padlen=200)
        filtered_sample[column] = lead_highpass
    return pd.DataFrame.from_dict(filtered_sample)


# def fri_lowpass(data, cutoff, fs, order=15):
#     numtaps, beta = scipy.signal.kaiserord(65, order/(0.5*fs))
#     b = scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta), scale=False, nyq=0.5*fs)
#     columns = data.columns
#     filtered_sample = {}
#     for column in columns:
#         lead_data = data[column].to_numpy()
#         lead_highpass = scipy.signal.filtfilt(b, 1, lead_data)
#         filtered_sample[column] = lead_highpass
#     return pd.DataFrame.from_dict(filtered_sample)


# def denoise(data):
#     columns = data.columns
#     filtered_sample = {}
#     for column in columns:
#         lead_data = data[column].to_numpy()
#         lead_highpass = denoise_signal(lead_data, 'db8', 6)
#         filtered_sample[column] = lead_highpass
#     return pd.DataFrame.from_dict(filtered_sample)


# def denoise_signal(X, dwt_transform, dlevels):
#     coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)  # wavelet transform 'bior4.4'
#     # scale 0 to cutoff_low
#     threshold = 0.1
#     for i in range(1, len(coeffs)):
#         coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
#         # if i >= len(coeffs)-1:
#         #     coeffs[i] *= 0
#     # # # scale cutoff_high to end
#     # for ca in range(cutoff_high, len(coeffs)):
#     #     coeffs[ca] = np.multiply(coeffs[ca], [0.0])
#     Y = pywt.waverec(coeffs, dwt_transform)  # inverse wavelet transform
#     return Y


def down_sample_data(data: pd.DataFrame, sample_num=1000):
    columns = data.columns
    data_down_sample = {}
    for column in columns:
        lead_data = data[column].to_numpy()
        lead_down_sample = scipy.signal.resample(lead_data, sample_num)
        data_down_sample[column] = lead_down_sample
    return pd.DataFrame.from_dict(data_down_sample)


# def smooth_data(data: pd.DataFrame, kernel_size=7):
#     columns = data.columns
#     data_smooth = {}
#     kernel = np.hamming(kernel_size)
#     for column in columns:
#         lead_data = data[column].to_numpy()
#         lead_smooth = np.convolve(lead_data, kernel, mode="same")
#         data_smooth[column] = lead_smooth
#     return pd.DataFrame.from_dict(data_smooth)


def main(kwargs, cpu=2):
    data_dir = kwargs["data_dir"]
    result_dir = kwargs["result_dir"]
    adaptive_segmentation = kwargs["adaptive_segmentation"]
    if adaptive_segmentation:
        save_dir = os.path.join(result_dir, "adaptive")
    else:
        save_dir = os.path.join(result_dir, "fix")
    os.makedirs(save_dir, exist_ok=True)

    # process labels of the data
    if "label_file" in kwargs:
        df = pd.read_csv(kwargs["label_file"])

        if kwargs.get("label_option", "").lower() == 'unique':
            # for ECG with multiple labels, use the first one
            df['label'] = df['label'].str.split(" ")
            df['label'] = df['label'].str.get(0)
        elif kwargs.get("label_option", "").upper() == 'MI':
            df['label'] = df['label'].str.contains("MI")
            df['label'] = df['label'].map({True: "MI", False: "NOT_MI"})

        df.to_csv(kwargs["label_file_save"], index=False)

    csv_dir = os.path.join(data_dir, "csv")

    csv_files = os.listdir(csv_dir)
    csv_files.sort()

    bar = tqdm(total=len(csv_files))
    update = lambda _: bar.update()

    if cpu > 1:
        pool = mp.Pool()
        for index in range(len(csv_files)):
            pool.apply_async(ecg_worker, args=(kwargs, index, csv_files[index]), callback=update)
        pool.close()
        pool.join()

    else:
        for index in range(len(csv_files)):
            ecg_worker(kwargs, index, csv_files[index])
            update(0)


def ecg_worker(kwargs, index, csv_file):
    plot_limit = 200
    data_dir = kwargs["data_dir"]
    xml_dir = os.path.join(data_dir, "xml")
    csv_dir = os.path.join(data_dir, "csv")
    plot_dir = kwargs["plot_dir"]
    result_dir = kwargs["result_dir"]
    adaptive_segmentation = kwargs["adaptive_segmentation"]

    file_name = csv_file[:-4]
    xml_file = file_name + '.xml'

    # load the decoded csv file
    csv_path = os.path.join(csv_dir, csv_file)
    assert os.path.isfile(csv_path), f"Can not find file {csv_path}"
    data_origin = pd.read_csv(csv_path) / 1000

    # get sampling rate of the data
    if "sampling_rate" in kwargs:
        sampling_rate = kwargs["sampling_rate"]
    else:
        # get sample frequency from the xml file
        xml_path = os.path.join(xml_dir, xml_file)
        assert os.path.isfile(xml_path), f"Can not find file {xml_path}"
        sampling_rate = get_sampling_rate(xml_path)
        assert sampling_rate > 0, f"Can not find sample frequency in {xml_path}"

    # plot original lead data
    if kwargs["plot_lead_origin"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_original")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_all_leads(data_origin, sampling_rate, img_path)

    # down sample data to 100 Hz
    down_sampling_rate = 100
    if sampling_rate != down_sampling_rate:
        data = down_sample_data(data_origin, sample_num=1000)
    else:
        data = data_origin

    # remove baseline wander with highpass filter
    data = butter_highpass(data, cutoff=0.5, fs=down_sampling_rate, order=2)

    # locate peak in lead II
    peaks, heart_rate = get_r_peaks(data, kwargs["split_lead_name"])
    if adaptive_segmentation:
        segment_length = 1.0 / heart_rate
    else:
        segment_length = 1.0

    # remove noise with low pass filter
    data = butter_lowpass(data, cutoff=15, fs=down_sampling_rate, order=2)

    # plot peak in lead II
    if kwargs["plot_lead_peak"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_peak")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_r_peaks(data, down_sampling_rate, kwargs["split_lead_name"], peaks, img_path)

    # plot lead segments according to peaks
    if kwargs["plot_lead_processed"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_processed")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_segments(data, down_sampling_rate, peaks, img_path, length=1.0)

    # plot lead segments according to peaks
    if kwargs["plot_lead_processed_adaptive"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_processed_adaptive")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_segments(data, down_sampling_rate, peaks, img_path, length=1.0 / heart_rate)

    # split data according to r peaks in lead II
    ecg_segments, ecg_segments_pad = split_data(data, peaks, segment_length=segment_length)
    if ecg_segments is None:
        print(f"No valid peaks in {file_name}, ignore file data.")
        return

    # extract signal feature_data
    columns = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal_feature = extract_signal_feature(ecg_segments_pad, columns)
    # print(signal_feature.shape)

    # extract time domain feature_data
    time_feature = extract_time_feature(ecg_segments, columns)
    # print(time_feature.shape)

    # extract frequency domain feature_data
    freq_feature = extract_freq_feature(ecg_segments, columns)

    # save result to disk
    data_extracted_no_label = np.hstack([signal_feature, time_feature, freq_feature])
    np.nan_to_num(data_extracted_no_label, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if adaptive_segmentation:
        save_dir = os.path.join(result_dir, "adaptive")
    else:
        save_dir = os.path.join(result_dir, "fix")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, file_name), data_extracted_no_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cad', help='name of the dataset')
    parser.add_argument('--adaptive', default=False, action='store_true', help='use adaptive segmentation')
    parser.add_argument('--plot', default=False, action='store_true', help='save plots.')
    args = parser.parse_args()

    ptb_config = dict(
        name="PTB-XL",
        data_dir="data/decoded/PTB-XL",
        plot_dir="plot/PTB-XL",
        result_dir="feature_data/PTB-XL",
        plot_lead_origin=True and args.plot,
        plot_lead_peak=True and args.plot,
        plot_lead_processed=True and args.plot,
        plot_lead_processed_adaptive=True and args.plot,
        split_lead_name="II",
        sampling_rate=100,
        label_file="data/decoded/PTB-XL/label.csv",
        label_option="unique",
        label_file_save="feature_data/PTB-XL/label.csv",
        adaptive_segmentation=args.adaptive,
    )

    sa_config = dict(
        name="SA",
        data_dir="data/decoded/SA",
        plot_dir="plot/SA",
        result_dir="feature_data/SA",
        plot_lead_origin=True and args.plot,
        plot_lead_peak=True and args.plot,
        plot_lead_processed=True and args.plot,
        plot_lead_processed_adaptive=True and args.plot,
        split_lead_name="II",
        adaptive_segmentation=args.adaptive
    )

    cad_config = dict(
        name="CAD",
        data_dir="data/decoded/CAD",
        plot_dir="plot/CAD",
        result_dir="feature_data/CAD",
        plot_lead_origin=True and args.plot,
        plot_lead_peak=True and args.plot,
        plot_lead_processed=True and args.plot,
        plot_lead_processed_adaptive=True and args.plot,
        split_lead_name="II",
        label_file="data/decoded/CAD/label.csv",
        label_file_save="feature_data/CAD/label.csv",
        adaptive_segmentation=args.adaptive
    )

    if args.dataset == 'ptbxl':
        print("Extreact feature_data from the PTB-XL database.")
        main(ptb_config)
    elif args.dataset == 'sa':
        print("Extreact feature_data from the SA database.")
        main(sa_config)
    elif args.dataset == 'cad':
        print("Extreact feature_data from the CAD database.")
        main(cad_config)
    else:
        print(f"{args.dataset} is not a valid name for dataset")
