#!/usr/bin/env python
# coding: utf-8
import os
import cv2
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
from process_img_feature import extract_image_feature


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


def down_sample_data(data: pd.DataFrame, sample_num=1000):
    columns = data.columns
    data_down_sample = {}
    for column in columns:
        lead_data = data[column].to_numpy()
        lead_down_sample = scipy.signal.resample(lead_data, sample_num)
        data_down_sample[column] = lead_down_sample
    return pd.DataFrame.from_dict(data_down_sample)


def main(config, cpu=2):
    data_dir = config["data_dir"]
    result_dir = config["result_dir"]

    # process labels of the data
    if "label_file" in config:
        df = pd.read_csv(config["label_file"])

        if config.get("label_option", "").lower() == 'unique':
            # for ECG with multiple labels, use the first one
            df['label'] = df['label'].str.split(" ")
            df['label'] = df['label'].str.get(0)
        elif config.get("label_option", "").upper() == 'MI':
            df['label'] = df['label'].str.contains("MI")
            df['label'] = df['label'].map({True: "MI", False: "NOT_MI"})

        # df.to_csv(kwargs["label_file_save"], index=False)

    assert "label_file" in config, "Must provide label file"
    df = pd.read_csv(config["label_file"])
    df_mask = None

    for used_label in config.get("used_labels", []):
        mask = df['label'] == used_label
        if df_mask is None:
            df_mask = mask
        else:
            df_mask = df_mask & mask

    for unused_label in config.get("unused_labels", []):
        mask = df['label'] != unused_label
        if df_mask is None:
            df_mask = mask
        else:
            df_mask = df_mask & mask

    if df_mask is None:
        df1 = df
    else:
        df1 = df[df_mask]

    # df1 = df1.head(10)

    all_file_names = df1['file_name'].to_numpy().tolist()
    all_diagnosis = df1['diagnosis_en_simplified'].to_numpy().tolist()

    csv_files = list(map(lambda x: f'{x}.csv', all_file_names))

    bar = tqdm(total=len(csv_files))
    train_idx = -1
    valid_idx = -1
    test_idx = -1
    all_idx = -1
    train_img_ids = []
    valid_img_ids = []
    test_img_ids = []
    train_sentences = []
    valid_sentences = []
    test_sentences = []

    os.makedirs(os.path.join(result_dir, 'images'), exist_ok=True)

    processed_id = 0
    worker_results = {}

    normal_num = 0
    abnormal_num = 0

    def process_results(x):
        nonlocal train_idx, valid_idx, test_idx, all_idx, normal_num, abnormal_num, all_diagnosis
        worker_idx, set_type, images = x
        diagnosis = all_diagnosis[worker_idx]
        file_name = all_file_names[worker_idx]

        for image in images:
            assert set_type in ['train', 'valid', 'test']
            all_idx += 1
            img_id = f'image_{all_idx}.jpg'
            img_path = os.path.join(result_dir, 'images', img_id)
            cv2.imwrite(img_path, image)

            if set_type == 'train':
                train_idx += 1
                train_img_ids.append(img_id)
                train_sentences.append(diagnosis)
            elif set_type == 'valid':
                valid_idx += 1
                valid_img_ids.append(img_id)
                valid_sentences.append(diagnosis)
            elif set_type == 'test':
                test_idx += 1
                test_img_ids.append(img_id)
                test_sentences.append(diagnosis)
            else:
                raise ValueError

    def gather_results(x):
        nonlocal processed_id
        bar.update()
        worker_idx = x[0]
        worker_results[worker_idx] = x
        while processed_id in worker_results:
            x = worker_results[processed_id]
            process_results(x)
            del worker_results[processed_id]
            processed_id += 1

    if cpu > 1:
        pool = mp.Pool()
        for index in range(len(csv_files)):
            pool.apply_async(ecg_worker, args=(config, index, csv_files[index]), callback=gather_results)
        pool.close()
        pool.join()
    else:
        for index in range(len(csv_files)):
            # if csv_files[index] == 'PTB-XL-00184.csv':
            returns = ecg_worker(config, index, csv_files[index])
            gather_results(returns)

    # print("Normal ECG:", normal_num)
    # print("Abnormal ECG:", abnormal_num)

    # save txt files
    with open(os.path.join(result_dir, 'train_img_ids.txt'), 'w') as f:
        text = '\n'.join(train_img_ids)
        f.write(text)
    with open(os.path.join(result_dir, 'valid_img_ids.txt'), 'w') as f:
        text = '\n'.join(valid_img_ids)
        f.write(text)
    with open(os.path.join(result_dir, 'test_img_ids.txt'), 'w') as f:
        text = '\n'.join(test_img_ids)
        f.write(text)

    with open(os.path.join(result_dir, 'train_sentence.txt'), 'w') as f:
        text = '\n'.join(train_sentences)
        f.write(text)
    with open(os.path.join(result_dir, 'valid_sentence.txt'), 'w') as f:
        text = '\n'.join(valid_sentences)
        f.write(text)
    with open(os.path.join(result_dir, 'test_sentence.txt'), 'w') as f:
        text = '\n'.join(test_sentences)
        f.write(text)


def ecg_worker(config, index, csv_file):
    np.random.seed(index)

    plot_limit = 200
    data_dir = config["data_dir"]
    xml_dir = os.path.join(data_dir, "xml")
    csv_dir = os.path.join(data_dir, "csv")
    plot_dir = config["plot_dir"]
    adaptive_segmentation = config["adaptive_segmentation"]
    visualize_style = config["visualize_style"]
    visualize_methods = config["visualize_methods"]

    file_name = csv_file[:csv_file.index('.')]
    xml_file = file_name + '.xml'

    # load the decoded csv file
    csv_path = os.path.join(csv_dir, csv_file)
    assert os.path.isfile(csv_path), f"Can not find file {csv_path}"
    data_origin = pd.read_csv(csv_path) / 1000

    # get sampling rate of the data
    if "sampling_rate" in config:
        sampling_rate = config["sampling_rate"]
    else:
        # get sample frequency from the xml file
        xml_path = os.path.join(xml_dir, xml_file)
        assert os.path.isfile(xml_path), f"Can not find file {xml_path}"
        sampling_rate = get_sampling_rate(xml_path)
        assert sampling_rate > 0, f"Can not find sample frequency in {xml_path}"

    # plot original lead data
    if config["plot_lead_origin"] and index < plot_limit:
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
    peaks, heart_rate = get_r_peaks(data, config["split_lead_name"])
    if adaptive_segmentation:
        segment_length = 1.0 / heart_rate
    else:
        segment_length = 1.0

    # remove noise with low pass filter
    data = butter_lowpass(data, cutoff=15, fs=down_sampling_rate, order=2)

    # plot peak in lead II
    if config["plot_lead_peak"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_peak")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_r_peaks(data, down_sampling_rate, config["split_lead_name"], peaks, img_path)

    # plot lead segments according to peaks
    if config["plot_lead_processed"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_processed")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_segments(data, down_sampling_rate, peaks, img_path, length=1.0)

    # plot lead segments according to peaks
    if config["plot_lead_processed_adaptive"] and index < plot_limit:
        img_dir = os.path.join(plot_dir, "lead_processed_adaptive")
        img_path = os.path.join(img_dir, file_name + ".png")
        os.makedirs(img_dir, exist_ok=True)
        plot_segments(data, down_sampling_rate, peaks, img_path, length=1.0 / heart_rate)

    # split data according to r peaks in lead II
    ecg_segments, ecg_segments_pad = split_data(data, peaks, segment_length=segment_length)
    if ecg_segments is None:
        print(f"No valid peaks in {file_name}, ignore file data.")
        return

    # get image description
    columns = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    segment_num, segment_length = list(ecg_segments_pad.values())[0].shape

    # only keep one segments:
    ecg_segments_pad_single = {}
    idx = np.random.choice(segment_num)
    for k, v in ecg_segments_pad.items():
        ecg_segments_pad_single[k] = v[idx: idx+1]

    images = extract_image_feature(ecg_segments_pad_single, columns, visualize_style, visualize_methods)

    # only select one image
    num = np.random.rand()
    if num <= 0.7:
        # training set
        set_type = 'train'
    elif num <= 0.9:
        # validation set
        set_type = 'valid'
    else:
        # test set
        set_type = 'test'

    # set_type = 'train'

    # return images and diagnosis
    return index, set_type, images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', default=False, action='store_true', help='use adaptive segmentation')
    parser.add_argument('--plot', default=False, action='store_true', help='save plots.')
    parser.add_argument('--style', type=str, default='grid', help='choose the visualization style.')
    parser.add_argument('--method', type=str, default="gaf,rp,mtf", help='choose the visualization method.')
    parser.add_argument('--output_path', type=str, default="data/output/abnormalPTB-XL_Grid",
                        help='directory for saving the output images')
    args = parser.parse_args()

    visualize_methods = args.method
    visualize_methods = visualize_methods.replace(" ", "")
    visualize_methods = visualize_methods.split(",")

    ptb_abnormal_config = dict(
        name="PTB-XL",
        data_dir="data/decoded/PTB-XL",
        plot_dir="plot/abnormalPTB-XL",
        result_dir=args.output_path,
        plot_lead_origin=args.plot,
        plot_lead_peak=args.plot,
        plot_lead_processed=args.plot,
        plot_lead_processed_adaptive=args.plot,
        split_lead_name="II",
        sampling_rate=100,
        label_file="data/decoded/PTB-XL/label.csv",
        used_labels=[],
        unused_labels=["NORM"],
        adaptive_segmentation=args.adaptive,
        visualize_style="Concat",
        visualize_methods=visualize_methods,
        # visualize_methods=["rp", "gaf"],
    )

    main(ptb_abnormal_config, cpu=16)



