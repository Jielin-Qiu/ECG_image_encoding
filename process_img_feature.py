import pandas as pd
import numpy as np
from scipy import signal
from typing import Iterable
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
import matplotlib.pyplot as plt
import cv2


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def fig_to_numpy(fig, ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.subplots_adjust(
        top=1.0,
        bottom=0.0,
        left=0.0,
        right=1.0,
        hspace=0.0,
        wspace=0.0
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def extract_image_feature_vanilla(ecg_segments: dict, columns: Iterable[str]):
    segment_num, segment_length = list(ecg_segments.values())[0].shape

    height = 1.1
    # create figure
    # fig, ax = plt.subplots()
    images = []
    fig, ax = plt.subplots(figsize=(3, 3))
    for i in range(segment_num):

        col_num = 4
        row_num = 3

        # get data of all 12 leads in each segment
        for j, column in enumerate(columns):
            lead_segment = ecg_segments[column][i]
            lead_segment_norm = lead_segment / (np.linalg.norm(lead_segment, np.inf) + 1e-8) / height * 0.5
            # lead_segment =
            x_shift = j // row_num * segment_length
            y_shift = -(j % row_num + 0.5) * height
            xs = np.asarray([i for i in range(len(lead_segment_norm))]) + x_shift
            ys = lead_segment_norm + y_shift
            ax.plot(xs, ys, linewidth=2.0)
            # ax.text(x_shift + 0.02 * segment_length, y_shift + 0.35 * height, column)

        # for i in range(1, col_num):
        #     ax.axvline(i * segment_length, color='blue')

        ax.set_xlim(0, col_num * segment_length - 1)
        ax.set_ylim(-height * row_num, 0)

        img = fig_to_numpy(fig, ax)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        images.append(img)

        # cv2.imshow('ekg', img)
        # cv2.waitKey(0)
        # print(img.shape)

        ax.clear()

    plt.close(fig)
    images = np.stack(images)
    return images


def signal_to_image(x, visualize_methods):
    x = x.reshape(1, -1)
    x_norm = x / (np.max(np.abs(x)) + 1e-8)
    img_blank = np.zeros((1, x.size, x.size))
    imgs = []

    visualize_methods += [None] * (3 - len(visualize_methods))

    for method in visualize_methods:

        if method == "gaf":
            gasf = GramianAngularField(sample_range=None, method='summation')
            img_gasf = gasf.fit_transform(x_norm)
            # print(img_gasf.min(), img_gasf.max(), x_norm.min(), x_norm.max())
            img_gasf = np.interp(img_gasf, [-1., 1.], [0., 255.])
            imgs.append(img_gasf)
        elif method == "rp":
            rp = RecurrencePlot()
            img_rp = rp.fit_transform(x_norm)
            # print(img_rp.min(), img_rp.max(), x_norm.min(), x_norm.max())
            img_rp = np.interp(img_rp, [0, 1], [0., 255.])
            imgs.append(img_rp)
        elif method == "mtf":
            mtf = MarkovTransitionField(n_bins=4)
            img_mtf = mtf.fit_transform(x_norm)
            # print(img_mtf.min(), img_mtf.max(), x_norm.min(), x_norm.max())
            img_mtf = np.interp(img_mtf, [0, 1], [0., 255.])
            imgs.append(img_mtf)
        else:
            imgs.append(img_blank)

    img = np.vstack(imgs)
    img = img.astype(np.uint8)
    img = np.moveaxis(img, 0, -1)
    return img


def extract_image_feature(
        ecg_segments: dict,
        columns: Iterable[str],
        visualize_style: str,
        visualize_methods: Iterable[str]
):
    visualize_style = visualize_style.lower()
    visualize_methods = [x.lower() for x in visualize_methods]
    assert visualize_style.lower() in ["grid", "concat"]
    assert all(x in ["gaf", "rp", "mtf"] for x in visualize_methods)
    assert 1 <= len(visualize_methods) <= 3
    if visualize_style == "grid":
        return extract_image_feature_grid(ecg_segments, columns, visualize_methods)
    else:
        return extract_image_feature_concat(ecg_segments, columns, visualize_methods)


def extract_image_feature_concat(ecg_segments: dict, columns: Iterable[str], visualize_methods: Iterable[str]):
    segment_num, segment_length = list(ecg_segments.values())[0].shape
    images = []
    for i in range(segment_num):
        all_lead_data = []
        for lead_name in columns:
            lead_data = ecg_segments[lead_name][i]
            lead_data_norm = lead_data / (np.max(np.abs(lead_data)) + 1e-8)
            all_lead_data.append(lead_data_norm)

        all_lead_data = np.concatenate(all_lead_data)
        img = signal_to_image(all_lead_data, visualize_methods)

        img = cv2.resize(img, (224, 224))
        images.append(img)
        
    images = np.stack(images)
    return images


def extract_image_feature_grid(ecg_segments: dict, columns: Iterable[str], visualize_methods: Iterable[str]):
    segment_num, segment_length = list(ecg_segments.values())[0].shape

    images = []
    for i in range(segment_num):

        col_num = 4
        row_num = 3

        lead_images = [[None for _ in range(col_num)] for _ in range(row_num)]

        # get data of all 12 leads in each segment
        for j, lead_name in enumerate(columns):
            col_idx = j // row_num
            row_idx = j % row_num
            lead_image = signal_to_image(ecg_segments[lead_name][i], visualize_methods)
            lead_images[row_idx][col_idx] = lead_image

        for row_idx in range(row_num):
            lead_images[row_idx] = cv2.hconcat(lead_images[row_idx])

        img = cv2.vconcat(lead_images)
        img = cv2.resize(img, (224, 224))
        images.append(img)

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

    images = np.stack(images)
    return images


