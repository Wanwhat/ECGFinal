import os

import mat73
import numpy as np
from tqdm import tqdm

from utils import baseline_remove, resample, min_max_normalize


def label_convert(i):
    if 0 <= i < 500:
        return 1
    elif 500 <= i < 1000:
        return 0
    return -1


if __name__ == '__main__':
    train_data_path = './CPSC2025/traindata.mat'
    train_data = mat73.loadmat(train_data_path)['traindata']

    segments, labels = [], []
    for i in tqdm(range(1000)):
        ecg_data = resample(train_data[i], sample_rate=400, resample_rate=300)
        ecg_data = baseline_remove(ecg_data, 300)
        ecg_data = min_max_normalize(ecg_data)

        label = label_convert(i)

        segments.append(ecg_data)
        labels.append(label)

    segments = np.array(segments)
    labels = np.array(labels)

    labeled_path = './data_labeled/'
    if not os.path.exists(labeled_path):
        os.makedirs(labeled_path, exist_ok=True)

    np.save(os.path.join(labeled_path, 'segments.npy'), segments, allow_pickle=True)
    np.save(os.path.join(labeled_path, 'labels.npy'), labels, allow_pickle=True)
