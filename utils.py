import numpy as np
import os
import random
import torch
from scipy.signal import butter, lfilter, medfilt, iirnotch, filtfilt
import scipy.signal as signal


def load_data(npy_file):
    file = np.load(npy_file, allow_pickle=True)[()]
    data = file['data']
    label = file['label']

    return data, label

def get_class_weights(dataset):
    class_counts = np.zeros(2, dtype=int)
    for _, label in dataset:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    return class_weights

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def min_max_normalize(ecg_signal):
    min_val = np.min(ecg_signal)
    max_val = np.max(ecg_signal)

    normalized_signal = (ecg_signal - min_val) / (max_val - min_val)

    return normalized_signal


def resample(data, sample_rate, resample_rate):
    number_of_samples = round(len(data) * float(resample_rate) / sample_rate)
    data = signal.resample(data, number_of_samples)

    return data


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)

    return y


def butter_lowpass_filter(data, lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, highcut, fs, order=1):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    y = lfilter(b, a, data)
    return y


def baseline_remove(data, fs):
    baseline = medfilt(medfilt(data, int(0.2 * fs) - 1), int(0.6 * fs) - 1)
    y = data - baseline
    return y


def powerline_remove(data, fs):
    f0 = 50
    Q = 30
    w0 = f0 / (fs / 2)

    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, data)

    return y


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='./best-model.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
