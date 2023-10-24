import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from scipy.signal import butter, lfilter, filtfilt
import itertools
from itertools import repeat
import sys
from torch.utils.data import BatchSampler


class Logger(object):
    """ A Logger for saving output of printings between functions start_logging() and stop_logging().

    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def start_logging(random_seed, saving_directory):
    log_filename = f"log_{random_seed}.txt"
    log_filepath = os.path.join(saving_directory, log_filename)
    sys.stdout = Logger(log_filepath)


def stop_logging():
    sys.stdout = sys.__stdout__


class MyBatchSampler(BatchSampler):
    """ A custom BatchSampler to shuffle the samples within each batch.
        It changes the local order of samples(samples in the same batch) per epoch,
        which does not break too much the distribution of pre-shuffled samples by function shuffle_feature_label().
        The goal is to shuffle the samples per epoch but make sure that there are samples from the same trial in a batch.

    """
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            random.shuffle(batch)
            yield batch


def shuffle_feature_label(X, y, shuffle_function='trial', batch_size=128):
    """ Call shuffle functions.
        The goal is to guarantee that there are samples from the same trial in a batch,
        while avoiding all the samples are from the same trial/patient (low diversity).

    Args:
        shuffle_function (str): specify the shuffle function
        batch_size (int): batch_size if apply batch shuffle
    """

    # do trial shuffle
    if shuffle_function == 'trial':
        return trial_shuffle_feature_label(X, y)

    # do batch shuffle
    elif shuffle_function == 'batch':
        return batch_shuffle_feature_label(X, y, batch_size)

    # do random shuffle
    elif shuffle_function == 'random':
        return shuffle(X, y, random_state=42)

    else:
        # print(shuffle_function)
        raise ValueError(f'\'{shuffle_function}\' is a wrong argument for shuffle function!')


def trial_shuffle_feature_label(X, y):
    """ shuffle each samples in a trial first, then shuffle the order of trials

    """

    # sort X, y by trial ID
    sorted_indices = np.argsort(y[:, 2], axis=0)
    # concatenate sorted indices and labels
    sorted_indices_labels = np.concatenate((sorted_indices.reshape(-1, 1), y[sorted_indices]), axis=1).astype(int)
    trials_list = []
    # group each trial by trial ID
    for _, trial in itertools.groupby(sorted_indices_labels, lambda x: x[3]):
        trial = list(trial)
        # shuffle each sample in a trial
        trial = shuffle(trial, random_state=42)
        trials_list.append(trial)
    # shuffle the order of trials
    shuffled_trials_list = shuffle(trials_list, random_state=42)
    shuffled_trials = np.concatenate(shuffled_trials_list, axis=0)
    # get the sorted indices
    shuffled_sorted_indices = shuffled_trials[:, 0]
    X_shuffled = X[shuffled_sorted_indices]
    y_shuffled = y[shuffled_sorted_indices]
    return X_shuffled, y_shuffled


def batch_shuffle_feature_label(X, y, batch_size=128):
    """ shuffle the order of batches first, then shuffle the samples in the batch

    """

    # sort X, y by trial ID
    sorted_indices = np.argsort(y[:, 2], axis=0)
    sorted_indices_list = np.array_split(sorted_indices, y.shape[0]/batch_size)
    # shuffle the batches
    sorted_indices_list = shuffle(sorted_indices_list, random_state=42)
    # shuffle samples in the batch
    shuffled_sorted_indices_list = []
    for batch in sorted_indices_list:
        shuffled_batch = shuffle(batch, random_state=42)
        shuffled_sorted_indices_list.append(shuffled_batch)
    shuffled_sorted_indices = np.concatenate(shuffled_sorted_indices_list, axis=0)
    X_shuffled = X[shuffled_sorted_indices]
    y_shuffled = y[shuffled_sorted_indices]
    return X_shuffled, y_shuffled


def split_data_label(X_trial, y_trial, sample_timestamps, overlapping):
    """ split a batch of time-series trials into samples and adding trial ids to the label array y

    Args:
        X_trial (numpy.ndarray): It should have a shape of (n_trials, trial_timestamps, features) B_trial x T_trial x C.
        y_trial (numpy.ndarray): It should have a shape of (n_trials, 2). The first column is the label and the second column is patient id.
        sample_timestamps (int): The length for sample-level data (T_sample).
        overlapping (float): How many overlapping for each sample-level data in a trial.

    Returns:
        X_sample (numpy.ndarray): It should have a shape of (n_samples, sample_timestamps, features) B_sample x T_sample x C. The B_sample = B x sample_num.
        y_sample (numpy.ndarray): It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
    """
    X_sample, trial_ids, sample_num = split_data(X_trial, sample_timestamps, overlapping)
    # all samples from same trial should have same label and patient id
    y_sample = np.repeat(y_trial, repeats=sample_num, axis=0)
    # append trial ids. Segments split from same trial should have same trial ids
    label_num = y_sample.shape[0]
    y_sample = np.hstack((y_sample.reshape((label_num, -1)), trial_ids.reshape((label_num, -1))))
    X_sample, y_sample = shuffle(X_sample, y_sample, random_state=42)
    return X_sample, y_sample


def split_data(X_trial, sample_timestamps=256, overlapping=0.5):
    """ split a batch of trials into samples and mark their trial ids

    Args:
        See split_data_label() function

    Returns:
        X_sample (numpy.ndarray): (n_samples, sample_timestamps, feature).
        trial_ids (numpy.ndarray): (n_samples,)
        sample_num (int): one trial splits into sample_num of samples
    """
    length = X_trial.shape[1]
    # check if sub_length and overlapping compatible
    if overlapping:
        assert (length - (1-overlapping)*sample_timestamps) % (sample_timestamps*overlapping) == 0
        sample_num = (length - (1 - overlapping) * sample_timestamps) / (sample_timestamps * overlapping)
    else:
        assert length % sample_timestamps == 0
        sample_num = length / sample_timestamps
    sample_feature_list = []
    trial_id_list = []
    trial_id = 1
    for trial in X_trial:
        counter = 0
        # ex. split one trial(5s, 1280 timestamps) into 9 half-overlapping samples (1s, 256 timestamps)
        while counter*sample_timestamps*(1-overlapping)+sample_timestamps <= trial.shape[0]:
            sample_feature = trial[int(counter*sample_timestamps*(1-overlapping)):int(counter*sample_timestamps*(1-overlapping)+sample_timestamps)]
            # print(f"{int(counter*length*(1-overlapping))}:{int(counter*length*(1-overlapping)+length)}")
            sample_feature_list.append(sample_feature)
            trial_id_list.append(trial_id)
            counter += 1
        trial_id += 1
    X_sample, trial_ids = np.array(sample_feature_list), np.array(trial_id_list)

    return X_sample, trial_ids, sample_num


def plot_channels(t, data):
    """ plot a time-series sample

    Args:
        t (float): The time range (in second) on x axis.
        data (numpy.ndarray): The plot time-series in shape (timestamps, feature).
    """

    data = data.reshape(data.shape[0], -1)
    timestamps = np.arange(0, t, t/data.shape[0])
    plt.figure(figsize=(12, 8))
    for i in range(data.shape[1]):
        plt.plot(timestamps, data[:, i], label="Channel"+str(i+1))
    plt.legend()
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ seed https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def process_ts(ts, fs, normalized=True, bandpass_filter=False):
    """ preprocess a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).
        fs (float): The sampling frequency for bandpass filtering.
        normalized (bool): Whether to normalize the time-series data.
        bandpass_filter (bool): Whether to filter the time-series data.

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """

    if bandpass_filter:
        ts = butter_bandpass_filter(ts, 0.5, 50, fs, 5)
    if normalized:
        scaler = StandardScaler()
        scaler.fit(ts)
        ts = scaler.transform(ts)
    return ts


def process_batch_ts(batch, fs=256, normalized=True, bandpass_filter=False):
    """ preprocess a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    """

    bool_iterator_1 = repeat(fs, len(batch))
    bool_iterator_2 = repeat(normalized, len(batch))
    bool_iterator_3 = repeat(bandpass_filter, len(batch))
    return np.array(list(map(process_ts, batch, bool_iterator_1, bool_iterator_2, bool_iterator_3)))


def seed_everything(seed=42):
    """
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # training is extremely slow when do following setting
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

