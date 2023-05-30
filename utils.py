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
from itertools import repeat
import sys


class Logger(object):
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


def split_data_label(data, labels, sub_length, overlapping):
    ''' split a batch of time-series trials into shorter samples and adding trial ids to the labels

    Args:
        data (numpy.ndarray): It should have a shape of (n_trials, n_timestamps, n_features) B x T x C
        labels (numpy.ndarray): It should have a shape of (n_trials, 2). The first column is the label and the second column is patient ID.
        sub_length (int): The length for sample-level data.
        overlapping (float): How many overlapping for each sample-level data in a trial

    Returns:
        sample_data (numpy.ndarray): B_sub x T_sub x C. B_sub = B x segments_num
        sample_labels (numpy.ndarray): B_sub x 3. (label, patient id, trial id)
    '''
    sample_data, trial_ids, sample_num = split_data(data, sub_length, overlapping)
    # all samples from same trial should have same label and patient id
    sample_labels = np.repeat(labels, repeats=sample_num, axis=0)
    # append trial ids. Segments split from same trial should have same trial ids
    label_num = sample_labels.shape[0]
    sample_labels = np.hstack((sample_labels.reshape((label_num, -1)), trial_ids.reshape((label_num, -1))))
    sample_data, sample_labels = shuffle(sample_data, sample_labels, random_state=42)
    return sample_data, sample_labels


def split_data(data, sub_length=256, overlapping=0.5):
    ''' split a batch of time-series into shorter samples and mark their trial ids
    Returns:
        sample_data (numpy.ndarray): (n_samples, n_sub_timestamps, n_features). n_samples = n_trials x sample_num
        trial_ids (numpy.ndarray): (n_samples,)
        sample_num (int): one trial splits into sample_num of samples
    '''
    length = data.shape[1]
    # check if sub_length and overlapping compatible
    if overlapping:
        assert (length - (1-overlapping)*sub_length) % (sub_length*overlapping) == 0
        sample_num = (length - (1 - overlapping) * sub_length) / (sub_length * overlapping)
    else:
        assert length % sub_length == 0
        sample_num = length / sub_length
    sample_feature_list = []
    trial_id_list = []
    trial_id = 1
    for trial in data:
        counter = 0
        # split one trial(5s, 1280 timestamps) into 9 half overlapping samples (1s, 256 timestamps)
        while counter*sub_length*(1-overlapping)+sub_length <= trial.shape[0]:
            sample_feature = trial[int(counter*sub_length*(1-overlapping)):int(counter*sub_length*(1-overlapping)+sub_length)]
            # print(f"{int(counter*length*(1-overlapping))}:{int(counter*length*(1-overlapping)+length)}")
            sample_feature_list.append(sample_feature)
            trial_id_list.append(trial_id)
            counter += 1
        trial_id += 1
    sample_data, trial_ids = np.array(sample_feature_list), np.array(trial_id_list)

    return sample_data, trial_ids, sample_num


# t: time interval
# data: time-series data in shape TxF. T is time sequence and F is feature\channel.
def plot_channels(t, data):
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
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    # y = filtfilt(b, a, data, axis=0)
    return y


# trial(numpy.ndarray): Shape in timestamps x channels
def process_trial(trial, normalized=True, bandpass_filter=False):
    if bandpass_filter:
        trial = butter_bandpass_filter(trial, 0.5, 50, 256, 5)
    if normalized:
        scaler = StandardScaler()
        scaler.fit(trial)
        trial = scaler.transform(trial)
    return trial


# batch(numpy.ndarray): Shape in batch x timestamps x channels
def process_batch_trial(batch, normalized=True, bandpass_filter=False):
    bool_iterator_1 = repeat(normalized, len(batch))
    bool_iterator_2 = repeat(bandpass_filter, len(batch))
    return np.array(list(map(process_trial, batch, bool_iterator_1, bool_iterator_2)))


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)


def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

