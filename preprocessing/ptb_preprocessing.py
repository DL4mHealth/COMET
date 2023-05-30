from sklearn.utils import shuffle
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_ptb(train_ids, val_ids, test_ids, data_path, label_path):
    ''' load ptb data

    Args:
        train_ids (list): list of ids for training set
        val_ids (list): list of ids for validation set
        test_ids (list): list of ids for test set
        data_path (str): path for data directory
        label_path (str): path for label file

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    '''
    train_trial_feature_list = []
    train_trial_label_list = []
    val_trial_feature_list = []
    val_trial_label_list = []
    test_trial_feature_list = []
    test_trial_label_list = []
    filenames = []
    # The first column is the label; the second column is the patient ID
    subject_label = np.load(label_path)
    for filename in os.listdir(data_path):
        filenames.append(filename)
    filenames.sort()
    # print(filenames[:5])
    print("Validation subjects: ", val_ids)
    print("Test subjects: ", test_ids)
    for j in range(len(filenames)):
        # print(j)
        trial_label = subject_label[j]
        path = data_path + filenames[j]
        subject_feature = np.load(path)
        for trial_feature in subject_feature:
            # val set
            if j+1 in val_ids:  # id starts from 1, not 0.
                val_trial_feature_list.append(trial_feature)
                val_trial_label_list.append(trial_label)
            # test set
            elif j+1 in test_ids:
                test_trial_feature_list.append(trial_feature)
                test_trial_label_list.append(trial_label)
            # training set
            elif j+1 in train_ids:
                train_trial_feature_list.append(trial_feature)
                train_trial_label_list.append(trial_label)
    # reshape and shuffle
    X_train = np.array(train_trial_feature_list)
    X_val = np.array(val_trial_feature_list)
    X_test = np.array(test_trial_feature_list)
    y_train = np.array(train_trial_label_list)
    y_val = np.array(val_trial_label_list)
    y_test = np.array(test_trial_label_list)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


"""def load_ptb(drop_ids, data_path, label_path):
    ''' load ptb data
    Args:
        drop_ids (list): id for subjects we drop
        data_path (str): path for data directory
        label_path (str): path for label file

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    '''
    trial_feature_list = []
    trial_label_list = []
    filenames = []
    # The first column is the label; the second column is the patient ID
    subject_label = np.load(label_path)
    for filename in os.listdir(data_path):
        filenames.append(filename)
    filenames.sort()
    print(filenames)
    quit()
    for j in range(len(filenames)):
        # drop some healthy subjects to keep balance
        if j+1 in drop_ids:
            continue
        # print(j)
        trial_label = subject_label[j]
        path = data_path + filenames[j]
        subject_feature = np.load(path)
        for trial_feature in subject_feature:
            trial_feature_list.append(trial_feature)
            trial_label_list.append(trial_label)

    # 60 : 20 : 20
    X_train, y_train = np.array(trial_feature_list), np.array(trial_label_list)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test"""

