#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Preprocessing the given datasets
@author: Jun Zhuang, Mohammad Al Hasan
"""

#import numpy as np
from utils import read_pickle, extract_data, inject_noise, rebuild_dataset, \
                normalize, select_outlier, split_label


def preproc_Network(data_name="KDD99"):
    """Preprocessing the Network Intrusion Dataset"""
    # Read the pickle file
    network_dataname_list = ["KDD99", "NSLKDD", "UNSWNB15"]
    labelname_dict = {'KDD99': "label", 'NSLKDD': "labels", 'UNSWNB15': 'attack_cat'}
    if data_name not in network_dataname_list:
        return "The name of dataset gets wrong. Fail to import dataset."
    df_train = read_pickle('../data/{0}_X_train.pkl'.format(data_name))
    df_test = read_pickle('../data/{0}_X_test.pkl'.format(data_name))
    # Split dataset and label
    X_train, Y_train = split_label(df_train, target=labelname_dict[data_name])
    X_test, Y_test = split_label(df_test, target=labelname_dict[data_name])
    # Normalize the training set / testing set
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    return X_train, Y_train, X_test, Y_test


def preproc_Synthetic():
    """Preprocessing the Synthetic Dataset"""
    # Read the pickle file
    print("Read Synthetic dataset.")
    dataset = read_pickle('../data/Synthetic_X.pkl')
    label = read_pickle('../data/Synthetic_Y.pkl')

    # Extract the dataset into three categories
    print("Extracting the dataset...")
    in_list = [0, 1, 2]
    ns_list = [13, 14, 15]
    out_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    X_in, Y_in = extract_data(dataset, label, in_list)
    X_ns, Y_ns = extract_data(dataset, label, ns_list)
    X_out, Y_out = extract_data(dataset, label, out_list)

    # Inject noises into inliers step by step
    print("Injecting the noises into inliers...")
    X_inj0, Y_inj0 = inject_noise(X_in, Y_in, X_ns, Y_ns, noise=13, target=0)
    X_inj1, Y_inj1 = inject_noise(X_inj0, Y_inj0, X_ns, Y_ns, noise=14, target=1)
    X_inj2, Y_inj2 = inject_noise(X_inj1, Y_inj1, X_ns, Y_ns, noise=15, target=2)
    X_in_new, Y_in_new = X_inj2, Y_inj2

    # Rebuild dataset and split training/testing set
    SPLIT_RATIO = 0.2
    X_train, Y_train, X_test_all, Y_test_all = \
        rebuild_dataset(X_in_new, Y_in_new, X_out, Y_out, split_rate=SPLIT_RATIO)
    print("The shape of X_train: ", X_train.shape)
    print("The shape of X_test_all: ", X_test_all.shape)
    print("The ID of classes: ", set(Y_test_all))

    # Normalize the training set and all testing set
    X_train = normalize(X_train)
    X_test_all = normalize(X_test_all)
    print("Dataset Normalized.")

    # Select outliers and reconstruct the new testing/validation set
    #target_valid = [11, 12]
    target_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #X_valid, Y_valid = select_outlier(X_test_all, Y_test_all, target_valid)
    X_test, Y_test = select_outlier(X_test_all, Y_test_all, target_test)
    #print("The shape of new validation set: {0}".format(X_valid.shape))
    print("The shape of new testing set: {0}".format(X_test.shape))
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    #X_train, Y_train, X_test, Y_test = preproc_Synthetic()
    X_train, Y_train, X_test, Y_test = preproc_Network("UNSWNB15")
    print(X_train.shape, X_test.shape)
