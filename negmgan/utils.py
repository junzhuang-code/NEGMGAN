#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Utils modules
@author: Jun Zhuang, Mohammad Al Hasan
"""

#import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import cluster


# For preprocessing -----
def read_pickle(file_name):
    """Load the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def extract_data(X, Y, y_list):
    """
    @topic: Extract specific dataset
    @input: X, Y (array): dataset; y_list: the type of class that we need to extract.
    @return: X_new, Y_new (array): new dataset.
    """
    X_new, Y_new = [], []
    for i, label in enumerate(Y):
        if label in y_list:
            X_new.append(X[i])
            Y_new.append(Y[i])
    X_new, Y_new = np.array(X_new), np.array(Y_new).astype(int)
    return X_new, Y_new

def inject_noise(X_in, Y_in, X_ns, Y_ns, noise, target):
    """
    @topic: Inject noises into targeted inliers
    @input:
        X_in, Y_in, X_ns, Y_ns (array): dataset for inliers/noise;
        noise/target: the class of noise/target.
    @return: New inliers with injected noises
    """
    X_inj, Y_inj = [], []
    for i, label in enumerate(Y_ns): # Extract noise and replace label
        if label == noise:
            X_inj.append(X_ns[i])
            Y_inj.append(int(target))
    X_new = np.vstack((X_in, np.array(X_inj))) # Merge the data
    Y_new = np.hstack((Y_in, np.array(Y_inj)))
    X_new_idx = [i for i in range(len(X_new))] # Setup the index
    np.random.seed(0)
    np.random.shuffle(X_new_idx) # shuffle the data
    return X_new[X_new_idx], Y_new[X_new_idx]

def split_data(X, Y, target):
    """Split the data with given target(label#)"""
    X_sp, Y_sp = [], []
    for i, label in enumerate(Y):
        if label == target:
            X_sp.append(X[i])
            Y_sp.append(Y[i])
    return np.array(X_sp), np.array(Y_sp)

def rebuild_dataset(X_in, Y_in, X_out, Y_out, split_rate=0.2):
    """
    @topic: Rebuild the dataset
    @input:
        X_in, Y_in, X_out, Y_out (array): dataset for inliers/outliers;
        split_rate: the rate for split the training/testing set.
    @return: New training/testing set
    """
    label_list = list(set(Y_in)) # The list of class#
    num_class = len(set(Y_in)) # The number of classes
    X_train, Y_train, X_test, Y_test = [], [], [], []
    # Split dataset with different class
    for i in range(num_class):
        X_sp, Y_sp = split_data(X_in, Y_in, label_list[i])
        cut_num = int(len(X_sp)*split_rate) # The number of samples for testing set
        X_train.extend(X_sp[cut_num:])
        Y_train.extend(Y_sp[cut_num:])
        X_test.extend(X_sp[:cut_num])
        Y_test.extend(Y_sp[:cut_num])
    X_test = np.vstack((np.array(X_test), X_out))
    Y_test = np.hstack((np.array(Y_test), Y_out))
    return np.array(X_train), np.array(Y_train).astype(int),\
            np.array(X_test), np.array(Y_test).astype(int)

def normalize(X):
    """Normalize the dataset"""
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)

def select_outlier(X, Y, target):
    """
    @topic: Select outliers for testing set
    @input: X, Y (array): Testing set; target (list): the target of outliers.
    @return: New testing set.
    """
    X_se, Y_se = [], [] # selected outliers
    for i, label in enumerate(Y):
        if label in target:
            X_se.append(X[i])
            Y_se.append(Y[i])
    return np.array(X_se), np.array(Y_se)

def resmaple_outlier(X_test, Y_test, sam_idx, cut_num, rate):
    """
    @topic: resampling the outliers with given ratio
    @input:
        X_test, Y_test: split dataset (list of list);
        sam_idx: the class of resampling (list of int);
        cut_num: the number of resampling in inlier class (int);
        rate: the percentage of anomaly class (float).
    @return:
        X_test_new, Y_test_new: resampled dataset (array).
    """
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test_cut = X_test[sam_idx]
    Y_test_cut = Y_test[sam_idx]
    for i in range(len(X_test_cut)):
        X_test_cut[i] = X_test_cut[i][0: int(cut_num*rate)]
        Y_test_cut[i] = Y_test_cut[i][0: int(cut_num*rate)]
    X_test_new = np.vstack((X_test_cut))
    Y_test_new = np.hstack((Y_test_cut))
    return X_test_new, Y_test_new

def split_label(df, target):
    """Split dataset and label and convert to array"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dataset = np.array(df[result]).astype(np.float32)
    labels = np.array(df[target]).flatten().astype(int)
    return dataset, labels

def split_class(X_test, Y_test):
    """
    @topic: Split the classes inside the dataset
    @input: X_test, Y_test (array);
    @return: X_test_sp, Y_test_sp (list of list).
    """
    X_test_sp = [[] for i in range(len(set(Y_test)))]
    Y_test_sp = [[] for i in range(len(set(Y_test)))]
    for i in range(len(X_test)):
        X_test_sp[int(Y_test[i])].append(X_test[i])
        Y_test_sp[int(Y_test[i])].append(int(Y_test[i]))
    return X_test_sp, Y_test_sp

def rseample_testset_ni(X_test_sp, Y_test_sp, comb_list, cut_num, cut_rate):
    """
    @topic: Resmapling testing set on network intrusion dataset
    @input: dataset&label--X, Y, cutting number, cutting rate.
    @return: list of dataset&label(X, Y) in combinations.
    """
    X_test_comblist, Y_test_comblist = [], []
    for i in range(len(comb_list)):
        sam_idx_i = list(comb_list[i])
        X_rs_i, Y_rs_i = resmaple_outlier(X_test_sp, Y_test_sp, sam_idx_i, cut_num, cut_rate)
        X_test_comblist.append(X_rs_i)
        Y_test_comblist.append(Y_rs_i)
    return X_test_comblist, Y_test_comblist

def resample_testset_syn(X_test, Y_test, comb_list):
    """
    @topic: Resmapling testing set on synthetic dataset
    @input: dataset&label--X, Y, comb_lis--combination_list.
    @return: list of dataset&label(X, Y) in combinations.    
    """
    X_test_comblist, Y_test_comblist = [], []
    for i in range(len(comb_list)):
        sam_idx_i = list(comb_list[i])
        X_rs_i, Y_rs_i = select_outlier(X_test, Y_test, sam_idx_i)
        X_test_comblist.append(X_rs_i)
        Y_test_comblist.append(Y_rs_i)
    return X_test_comblist, Y_test_comblist

def resmapling(X_test, Y_test, sam_idx, cut_num, rate):
    """
    @topic: Resampling the network intrusion dataset
    @input:
        X_test, Y_test: split dataset (list of list);
        sam_idx: the class of resampling (list of int);
        cut_num: the number of resampling in inlier class (int);
        rate: the percentage of anomaly class (float).
    @return:
        X_test_new, Y_test_new: resampled dataset (array).
    """
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test_cut = X_test[sam_idx]
    Y_test_cut = Y_test[sam_idx]
    print(X_test_cut.shape)
    for i in range(len(X_test_cut)):
        if i == 0: # for inlier class
            X_test_cut[i] = X_test_cut[i][0: int(cut_num)]
            Y_test_cut[i] = Y_test_cut[i][0: int(cut_num)]
        else: # for outlier class
            X_test_cut[i] = X_test_cut[i][0: int(cut_num*rate)]
            Y_test_cut[i] = Y_test_cut[i][0: int(cut_num*rate)]
    X_test_new = np.vstack((X_test_cut))
    Y_test_new = np.hstack((Y_test_cut))
    #print("The shape of X_test_new: ", X_test_new.shape)
    #print("The shape of Y_test_new: ", Y_test_new.shape)
    return X_test_new, Y_test_new


# For Model GM-GAN -----
def reshape_to4D(X):
    """Reshape the dataset to 4D"""
    def can_sqrt(n):
        # Tell if n has int sqaure root
        a = int(np.sqrt(n))
        return a * a == n
    if can_sqrt(len(X[0])):
    #if int(np.sqrt(len(X_data[0]))) == np.sqrt(len(X_data[0])):
        return np.array(X).reshape((len(X), int(np.sqrt(len(X[0]))), -1, 1))
    else:
        return print("Fail to reshape!")

def reshape_to2D(X):
    """Reshape the dataset to 2D"""
    return np.array(X).reshape((len(X), -1))

def relabel(label, inlier_list):
    """
    @topic: Rebuild the label to -1 or 0.
    @input: label (vector), inlier_list: inlier id list (list).
    @return: rebuilt label (vector).
    """
    Y_out = np.zeros_like(label)
    for l in range(len(label)):
        if label[l] in inlier_list:
            Y_out[l] = 0 # inlier
        else:
            Y_out[l] = -1 # outlier
    return Y_out

def split_pred_data(X, Y, Y_pred, v):
    """
    @topic: Split target class from dataset based on predicted result
    @input: X, Y: dataset & label, Y_pred: predicted label, v: target value (int).
    @return: X_new, Y_new: new dataset & label.
    """
    X_new, Y_new = [], []
    for i in range(len(Y_pred)):
        if Y_pred[i] == int(v):
            X_new.append(X[i])
            Y_new.append(Y[i])
    return np.array(X_new), np.array(Y_new)

def evaluation(y_true, y_pred, c_name):
    """Output the classification report for GM-GAN"""
    print(classification_report(y_true, y_pred, target_names=c_name))


# For Model I-means -----
def compute_loss(X1, X2, p=2):
    """Compute L_{p} loss among two instances."""
    delta = X1 - X2
    delta_flat = delta.flatten()
    return np.linalg.norm(delta_flat, p)


# For others -----
def find_target_index(sortedList, target):
    """Find the index of target in sorted list by binary search."""
    low = 0
    high = len(sortedList) - 1
    while low <= high:
        mid = (high + low) // 2
        if sortedList[mid] == target:
            return mid
        elif target < sortedList[mid]:
            high = mid -1
        else:
            low = mid + 1
    return low

def purity_score(y_true, y_pred):
    """Compute contingency matrix (also called confusion matrix)"""
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
