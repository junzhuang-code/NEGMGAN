#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Ablation study: Implement I-means algorithm to detect the number of new emerging classes on streaming data.
@author: Jun Zhuang, Mohammad Al Hasan
@ref:
    https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
@run: python ablation_DetectUCs.py Syn 2 10 3 200
"""

import sys
import time
import random
import itertools
import numpy as np
from collections import Counter
from preproc import preproc_Synthetic, preproc_Network
from model_GMGAN import GMM
from model_imeans import Imeans
from utils import split_class, compute_loss, find_target_index, \
                    rseample_testset_ni, resample_testset_syn


def compute_confidence_interval(X_train, mu, cov, Z):
    """Test the confidence interval for 'Z sigma rule' """
    mu = list(mu)
    sigma = [np.sqrt(np.diag(cov[i])) for i in range(len(cov))] # convert to (3, 121, 121) diag matrix.
    ts = [compute_loss(mu[i]+Z*sigma[i], mu[i]) for i in range(len(mu)) if len(mu)==len(sigma)] # threshold list (3,)
    norm_list = [[] for _ in range(len(mu))] # collect norm_min for each cluster
    for x in range(len(X_train)):
        # Compute L2 norm to each cluster
        norm_c_list = []
        for c in range(len(mu)):
            norm_c = compute_loss(X_train[x], mu[c])
            norm_c_list.append(norm_c)
        norm_min = min(norm_c_list) # select the min value from norm_c_list.
        nidx = norm_c_list.index(norm_min) # return the index of norm_min.
        norm_list[nidx].append(norm_min) # store the norm_min to corresponding cluster
    # sort the norm list
    for i in range(len(norm_list)):
        norm_list[i].sort()
    # find the index of ts in sorted norm list
    idx_list = [find_target_index(norm_list[i], ts[i]) for i in range(len(norm_list))]
    # compute the percentage of each ts.
    percent_list = [idx_list[i]/len(norm_list[i]) for i in range(len(idx_list))]
    return np.mean(percent_list)


if __name__ == "__main__":
    # Initialize the arugments
    try:
        DATA_NAME = str(sys.argv[1])
        NUM_CLASS_TEST = int(sys.argv[2])
        NUM_COMB = int(sys.argv[3])
        Z = int(sys.argv[4])        
        WS = int(sys.argv[5])
    except:
        DATA_NAME = "Syn" # dataset name = "KDD99", "NSLKDD", "UNSWNB15", "Syn".
        NUM_CLASS_TEST = 2 # the number of new emerging class on testing set
        NUM_COMB = 10 # the number of combinations selected
        Z = 3 # The coefficient of confidence interval
        WS = 200 # the number of warm-up step for building the beta prior

    # Load dataset
    network_dataname_list = ["KDD99", "NSLKDD", "UNSWNB15"]
    if DATA_NAME in network_dataname_list:
        print("Seleting {0} dataset...".format(DATA_NAME))
        X_train, Y_train, X_test, Y_test = preproc_Network(DATA_NAME)
        print("The shape of training/testing set: ", X_train.shape, X_test.shape)
        # Split the classes inside the dataset
        X_test_sp, Y_test_sp = split_class(X_test, Y_test)
        # Define the ID of selected tesing set
        outliers_No_list = [1, 2, 3, 4, 5, 6, 7, 8]
    elif DATA_NAME == "Syn":
        print("Seleting synthetic dataset...")
        X_train, Y_train, X_test, Y_test = preproc_Synthetic()
        print("The shape of training/testing set: ", X_train.shape, X_test.shape)
        # Define the ID of selected tesing set
        outliers_No_list = [3, 4, 5, 6, 7, 8, 9, 10]
    else:
        print("Fail to import dataset!")

    # Get the mu, cov matrix from training set
    k0 = len(set(Y_train)) # the number of clusters from training set
    _, mu_train, cov_train = GMM(X_train, k0)
    print("The shape of mu_train/cov_train: ", mu_train.shape, cov_train.shape)

    # Test the Z by computing confidence interval
    is_TestCI = False
    if is_TestCI:
        ci = compute_confidence_interval(X_train, mu_train, cov_train, Z)
        print("The data points have {0}% to fall in the range of μ ± {1}σ.".format(ci*100, Z))

    # Generate the combinations of testing set
    comb_list = list(itertools.combinations(outliers_No_list, NUM_CLASS_TEST))
    # Select n combinations
    comb_list_n = random.sample(comb_list, NUM_COMB)
    print("The combinations selected: \n", comb_list_n)
    # Resmapling testing sets in combinations.
    if DATA_NAME in network_dataname_list:
        X_test_comblist, Y_test_comblist = \
        rseample_testset_ni(X_test_sp, Y_test_sp, comb_list_n, cut_num=10000, cut_rate=0.1)
    elif DATA_NAME == "Syn":
        X_test_comblist, Y_test_comblist = resample_testset_syn(X_test, Y_test, comb_list_n)

    # Implement I-means to detect the number of new emerging clusters on streaming data
    Imeans = Imeans()
    N_list = list(Counter(Y_train).values()) # the number of instances for each cluster in training set.
    k_new_list = []
    time_list = []
    for x_test_i in X_test_comblist:
        time_start = time.clock()
        k_new = Imeans.imeans(x_test_i, mu_train, cov_train, N_list, \
                              Z=Z, WS=WS, verbose=False)
        time_end = time.clock()
        elapse = time_end-time_start
        k_new_list.append(k_new)
        time_list.append(elapse)
    print("The list of predicted result: ", k_new_list, np.mean(k_new_list))
    print("Average time: ", np.mean(time_list))
