#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Implement NE-GM-GAN on one streaming batch data.
@author: Jun Zhuang, Mohammad Al Hasan
@run: python main.py Syn 1500 3 200
"""

import sys
import numpy as np
from preproc import preproc_Synthetic, preproc_Network
from utils import evaluation, split_class, select_outlier, resmapling, \
                    relabel, split_pred_data, purity_score
from model_GMGAN import GMM, GMGAN
from model_imeans import Imeans
from collections import Counter


# Initialize the arugments
try:
    DATA_NAME = str(sys.argv[1])
    NUM_EPOCHS = int(sys.argv[2])
    Z = int(sys.argv[3])
    WS = int(sys.argv[4])
except:
    DATA_NAME = "Syn" # dataset name = "KDD99", "NSLKDD", "UNSWNB15", "Syn".
    NUM_EPOCHS = 1500
    Z = 3 # The coefficient of confidence interval
    WS = 200 # the number of warm-up step for building the beta prior
BATCH_SIZE = 50
PRIOR = 'Guassian' # 'Guassian' or random"

# Load dataset
network_dataname_list = ["KDD99", "NSLKDD", "UNSWNB15"]
if DATA_NAME in network_dataname_list:
    print("Seleting {0} dataset...".format(DATA_NAME))
    X_train, Y_train, X_test, Y_test = preproc_Network(DATA_NAME)
    print("The shape of training/testing set: ", X_train.shape, X_test.shape)
    # Split the classes inside the dataset
    X_test_sp, Y_test_sp = split_class(X_test, Y_test)
    # Define the ID of selected tesing set
    inlier_No_list = [0]
    #outliers_No_list = [1, 2, 3, 4, 5, 6, 7, 8]
    target_test = [0, 1, 2]
    idx = network_dataname_list.index(DATA_NAME)
    cut_num_list = [50000, 15000, 10000]
    cut_ratio_list = [0.02, 0.08, 0.1]
    # Select testing set with resampling
    X_test, Y_test = resmapling(X_test_sp, Y_test_sp, target_test, \
                                cut_num_list[idx], cut_ratio_list[idx])
elif DATA_NAME == "Syn":
    print("Seleting synthetic dataset...")
    X_train, Y_train, X_test, Y_test = preproc_Synthetic()
    print("The shape of training/testing set: ", X_train.shape, X_test.shape)
    # Define the ID of selected tesing set
    inlier_No_list = [0, 1, 2]
    #outliers_No_list = [3, 4, 5, 6, 7, 8, 9, 10]
    #target_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_test = [0, 1, 2, 3, 4]
    # Select testing set without resampling
    X_test, Y_test = select_outlier(X_test, Y_test, target_test)
else:
    print("Fail to import dataset!")

# Get the k0, mu, cov matrix from training set
k0 = len(set(Y_train)) # the number of clusters from training set
_, mu_train, cov_train = GMM(X_train, k0)
print("The shape of mu_train/cov_train: ", mu_train.shape, cov_train.shape)

# Training the model
N, Nsize = len(X_train[1]), int(np.sqrt(len(X_train[1])))
assert Nsize * Nsize == N and len(X_train.shape) == 2
gmgan = GMGAN(Nsize, Nsize, 1)
gmgan.train(X_train, k0, prior=PRIOR, NUM_EPOCHS=NUM_EPOCHS, BATCH_SIZE=BATCH_SIZE)

# Unknown Classes Detection
UCS = gmgan.compute_UC_Score(X_train, Y_train, X_test)
Y_test_re = relabel(Y_test, inlier_No_list)

# Evaluation
ts = Counter(Y_test_re)[-1]/len(Y_test_re) # Find out the best threshold
Y_pred = gmgan.predict_outlier(UCS, ts)
c_name = ['Outliers: ', 'Inliers: '] # The name of classes
evaluation(Y_test_re, Y_pred, c_name)

# Split the predicted results to inliers and outliers
X_test_in, Y_test_in = split_pred_data(X_test, Y_test, Y_pred, 0)
X_test_out, Y_test_out = split_pred_data(X_test, Y_test, Y_pred, -1)
print("Y_test_in: ", Counter(Y_test_in))
print("Y_test_out: ", Counter(Y_test_out))

# Detect the number of UCs by I-means
Imeans = Imeans()
N_list = list(Counter(Y_train).values()) # the number of instances for each cluster in training set.
k_new = Imeans.imeans(X_test_out, mu_train, cov_train, N_list, Z=Z, WS=WS, verbose=False)
print("The predicted number of k_new: ", k_new)

# Merge and rebuild the new training set.
X_train_new = np.vstack((X_train, X_test_in))
Y_train_new = np.hstack((Y_train, Y_test_in))

# Evaluate the re-clustering 
Y_pred_out, _, _ = GMM(X_test_out, k_new)
purity_out = purity_score(Y_test_out, Y_pred_out)
print("Purity score for predicted outliers: ", purity_out)

# Update the parameters
_, mu_train_new, cov_train_new = GMM(X_train_new, k0)
_, mu_test_out, cov_test_out = GMM(X_test_out, k_new)
print("The shape of cov_train_new/cov_test_out: ", cov_train_new.shape, cov_test_out.shape)
