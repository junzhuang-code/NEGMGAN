#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Ablation study: Implement GM-GAN to extract UCs on streaming data.
@author: Jun Zhuang, Mohammad Al Hasan
@run: python ablation_ExtractUCs.py KDD99 1000
"""

import sys
import numpy as np
from preproc import preproc_Synthetic, preproc_Network
from utils import evaluation, select_outlier, relabel
from model_GMGAN import GMGAN
from collections import Counter


# Initialize the arugments
try:
    DATA_NAME = str(sys.argv[1])
    NUM_EPOCHS = int(sys.argv[2])
except:
    DATA_NAME = "UNSWNB15" # dataset name = "KDD99", "NSLKDD", "UNSWNB15", "Syn".
    NUM_EPOCHS = 10
BATCH_SIZE = 50
PRIOR = 'Guassian' # 'Guassian' or random"

# Load dataset
network_dataname_list = ["KDD99", "NSLKDD", "UNSWNB15"]
if DATA_NAME in network_dataname_list:
    print("Seleting {0} dataset...".format(DATA_NAME))
    X_train, Y_train, X_test, Y_test = preproc_Network(DATA_NAME)
    print("The shape of training/testing set: ", X_train.shape, X_test.shape)
    # Define the ID of selected tesing set
    inlier_No_list = [0]
    #outliers_No_list = [1, 2, 3, 4, 5, 6, 7, 8]
    target_test = [0, 1, 2]
elif DATA_NAME == "Syn":
    print("Seleting synthetic dataset...")
    X_train, Y_train, X_test, Y_test = preproc_Synthetic()
    print("The shape of training/testing set: ", X_train.shape, X_test.shape)
    # Define the ID of selected tesing set
    inlier_No_list = [0, 1, 2]
    #outliers_No_list = [3, 4, 5, 6, 7, 8, 9, 10]
    #target_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_test = [0, 1, 2, 3, 4]
else:
    print("Fail to import dataset!")

# Select testing set without resampling
X_test, Y_test = select_outlier(X_test, Y_test, target_test)

# Training the model
k0 = len(set(Y_train)) # the number of clusters from training set
N, Nsize = len(X_train[1]), int(np.sqrt(len(X_train[1])))
assert Nsize * Nsize == N and len(X_train.shape) == 2
gmgan = GMGAN(Nsize, Nsize, 1)
gmgan.train(X_train, k0, prior=PRIOR, NUM_EPOCHS=NUM_EPOCHS, BATCH_SIZE=BATCH_SIZE)

# Unknown Classes Detection
UCS = gmgan.compute_UC_Score(X_train, Y_train, X_test)
Y_test_re = relabel(Y_test, inlier_No_list, -1)

# Evaluation
ts = Counter(Y_test_re)[-1]/len(Y_test_re) # Find out the best threshold
Y_pred = gmgan.predict_outlier(UCS, ts)
c_name = ['Outliers: ', 'Inliers: '] # The name of classes
evaluation(Y_test_re, Y_pred, c_name)
# Extraction done!
