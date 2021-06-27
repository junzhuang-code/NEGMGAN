#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: Generate qualified dataset from raw data
@author: Jun Zhuang, Mohammad Al Hasan
@run: python gen_Data.py KDD99 ../data/
"""

import os
import sys
import numpy as np
import pandas as pd
from utils import compute_loss, dump_pickle


# For Network Intrusion Dataset -----
def onehot_embedding(df, name):
    # Employ one-hot embedding on categorical values (i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
    dummies = pd.get_dummies(df.loc[:,name]) # one-hot embedding for selected columns
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x) # rename
        df.loc[:, dummy_name] = dummies[x] # add new columns to the dataframe
    df.drop(name, axis=1, inplace=True) # remove original columns

def update_label(df_data, inliers, outliers, label_name="labels"):
    # Replace the string with integer
    labels = df_data[label_name].copy()
    labels = np.array(labels)
    for l in range(len(list(labels))):
        if labels[l] in outliers:
            for c in range(len(outliers)):
                if labels[l] == outliers[c]:
                    labels[l] = c+1
        elif labels[l] in inliers:
            labels[l] = 0
    df_data[label_name] = labels
    return df_data

def split_network_data(df_data, label_name="label", test_ratio=0.2, seed=42):
    # Split the train/test data on network intrusion dataset
    df_inliers = df_data[df_data[label_name] == 0]
    df_outliers = df_data[df_data[label_name] != 0]
    df_test1 = df_inliers.sample(frac=test_ratio, random_state=seed)
    df_test = pd.concat([df_test1, df_outliers], axis=0)
    df_train = df_inliers[~df_inliers.index.isin(df_test1.index)]
    print("The Shape of Train/Test: {0}, {1}.".format(df_train.shape, df_test.shape))
    return df_train, df_test

def generate_KDD99(data_path, LABEL_NAME="label"):
    # Generate KDD99 dataset
    # Read dataset
    col_names = ["duration","protocol_type","service","flag","src_bytes", \
                "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins", \
                "logged_in","num_compromised","root_shell","su_attempted","num_root", \
                "num_file_creations","num_shells","num_access_files","num_outbound_cmds", \
                "is_host_login","is_guest_login","count","srv_count","serror_rate", \
                "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate", \
                "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", \
                "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate", \
                "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate", \
                "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    df_data = pd.read_csv(data_path, header=None, names=col_names)
    # Implement one-hot embedding
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    for name in text_l:
        onehot_embedding(df_data, name)
    # Rename the order of labels for outliers
    outliers = ['neptune.', 'normal.', 'back.', 'satan.', 'ipsweep.', 'portsweep.', 'warezclient.', 'teardrop.']
    labels = df_data[LABEL_NAME].copy()
    for l in range(len(labels)):
        if labels[l] in outliers:
            for c in range(len(outliers)):
                if labels[l] == outliers[c]:
                    labels[l] = c+1
        else:
            labels[l] = 0
    df_data[LABEL_NAME] = labels
    # Split the train, val, test set
    df_train, df_test = split_network_data(df_data, label_name=LABEL_NAME, test_ratio=0.2, seed=42)
    return df_train, df_test

def generate_NSLKDD(data_path, LABEL_NAME="labels"):
    # Generate NSL-KDD dataset
    # Read dataset
    col_names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","labels_int"] # len = 43
    df_train = pd.read_csv(data_path+"Train.txt", header=None, names=col_names)
    df_test = pd.read_csv(data_path+"Test.txt", header=None, names=col_names)
    df_data = pd.concat([df_train, df_test], ignore_index=True)
    df_data.drop('labels_int', axis=1, inplace=True)
    # Drop columns whose binary values are almost 0
    df_data_ohe = df_data.copy() # copy the dataframe for encoding
    columns_drop_list = ['land', 'num_outbound_cmds', 'is_host_login']
    for col in columns_drop_list:
        df_data_ohe.drop(col, axis=1, inplace=True)
    # Select features for one-hot embedding
    columns_encoding_list = ['protocol_type', 'service', 'flag', 'logged_in', 'is_guest_login']
    for col in columns_encoding_list:
        onehot_embedding(df_data_ohe, col)
    # Decide the classes of inliers & outliers
    inliers = ['normal', 'buffer_overflow', 'land', 'multihop', 'rootkit', 'named', \
                'ps', 'sendmail', 'xterm', 'imap', 'ftp_write', 'loadmodule', 'xlock', \
                'phf', 'perl', 'xsnoop', 'worm', 'udpstorm', 'spy', 'sqlattack']
    outliers = ['neptune', 'satan', 'ipsweep', 'smurf', 'portsweep', 'nmap', 'back', 'guess_passwd']
    class_ = inliers + outliers
    # Extract classes and reform the new dataframe
    df_data = df_data_ohe[df_data_ohe[LABEL_NAME].isin(class_)]
    # Replace the string with integer
    df_data = update_label(df_data, inliers, outliers, label_name=LABEL_NAME)
    # Split dataset into train/test set
    df_train, df_test = split_network_data(df_data, label_name=LABEL_NAME, test_ratio=0.2, seed=42)
    return df_train, df_test

def generate_UNSWNB15(data_path, LABEL_NAME="attack_cat"):
    # Generate UNSW-NB15 dataset
    # Read dataset
    df_data = pd.read_csv(data_path+"train.csv")
    # Select features for one-hot embedding
    df_data_ohe = df_data.copy() # copy the dataframe for encoding
    columns_encoding_list = ['proto', 'state']
    for col in columns_encoding_list:
        onehot_embedding(df_data_ohe, col)
    # Drop necessary columns
    features_drop = ['id', 'service', 'dwin', 'swin', 'is_sm_ips_ports', 'is_ftp_login', 'ct_ftp_cmd', 'label']
    onehot_drop = ['proto-rtp', 'proto-icmp', 'proto-igmp', 'state-ECO', 'state-URN', 'state-no', 'state-PAR']
    columns_drop_list = features_drop + onehot_drop
    for col in columns_drop_list:
        df_data_ohe.drop(col, axis=1, inplace=True)
    # Decide the classes of inliers & outliers
    inliers = ['Normal', 'Worms'] 
    outliers = ['Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode']
    class_ = inliers + outliers
    # Extract classes and reform the new dataframe
    df_data = df_data_ohe[df_data_ohe[LABEL_NAME].isin(class_)]
    # Replace the string with integer
    df_data = update_label(df_data, inliers, outliers, label_name=LABEL_NAME)
    # Split dataset into train/test set
    df_train, df_test = split_network_data(df_data, label_name=LABEL_NAME, test_ratio=0.2, seed=42)
    return df_train, df_test

def generate_Network(data_name="KDD99", data_dir="../data/"):
    # Generate network intrusion dataset
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    labelname_dict = {'KDD99': "label", 'NSLKDD': "labels", 'UNSWNB15': 'attack_cat'}
    if data_name not in labelname_dict.keys():
        return "The name of dataset gets wrong. Fail to generate dataset."
    elif data_name == "KDD99":
        data_path = data_dir+"kddcup.data_10_percent_corrected"
        df_train, df_test = generate_KDD99(data_path, labelname_dict[data_name])
    elif data_name == "NSLKDD":
        data_path = data_dir+"NSL_KDD_"
        df_train, df_test = generate_NSLKDD(data_path, labelname_dict[data_name])
    elif data_name == "UNSWNB15":
        data_path = data_dir+"UNSW_NB15_"
        df_train, df_test = generate_UNSWNB15(data_path, labelname_dict[data_name])
    # Dump to pickle file
    dump_pickle(data_dir+data_name+'_X_train.pkl', df_train)
    dump_pickle(data_dir+data_name+'_X_test.pkl', df_test)


# For Synthetic Dataset -----
def find_min_norm(matrix, k):
    # Find out the min norm in each sub-matrix
    norm_min = float("inf")
    for i in range(k-1):
        for j in range(i+1, k):
            norm = compute_loss(matrix[i], matrix[j], p=1)
            #print("norm: ", norm)
            if norm < norm_min:
                norm_min = norm
    return norm_min

def find_max_seed(k, dim, iters=100):
    # topic: Find out the max seed
    # input: k: #clusters; dim: the dimension of the mean vector; iters: the number of iterations.
    # output: mean_vector (k, dim).
    norm_min_list = []
    for i in range(iters): # run "iters" times
        np.random.seed(i)
        #mu_mx = np.random.random((k, dim)) # generate the mean vector
        mu_mx = np.random.randint(0, 20, size=(k, dim))
        norm_min = find_min_norm(mu_mx, k) # test its min norm among each sub-means.
        norm_min_list.append(norm_min)
    #print(norm_min_list)
    max_v = max(norm_min_list) # select the max value from norm_min_list.
    n_seed = norm_min_list.index(max_v) # return the index of norm_min_list.
    print("max_v, n_seed: ", max_v, n_seed)
    return n_seed

def gen_mean_vector(k, dim, n_seed):
    # topic: Generate a mean vector s.t. maximize the distance for each cluster.
    # input: k: #clusters; dim: the dimension of the mean vector; n_seed: seed id.
    # output: mean_vector (k, dim).
    np.random.seed(n_seed) # use this seed to generate the mean vector
    return np.random.randint(0, 20, size=(k, dim))

def gen_var_matrix(k, dim, size_list):
    # topic: Generate a variance matrix with given size.
    # input: k: #clusters; dim: the dimension of the mean vector; size_list: the size of var matrix。
    # output: var_matrix (k, dim, dim).
    var_mx = []
    for i in range(k): # 总共生成k个
        var = np.random.randint(1, size_list[i], size=dim)
        var_diag = np.diag(var)*0.1
        var_mx.append(var_diag)
    return np.array(var_mx)

def make_multi_normal(mu_mx, var_mx, n_sam, k):
    # topic: Generate the dataset with multivariate mormal distribution (non-symmetric Gaussian clusters).
    # input: mu_mx\var_mx: mean\var matrix; n_sam: the number of instances. output: dataset\label.
    X, Y = [], []
    for i in range(k):
        X_k = np.random.multivariate_normal(mu_mx[i], var_mx[i], n_sam[i])
        Y_k = [i for _ in range(len(X_k))]
        X.extend(X_k)
        Y.extend(Y_k)
    return np.array(X), np.array(Y)

def generate_Synthetic(k, dim, var_list, n_sam, data_path):
    # Generate synthetic dataset
    n_seed = find_max_seed(k, dim, iters=1000)
    mu_mx = gen_mean_vector(k, dim, n_seed)
    mu_mx = np.vstack((mu_mx[0:-3], mu_mx[0:3]))
    var_mx = gen_var_matrix(k, dim, var_list)
    dataset, label = make_multi_normal(mu_mx, var_mx, n_sam, k)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dump_pickle(data_path+'Synthetic_X.pkl', dataset)
    dump_pickle(data_path+'Synthetic_Y.pkl', label)


if __name__ == "__main__":
    # Initialize the arugments
    try:
        DATA_NAME = str(sys.argv[1])
        DATA_DIR = str(sys.argv[2])
    except:
        DATA_NAME = "KDD99" # dataset name = "KDD99", "NSLKDD", "UNSWNB15", "Syn".
        DATA_DIR = "../data1/"
    print("Generating {0} dataset... ".format(DATA_NAME))
    if DATA_NAME == "Syn": # Generate Synthetic Dataset
        k = 16
        dim = 121
        var_list = [3, 3, 4, 5, 4, 5, 6, 4, 7, 4, 5, 5, 5, 15, 15, 16]
        n_sam = [30000, 30000, 30000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100] # The number of instances    
        generate_Synthetic(k, dim, var_list, n_sam, data_path=DATA_DIR)
    else: # Generate Network Intrusion Dataset
        generate_Network(data_name=DATA_NAME, data_dir=DATA_DIR)
    print("The Generation of {0} dataset is completed!".format(DATA_NAME))
