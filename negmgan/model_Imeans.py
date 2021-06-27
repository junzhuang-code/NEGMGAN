#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: I-means Model
@author: Jun Zhuang, Mohammad Al Hasan
@ref:
    https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
"""

import numpy as np
from utils import compute_loss


class Imeans():

    def denoising(self, alist, ts=0.5):
        """
        @topic: Denoising Activation Function
        @input: 1D list, threshold(float); @output: 1D list.
        """
        if ts > 1 or ts < 0:
            return "Given threshold should be in the range of [0, 1]."
        list_dn = []
        #list_max, list_min = max(alist), min(alist)
        list_max, list_min = max(alist), 0
        for i in range(len(alist)):
            # normalize the data
            i_nm = (alist[i] - list_min) / (list_max - list_min)
            # filter the data with given threshold
            if i_nm > ts:
                list_dn.append(1)
            else:
                list_dn.append(0)
        return list_dn


    def testing(self, x, i, n_round):
        """Output the information in n_round"""
        if n_round <= 0:
            return "n_round must be larger than zero."
        if i % n_round == 0:
            print(x)


    def imeans(self, X, mu, cov, N, Z=3, WS=100, verbose=True):
        """
        @topic: I-means algorithm: detect the number of new cluster.
        @input: X: a batch of testing points (array);
                mu: mean of original clusters (list of list); e.g. mu = [[mu0], [mu1], [mu2], ...]
                cov: covariance of original clusters (list of list); e.g. cov = [[cov0], [cov1], [cov2], ...]
                N: the number of samples in original clusters (list of int); N = [n0, n1, n2, ...]
                Z: the value for "Z sigma rule" based on the test of confidence interval (int); 
                WS: the number of epochs in the warm-up stage for learning beta prior knowledge (int).
        @output: k_new: the number of new cluster (int).
        """
        # Initializ parameters
        mu = list(mu)
        sigma = [np.sqrt(np.diag(cov[i])) for i in range(len(cov))] # convert to (3, 121, 121) diag matrix.
        ts = [compute_loss(mu[i]+Z*sigma[i], mu[i]) for i in range(len(mu)) if len(mu)==len(sigma)] # threshold list (3,)
        N_test = list(np.zeros_like(N)) # empty list for storing the number of testing clusters
        N_loss = [[] for i in range(len(N))] # collect the historical loss_{min} of existing clusters
        N_sp = [[1, 1] for i in range(len(N))] # store the shape parameters [alpha, beta]

        for i in range(len(X)): # for each testing point in a batch
            if verbose:
                self.testing("Round {0}: ".format(i), i, 100)
            # Compute the loss to each cluster and find out the loss_{min}.
            loss_k_list = []
            for k in range(len(mu)):
                loss_k = compute_loss(X[i], mu[k])
                loss_k_list.append(loss_k)
            if verbose:
                self.testing("The loss to {0} clusters: \n {1}".format(len(loss_k_list), loss_k_list), i, 100)
            loss_min = min(loss_k_list) # select the min value from loss_k_list.
            nidx = loss_k_list.index(loss_min) # return the index of loss_min.

            # Select the threshold TS
            if len(N_loss[nidx]) <= WS:
                TS = ts[nidx] # select TS based on "Z sigma rule" (Z=3).
                ts[nidx] = compute_loss(mu[nidx]+Z*sigma[nidx], mu[nidx]) # Update TS
            else:
                # Compute the theta_MAP for "nidx" cluster: theta_MAP = alpha / (alpha + beta)
                theta_MAP = N_sp[nidx][0] / (N_sp[nidx][0] + N_sp[nidx][1])
                ts_idx = int(len(N_loss[nidx])*(1 - theta_MAP)) # compute the threshold TS index based on theta_MAP.
                TS = N_loss[nidx][ts_idx] # select the "ts_idx"-th norm in "N_loss" as threshold.

            # Make a decision
            if loss_min <= TS: # if loss_min < TS: Xi belongs to cluster[nidx].
                # Update mu and sigma in streaming data
                mu_old = mu[nidx] 
                # mu_{n+1} = mu_{n} + (x_{n+1} - mu_{n})/(n+1)
                mu[nidx] = mu_old + (X[i] - mu[nidx])/(N[nidx]+1)
                # v_{n+1} = v_{n} + (x_{n+1} - mu_{n})*(x_{n+1} - mu_{n+1}); sigma_{n+1} = √[v_{n+1}/n]
                sigma[nidx] = np.sqrt(((sigma[nidx]**2)*N[nidx] + (X[i] - mu_old)*(X[i] - mu[nidx]))/N[nidx])
                N[nidx] = N[nidx] + 1
                N_test[nidx] = N_test[nidx] + 1
                N_loss[nidx].append(loss_min) # store the loss_min to corresponding clusters.
                N_loss[nidx].sort() # sort the list of loss_min.
                N_sp[nidx][1] = N_sp[nidx][1] + 1 # beta+1
                if verbose:
                    self.testing("The number of samples in cluster {0}: {1}.".format(nidx, N[nidx]), i, 50)
            else: # if loss_min > TS: Xi belongs to new cluster.
                mu.append(X[i]) # assign current Xi as new mean vector
                sigma.append(np.zeros_like(X[i])) # the sigma is 0 for only one point
                ts.append(np.mean(ts)) # use the mean of ts list as the initial threshold of new point
                N.append(1)
                N_test.append(1)
                N_loss.append([loss_min]) # store loss_min to new entry
                N_sp.append([1,1]) # initialize a beta distribution for new cluster
                N_sp[nidx][0] = N_sp[nidx][0] + 1 # alpha+1

        # Filter the noise inside predicted result
        if verbose:
            print("Predicted clusters and corresponding numbers: \n", N_test)
        N_test_dn = self.denoising(N_test, 0.3)
        k_new = sum(N_test_dn)
        return k_new
