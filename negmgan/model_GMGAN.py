#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Non-Exhaustive Gaussian Mixture Generative Adversarial Networks (NE-GM-GAN)
@topic: GM-GAN Models
@author: Jun Zhuang, Mohammad Al Hasan
@ref: github: https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn import mixture
import numpy as np
from utils import reshape_to4D, split_data


def GMM(X, num_class=1):
    # Build GMM model
    gmm = mixture.GaussianMixture(n_components=num_class, covariance_type='full')
    gmm.fit(X)
    return gmm.predict(X), gmm.means_, gmm.covariances_


# GM-GAN —————————————————————————————————————————————————————————————————————
class GMGAN():
    def __init__(self, img_rows, img_cols, channels):
        self.img_shape = (img_rows, img_cols, channels) # (11, 11, 1)
        self.latent_dim = 32

        optimizer = Adam(1e-5, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                    optimizer=optimizer,
                                    metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                    optimizer=optimizer)

    def build_encoder(self):
        # Encoder
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))
        model.summary()
        img = Input(shape=self.img_shape)
        z = model(img)
        return Model(img, z)

    def build_generator(self):
        # Generator
        model = Sequential()
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()
        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)
        return Model(z, gen_img)

    def build_discriminator(self):
        # Discriminator
        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])
        model = Dense(128)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.5)(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)
        return Model([z, img], validity)


    def reparameterization(self, mu, cov, batch_size):
        """
        @topic: Reparameterization trick: Instead of sampling z ~ N(µ, ∑), sample eps ~ N(0,I).
        @ref:
            https://keras.io/examples/variational_autoencoder/
            https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
            https://www.tensorflow.org/tutorials/generative/cvae
        # @input:
            mean (batch_size, latent_dim): mean matrix
            var (batch_size, latent_dim, latent_dim): variance matrix
        # @returns:
            z_batch (batch_size, latent_dim): sampled latent vector
        """
        if (len(mu) != batch_size) or (len(mu[0]) != self.latent_dim):
            return print("The size of mu is not correct!")
        z_batch = []
        for i in range(batch_size):
            # by default, random_normal has mean=0 and std=1.0
            epsilon = np.random.normal(size=(1, self.latent_dim))
            # z = mu[i] + np.dot(epsilon, np.sqrt(abs(cov[i])))
            # exp(log(.)*0.5) avoid negative inside np.sqrt(.)
            logvar_i = np.log(np.diag(cov[i]))
            z = mu[i] + epsilon * np.exp(logvar_i * .5)
            z_batch.append(z[0])
        return np.array(z_batch)


    def train(self, X_train, K, prior='Guassian', NUM_EPOCHS=100, BATCH_SIZE=50):
        """
        @topic: Train the model
        @input: X_train: training set; K: the number of KCs; prior: prior distribution.
        """
        # Adversarial ground truths
        valid = np.ones((BATCH_SIZE, 1))
        fake = np.zeros((BATCH_SIZE, 1))

        # Reshape the training set into 4D
        X_train = reshape_to4D(X_train)

        if prior == 'Guassian':
            # Compute the initial mu & cov in Guassian distribution
            X_encoded = self.encoder.predict(X_train)
            _, mu, cov = GMM(X_encoded, K)
            # mu: (K, latent_dim), cov: (K, latent_dim, latent_dim).
            #print("mu.shape, cov.shape: ", mu.shape, cov.shape)

        for epoch in range(NUM_EPOCHS):
            if prior == 'Guassian':
                # Get parameters, mu & cov in batch
                K_idx = np.random.randint(low=0, high=len(mu), size=BATCH_SIZE)
                mu_Kbatch = mu[K_idx] # (batch_size, latent_dim)
                sigma_coeff = 1
                cov_Kbatch = sigma_coeff*cov[K_idx] # (batch_size, latent_dim, latent_dim)
                # Reparameterization trick to sample z
                z = self.reparameterization(mu_Kbatch, cov_Kbatch, BATCH_SIZE)
            else:
                # Sample random noise z
                z = np.random.normal(size=(BATCH_SIZE, self.latent_dim))

            # generate fake data point x
            x_fake = self.generator.predict(z)

            # Select a random batch of data point x and encode
            idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE) # len(idx)=batch_size
            x_real = X_train[idx] # size = batch_size x (img_rows, img_cols, 1)
            z_ = self.encoder.predict(x_real) # z_ (batch_size, latent_dim) 

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, x_real], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, x_fake], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (z -> x_real is valid, & x_real -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, x_real], [valid, fake])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]"\
                   % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))


    def compute_rec_loss(self, X):
        """
        @topic: Compute the loss between real instance and reconstructed instance.
        @input: X (4D): real instance.
        @return: array of loss between x and G(E(x)).
        """
        z_ = self.encoder.predict(X) # generated latent variable
        x_rec = self.generator.predict(z_) # generated x from z_
        loss_list = []
        for i in range(0, len(X)):
            delta = X[i] - x_rec[i] # delta = data point - reconstructed point
            delta_flat = delta.flatten() # or = np.ndarray.flatten(delta)
            rec_loss = np.linalg.norm(delta_flat) # compute L2 norm
            #rec_loss = np.linalg.norm(delta_flat, 1) # compute L1 norm
            loss_list.append(rec_loss)
        return np.array(loss_list)


    def compute_UC_Score(self, X_train, Y_train, X_test):
        """
        @topic: Compute UC_Score.
        @input: X_train, X_test (2D); Y_train (1D).
        @return: array of UC_Score.
        """
        # Compute the median of the loss for K clusters in train set
        K_list = list(set(Y_train))
        L_train_k_list = [] # len = K
        for k in range(len(K_list)): 
            X_train_k, _ = split_data(X_train, Y_train, Y_train[k])
            X_train_k_4d = reshape_to4D(X_train_k)
            L_train_k = self.compute_rec_loss(X_train_k_4d)
            L_train_k_median = np.median(L_train_k)
            L_train_k_list.append(L_train_k_median)
        # Compute UC_Score
        X_test_4d = reshape_to4D(X_test)
        L_test = self.compute_rec_loss(X_test_4d)
        UCS = []
        for i in range(len(X_test)):
            UCS_k_list = []
            for k in range(len(L_train_k_list)):
                UCS_k = abs(L_test[i] - L_train_k_list[k])
                UCS_k_list.append(UCS_k)
            UCS_i = min(UCS_k_list)
            UCS.append(UCS_i)
        return np.array(UCS)


    def predict_outlier(self, UCS, ts):
        """
        @topic: Predict the outlier based on UCS.
        @input: UCS (vector), given threshold (float).
        @return: Y_pred: the predicted labels based on the UC Score.
        """
        ts_idx = int(len(UCS)*(1 - ts)) # the cut-off index
        UCS_copy = UCS.copy()
        UCS_copy.sort() # sorting as increasing order
        ts_UCS = UCS_copy[ts_idx] # the cut-off score
        Y_pred = []
        for i in range(len(UCS)):
            if UCS[i] >= ts_UCS:
                Y_pred.append(-1)
            else:
                Y_pred.append(0)
        return Y_pred

