#!/usr/bin/env python
# encoding: utf-8
"""
cae.py

A pythonic library for Contractive Auto-Encoders. This is
for people who want to give CAEs a quick try and for people
who want to understand how they are implemented. For this
purpose we tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Created by Yann N. Dauphin, Salah Rifai on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin, Salah Rifai. All rights reserved.
"""

import sys
import os
import numpy as np
import scipy.optimize

from utils import vector_from_args, args_from_vector


class CAE(object):
    """
    A Contractive Auto-Encoder (CAE) with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, 
                 n_hiddens=1024,
                 jacobi_penalty=0.1,
                 learning_rate=0.1,
                 W=None,
                 b=None,
                 c=None,
                 batch_size=20,
                 epochs=20):
        """
        Initialize a CAE.
        
        Parameters
        ----------
        n_hiddens : int, optional
            Number of binary hidden units
        jacobi_penalty : float, optional
            Scalar by which to multiply the gradients coming from the jacobian
            penalty.
        learning_rate : float, optional
            Learning rate to use during learning
        W : array-like, shape (n_inputs, n_hiddens), optional
            Weight matrix, where n_inputs in the number of input
            units and n_hiddens is the number of hidden units.
        b : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        c : array-like, shape (n_inputs,), optional
            Biases of the input units
        batch_size : int, optional
            Number of examples to use per gradient update
        epochs : int, optional
            Number of epochs to perform during learning
        """
        self.n_hiddens = n_hiddens
        self.jacobi_penalty = jacobi_penalty
        self.learning_rate = learning_rate
        self.W = W
        self.b = b
        self.c = c
        self.batch_size = batch_size
        self.epochs = epochs
        self.W = W
        self.b = b
        self.c = c


    def _sigmoid(self, x):
        """
        Implements the logistic function.
        
        Parameters
        ----------
        x: array-like, shape (M, N)

        Returns
        -------
        x_new: array-like, shape (M, N)
        """
        #return 1. / (1. + np.exp(-np.maximum(np.minimum(x, 30), -30)))
        return 1. / (1. + np.exp(-x))

    def encode(self, x):
        """
        Computes the hidden code for the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)

        Returns
        -------
        h: array-like, shape (n_examples, n_hiddens)
        """
        return self._sigmoid(np.dot(x, self.W) + self.b)

    def decode(self, h):
        """
        Compute the reconstruction from the hidden code {\bf h}.
        
        Parameters
        ----------
        h: array-like, shape (n_examples, n_hiddens)
        
        Returns
        -------
        x: array-like, shape (n_examples, n_inputs)
        """
        return self._sigmoid(np.dot(h, self.W.T) + self.c)

    def reconstruct(self, x):
        """
        Compute the reconstruction of the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        x_new: array-like, shape (n_examples, n_inputs)
        """
        return self.decode(self.encode(x))

    def jacobian(self, x):
        """
        Compute jacobian of {\bf h} with respect to {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        jacobian: array-like, shape (n_examples, n_hiddens, n_inputs)
        """
        h = self.encode(x)
        
        return (h * (1 - h))[:, :, None] * self.W.T

    def sample(self, x, sigma=1):
        """
        Sample a point {\bf y} starting from {\bf x} using the CAE
        generative process.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        sigma: float
        
        Returns
        -------
        y: array-like, shape (n_examples, n_inputs)
        """
        h = self.encode(x)
        s = h * (1. - h)
        JJ = np.dot(self.W.T, self.W) * s[:, None, :] * s[:, :, None]
        alpha = np.random.normal(0, sigma, h.shape)
        delta = (alpha[:, :, None] * JJ).sum(1)
        return self.decode(h + delta)

    def loss(self, x, h=None, r=None):
        """
        Computes the error of the model with respect
        to the total cost.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        h: array-like, shape (n_examples, n_hiddens), optional
        r: array-like, shape (n_examples, n_inputs), optional
        
        Returns
        -------
        loss: array-like, shape (n_examples,)
        """
        if h == None:
            h = self.encode(x)
        if r == None:
            r = self.decode(h)
        
        def _reconstruction_loss(h, r):
            """
            Computes the error of the model with respect
            to the reconstruction (cross-entropy) cost.
            """
            return (- (x * np.log(r)
                + (1 - x) * np.log(1 - r)).sum(1)).sum()
        def _jacobi_loss(h):

            """
            Computes the error of the model with respect
            the Frobenius norm of the jacobian.
            """
            return (((h * (1 - h))**2).sum(0) * self.W**2).sum()

        return (_reconstruction_loss(h, r)
            + self.jacobi_penalty * _jacobi_loss(h))

    def get_params(self):
        return vector_from_args([self.W, self.b, self.c])

    def set_params(self, theta):
        params = args_from_vector(theta, [self.W, self.b, self.c])
        self.W = params[0]
        self.b = params[1]
        self.c = params[2]

    def f_df(self, theta, x):
        """
        Compute objective and gradient of the CAE objective using the
        examples {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Parameters
        ----------
        loss: array-like, shape (n_examples,)
            Value of the loss function for each example before the step.
        """
        self.set_params(theta)
        h = self.encode(x)
        r = self.decode(h)
        def _contraction_jacobian():
            """
            Compute the gradient of the contraction cost w.r.t parameters.
            """
            a = (h * (1 - h))**2 

            d = ((1 - 2 * h) * a * (self.W**2).sum(0)[None, :])

            b = np.dot(x.T / x.shape[0], d)

            c = a.mean(0) * self.W

            return (b + c), d.mean(0)
        
        def _reconstruction_jacobian():
            """                                                                 
            Compute the gradient of the reconstruction cost w.r.t parameters.      
            """
            dr = (r - x) / x.shape[0]
            dd = np.dot(dr.T, h)
            dh = np.dot(dr, self.W) * h * (1. - h)
            de = np.dot(x.T, dh)

            return (dd + de), dr.sum(0), dh.sum(0)

        W_rec, c_rec, b_rec = _reconstruction_jacobian()
        W_con, b_con = _contraction_jacobian()
        dW = W_rec + self.jacobi_penalty * W_con
        db = b_rec + self.jacobi_penalty * b_con
        dc = c_rec 
        return self.loss(x, h, r), vector_from_args([dW, db, dc]);

    def init_weights(self, n_input, dtype=np.float32):
        self.W = np.asarray(np.random.uniform(
            low=-4*np.sqrt(6./(n_input+self.n_hiddens)),
            high=4*np.sqrt(6./(n_input+self.n_hiddens)),
            size=(n_input, self.n_hiddens)), dtype=dtype)
        self.b = np.zeros(self.n_hiddens, dtype=dtype)
        self.c = np.zeros(n_input, dtype=dtype)


    def fit(self, X, verbose=False, callback=None):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_examples, n_inputs)
            Training data, where n_examples in the number of examples
            and n_inputs is the number of features.
        """

        if self.W == None:
            self.init_weights(X.shape[1], X.dtype)

        inds = range(X.shape[0])
        np.random.shuffle(inds)
        n_batches = len(inds) / self.batch_size
        theta = self.get_params()
        for epoch in range(self.epochs):
            loss = 0.
            for minibatch in range(n_batches):
                lossi, gradi = self.f_df(theta, X[inds[minibatch::n_batches]])
                theta -= self.learning_rate * gradi
                loss += lossi
            if verbose:
                print "Epoch %d, Loss = %.2f" % (epoch, loss / len(inds))
                sys.stdout.flush()
            if callback != None:
                callback(epoch)


def main():
    pass


if __name__ == '__main__':
    main()
