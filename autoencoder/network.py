# -*- coding: utf-8 -*-
"""
Created on 2018/07/27 16:14

@author: Hironori Yamamoto
"""

import numpy as np
import autoencoder.functions as F


class AutoEncoder(object):
    def __init__(self, n_in, n_out):
        self.W = np.random.rand(n_in, n_out)

    def encode(self, x):
        """
        :param x:
        :return:
        ```
            W^{T} x
        ```
        """
        z = F.batch_matmul(self.W, x, transa=True, bcasta=True)

        return z

    def decode(self, z):
        """
        :param z:
        :return:
        ```
             W(W^{\mathrm{T}}W)^{-1} z
        ```
        """
        h1 = np.linalg.inv(np.matmul(self.W.T, self.W))
        h2 = np.matmul(self.W, h1)
        y = F.batch_matmul(h2, z, bcasta=True)

        return y

    def reconstruction(self, x):
        """
        :param x:
        :return:
        ```
            W(W^{\mathrm{T}}W)^{-1}W^{\mathrm{T}} x
        ```
        """
        z = self.encode(x)
        y = self.decode(z)

        return y

    def __call__(self, x):
        return self.reconstruction(x)

    def load_weight(self, file_path):
        self.W = np.load(file_path)
        print('Loading weight from {}'.format(file_path))