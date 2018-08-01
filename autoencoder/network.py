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
        z = F.batch_matmul(self.W, x, transa=True, bcasta=True)

        return z

    def decode(self, z):
        y = F.batch_matmul(self.W, z, bcasta=True)

        return y

    def reconstruction(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y

    def __call__(self, x):
        return self.reconstruction(x)

    def load_weight(self, file_path):
        self.W = np.load(file_path)
        print('Loading weight from {}'.format(file_path))