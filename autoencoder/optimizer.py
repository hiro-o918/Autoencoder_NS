# -*- coding: utf-8 -*-
"""
Created on 2018/07/27 16:30

@author: Hironori Yamamoto
"""

import numpy as np

import autoencoder.functions as F
from autoencoder.network import AutoEncoder


class Optimizer(object):
    def __init__(self, model, lr, lr_decay=1, min_lr=1e-6, loss_function=F.least_squares_error, theta_function=F.sanger_rule):
        if not isinstance(model, AutoEncoder):
            msg = 'This optimizer is only aimed at {}'.format(AutoEncoder)
            raise Exception(msg)

        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.loss_function = loss_function
        self.theta_function = theta_function

    def update(self, iterator):
        lse = [self._update_core(x) for x in iterator]

        return lse

    def _update_core(self, x):
        # decode and encode
        z = self.model.encode(x)
        y = self.model.decode(z)
        # calculate updating terms
        xx = F.batch_matmul(x, x, transb=True)
        xxW = F.batch_matmul(xx, self.model.W, bcastb=True)
        Theta = self.theta_function(z)
        WTh = F.batch_matmul(self.model.W, Theta, bcasta=True)
        update_term = np.mean(xxW - WTh, axis=0)
        # update the model
        self.model.W += self.lr * update_term + self.lr ** 2

        return self.loss_function(x, y)

    def set_current_lr(self, epoch):
        if self.lr_decay is not None:
            self.lr = max(self.lr * self.lr_decay ** epoch, self.min_lr)

