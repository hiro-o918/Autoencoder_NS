# -*- coding: utf-8 -*-
"""
Created on 2018/07/27 22:34

@author: Hironori Yamamoto
"""

import numpy as np
import pandas as pd
import pathlib as Path
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, epoch, optimizer, iterator, out=Path.Path('result'), initial_epoch=0):
        self.epoch = epoch
        self.optimizer = optimizer
        self.iterator = iterator
        self.report = pd.DataFrame(columns=['epoch', 'lse'])
        self.report = self.report.set_index('epoch')
        self.out = out
        self.cur_epoch = initial_epoch
        self.min_loss = float('inf')

    def run(self):
        pre_loss = float('inf')
        self.cur_epoch = 0

        while self.cur_epoch != self.epoch:
            self.optimizer.set_current_lr(self.cur_epoch)
            loss = np.mean(self.optimizer.update(self.iterator))
            improvement = pre_loss - loss

            self.report.loc[self.cur_epoch] = loss
            self.save_reporter()

            print('epoch: {}/{}'.format(self.cur_epoch, self.epoch))
            print('loss: {}'.format(self.report.lse[self.cur_epoch]))
            print('current_lr: {}'.format(self.optimizer.lr))
            print('improvement: {}'.format(improvement))

            pre_loss = loss
            self.cur_epoch += 1

        self.save_model_weight()

    def save_reporter(self):
        self.report.plot(logy=True)
        plt.savefig(str(self.out.joinpath('loss.pdf')))
        self.report.to_csv(self.out.joinpath('log'))
        plt.close()

    def save_model_weight(self):
        np.save(self.out.joinpath('weight.npy'.format(self.cur_epoch)), self.optimizer.model.W)
