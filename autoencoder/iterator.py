# -*- coding: utf-8 -*-
"""
Created on 2018/07/27 22:58

@author: Hironori Yamamoto
"""

import numpy as np


class DataIterator(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cur_point = 0
        self.is_new_epoch = False
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_new_epoch:
            self.reset()
            raise StopIteration

        try:
            batch = self.dataset[self.cur_point:self.cur_point + self.batch_size]
        except IndexError:
            batch = self.dataset[self.cur_point:]

        self.cur_point += self.batch_size
        if self.cur_point >= len(self.dataset):
            self.is_new_epoch = True

        return batch

    def reset(self):
        self.cur_point = 0
        self.is_new_epoch = False
        np.random.shuffle(self.dataset)
