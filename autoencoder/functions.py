# -*- coding: utf-8 -*-
"""
Created on 2018/07/28 0:08

@author: Hironori Yamamoto
"""

import numpy as np


def matmul(a, b, transa=False, transb=False, transout=False):
    if transout:
        transa, transb = not transb, not transa
        a, b = b, a
    if transa and a.ndim != 1:
        a = a.swapaxes(-1, -2)
    if transb and b.ndim != 1:
        b = b.swapaxes(-1, -2)

    if a.ndim <= 2 or b.ndim <= 2:
        return np.dot(a, b)
    else:
        return np.einsum('...ij,...jk->...ik', a, b)


def batch_matmul(a, b, transa=False, transb=False, transout=False,
                 bcasta=False, bcastb=False):
    if bcasta and bcastb:
        msg = 'The target for broadcasting must be decided between the two.'
        raise ValueError(msg)
    if bcasta:
        a = np.broadcast_to(a, (len(b), *a.shape))
    if bcastb:
        b = np.broadcast_to(b, (len(a), *b.shape))

    a = a.reshape(a.shape[:2] + (-1,))
    b = b.reshape(b.shape[:2] + (-1,))

    return np.squeeze(matmul(a, b, transa, transb, transout))


def sanger_rule(z):
    mask = np.array([[False if i >= j else True for i in range(z.shape[1])]
                     for j in range(z.shape[1])])
    zz = batch_matmul(z, z, transb=True)
    zz[np.broadcast_to(mask, (len(z), *mask.shape))] = 0

    return zz


def least_squares_error(x, y):
    return np.mean((x-y)**2, axis=0)
