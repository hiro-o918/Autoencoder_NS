# -*- coding: utf-8 -*-
"""
Created on 2018/07/28 11:12

@author: Hironori Yamamoto
"""
import itertools
import pandas as pd
from sklearn.decomposition.pca import PCA
import numpy as np
import matplotlib.pyplot as plt


def pca_analysis(model, dataset, out):
    pca = PCA(n_components=len(dataset[0]))
    pca.fit(dataset)
    columns = ['W id', 'component vector id', 'dot']
    analysis_result = pd.DataFrame(columns=columns)\
        .astype({'W id': int, 'component vector id': int, 'dot': float})

    for n, (i, j) in enumerate(itertools.product(range(len(model.W.T)), range(len(pca.components_)))):
        analysis_result.loc[n] = [i, j, np.dot(model.W[:, i], pca.components_[j])]
    analysis_result.to_csv(out.joinpath('pca_analysis.csv'))

    plot_pca_analysis(analysis_result, out)


def plot_pca_analysis(analysis_result, out):
    fig, axes = plt.subplots(len(list(analysis_result.groupby('W id'))),
                             figsize=(4, 8), sharey=True)
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    for ax, (i, df) in zip(axes, analysis_result.groupby('W id')):
        ax.bar(x=df['component vector id']+1, height=np.abs(df['dot']), color=[249./255, 72./255, 117./255])
        ax.set_title('W index: {}'.format(i+1))

    fig.savefig(str(out.joinpath('analysis_result.pdf')))
