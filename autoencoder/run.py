# -*- coding: utf-8 -*-
"""
Created on 2018/07/28 11:08

@author: Hironori Yamamoto
"""
import argparse
import json
import collections
import pathlib as Path
import numpy as np

from autoencoder.analysis import pca_analysis
from autoencoder.iterator import DataIterator
from autoencoder.network import AutoEncoder
from autoencoder.optimizer import Optimizer
from autoencoder.train import Trainer


def load_config(file_path):
    print('Loading configs from {}'.format(file_path))
    with open(file_path, 'r') as f:
        config = json.load(f)
    Config = collections.namedtuple('Config', config)

    return Config(**config)


def create_dataset(model_dir, config, file_name='dataset.npy'):
    if model_dir.joinpath(file_name).exists():
        dataset = np.load(model_dir.joinpath(file_name))
    else:
        dataset = np.random.multivariate_normal(*config.gauss_params, config.n_data)
        np.save(model_dir.joinpath(file_name), dataset)
        print('Saving dataset at {}'.format(model_dir.joinpath(file_name)))

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Parameters of autoencoder')
    parser.add_argument('--model_dir', type=Path.Path, default=Path.Path('result'))
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--do', type=str)
    args = parser.parse_args()

    model_dir = args.model_dir
    epoch = args.epoch
    do = args.do
    config = load_config(model_dir.joinpath('config.json'))
    np.random.seed(config.seed)
    dataset = create_dataset(model_dir, config)

    if do == 'train':
        iterator = DataIterator(dataset, batch_size=config.batch_size)
        model = AutoEncoder(n_in=config.n_in, n_out=config.n_out)
        optimizer = Optimizer(model, lr=config.lr, lr_decay=config.lr_decay, min_lr=config.min_lr)
        trainer = Trainer(epoch, optimizer, iterator, out=model_dir)
        trainer.run()

    if do == 'analysis':
        model = AutoEncoder(n_in=config.n_in, n_out=config.n_out)
        model.load_weight(model_dir.joinpath('weight.npy'))
        pca_analysis(model, dataset, model_dir)


if __name__ == '__main__':
    main()
