# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import theano

import sdeepy.utils.pylab as pl
from sdeepy.core.network import Sequential
from sdeepy.data_provider import DataProviderFromMemory
from sdeepy.edge.convolution import Convolution, MaxPooling
from sdeepy.edge.activation import Relu
from sdeepy.edge.affine import Affine
from sdeepy.edge.loss_function import CategoricalCrossentropy, ZeroOne
from sdeepy.edge.unclassified import Softmax
from sdeepy.monitor import Monitor
from sdeepy.support import solver
from sdeepy.optimize import AdamUpdater, GradientDescent
from sdeepy.core import save_net
from scipy.io import loadmat

if __name__ == '__main__':
    print "change44444..."
    save_path = os.path.dirname(__file__)+'/result_fpnet'
    batch_size = 128
    max_epoch = 200
    base_lrate = 1e-4

    # Load data, subtract mean
    data_train = loadmat(os.path.dirname(__file__)+'/SVHN_data/train_32x32.mat')
    data_extra = loadmat(os.path.dirname(__file__)+'/SVHN_data/extra_32x32.mat')
    data_test = loadmat(os.path.dirname(__file__)+'/SVHN_data/test_32x32.mat')

    # convert images into single precision and aligned memory    
    data_train['X'] = np.transpose(np.require(data_train['X'], dtype=theano.config.floatX, requirements=['ALIGNED']),
                                   axes=[3, 2, 0, 1])
    data_extra['X'] = np.transpose(np.require(data_extra['X'], dtype=theano.config.floatX, requirements=['ALIGNED']),
                                   axes=[3, 2, 0, 1])
    data_test['X'] = np.transpose(np.require(data_test['X'], dtype=theano.config.floatX, requirements=['ALIGNED']),
                                  axes=[3, 2, 0, 1])        

    # subtract mean and divide by scale
    data_train['X'] -= 127.5
    data_train['X'] /= 127.5

    data_extra['X'] -= 127.5
    data_extra['X'] /= 127.5

    data_test['X'] -= 127.5
    data_test['X'] /= 127.5

    # convert labels into one-hot encoding
    data_train['y_onehot'] = np.require(np.zeros((data_train['y'].size, 10)), theano.config.floatX, ['ALIGNED'])
    data_train['y_onehot'][np.arange(data_train['y'].size), data_train['y'].flatten()-1] = 1

    data_extra['y_onehot'] = np.require(np.zeros((data_extra['y'].size, 10)), theano.config.floatX, ['ALIGNED'])
    data_extra['y_onehot'][np.arange(data_extra['y'].size), data_extra['y'].flatten()-1] = 1

    data_test['y_onehot'] = np.require(np.zeros((data_test['y'].size, 10)), theano.config.floatX, ['ALIGNED'])
    data_test['y_onehot'][np.arange(data_test['y'].size), data_test['y'].flatten()-1] = 1

    trainDP = DataProviderFromMemory([np.concatenate((data_train['X'], data_extra['X'])),
                                      np.concatenate((data_train['y_onehot'], data_extra['y_onehot']))],
                                     batch_size, shuffle=True)
    validDP = DataProviderFromMemory([data_test['X'], data_test['y_onehot']], batch_size, shuffle=False)

    # Network configuration
    rng = np.random.RandomState()
    edges = [
        # 64C3-64C3-P2
        Convolution(inshape=(3, 32, 32), outmaps=64, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(64, 32, 32), outmaps=64, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),        
        MaxPooling(inshape=(64, 32, 32), pool_shape=(2, 2)),
        Relu(),
        
        # 128C3-128C3-P2
        Convolution(inshape=(64, 16, 16), outmaps=128, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(128, 16, 16), outmaps=128, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),        
        MaxPooling(inshape=(128, 16, 16), pool_shape=(2, 2)),
        Relu(),
        
        # 256C3-256C3-P2
        Convolution(inshape=(128, 8, 8), outmaps=256, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),
        Relu(),
        Convolution(inshape=(256, 8, 8), outmaps=256, kernel_shape=(3, 3,), border_mode='same',
                    with_bias=True, init_method='bengio2010_tanh', batch_size=batch_size, rng=rng),        
        MaxPooling(inshape=(256, 8, 8), pool_shape=(2, 2)),
        Relu(),

        # 1024FP-1024FP-10FP
        Affine(inshape=(256, 4, 4), outshape=(1024,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Relu(),
        Affine(inshape=(1024,), outshape=(1024,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Relu(),
        Affine(inshape=(1024,), outshape=(10,), with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Softmax()
    ]

    # Create convolutional neural net
    net = Sequential(edges, name='cnn')
    # Monitors
    monitors = [
        Monitor(
            net, ZeroOne(), trainDP, validDP, name='Error',
            popup_figure=False, save_path=save_path, monitor_condition=lambda epoch: True
        ),
    ]

    # Optimization for training
    opt = GradientDescent(net, losses=CategoricalCrossentropy(),
                          data_provider=trainDP, 
                          updater=AdamUpdater(alpha=base_lrate))

    print('start training!')
    solver.train(opt, monitors=monitors, max_epoch=max_epoch, save_path=save_path)
    save_net(net, save_path+'/SVHN_fpnet.sdn')
