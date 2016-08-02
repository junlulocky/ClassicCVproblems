
import os

import numpy as np

import sdeepy.utils.pylab as pl
from sdeepy.datasets.base import load_mnist
from sdeepy.core.network import Sequential
from sdeepy.data_provider import DataProvider
from sdeepy.data_provider import DataReaderFromMemory, DataReaderFromFiles
from sdeepy.edge.activation import Tanh
from sdeepy.edge.affine import Affine
from sdeepy.edge.convolution import Convolution, MaxPooling
from sdeepy.edge import Softmax
from sdeepy.edge.loss_function import CategoricalCrossentropy, ZeroOne
from sdeepy.monitor import Monitor, MonitorConvolution
from sdeepy.support import solver
from sdeepy.optimize import GradientDescent
from sdeepy.utils.process_label import label_to_onehot
from sdeepy.core.save import save_net
from sdeepy.core.load import load_net
from sdeepy.data_provider import DataProviderFromMemory
from morph_test.utils_local import *
from morph_test.morph_net_loader import *


# Settings
batch_size = 500
max_epoch = 100
lr = 0.01  # Learning rate
c1 = 16
c2 = 16

net_file_name = os.path.dirname(__file__) + '/mnist_lenet_{}_{}.sdn'.format(c1, c2)


## load data
train_x, train_y, num_train_samples, test_x, test_y, num_test_samples = morph_net_loader('mnist_cnn')
train_dp = DataProviderFromMemory([train_x, train_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])
test_dp = DataProviderFromMemory([test_x, test_y], batch_size=batch_size, shuffle=True, modal_names=['x', 'y'])

#Build inital net
##################################
if os.path.isfile(net_file_name):
    print("Load inital net ...")
    net = load_net(net_file_name)
    A1 = net.get_edge_linear_order()[0]
else:
    # Build network
    rng = np.random.RandomState(0)
    edges = [
        Convolution(
            inshape=(1, 28, 28), outmaps=16, kernel_shape=(5, 5,),
            with_bias=True, init_method='bengio2010_tanh',
            batch_size=None, rng=rng),
        MaxPooling(
            inshape=(16, 24, 24), pool_shape=(2, 2)),
        Tanh(),
        Convolution(
            inshape=(16, 12, 12), outmaps=16, kernel_shape=(5, 5,),
            with_bias=True, init_method='bengio2010_tanh',
            batch_size=None, rng=rng),
        MaxPooling(
            inshape=(16, 8, 8), pool_shape=(2, 2)),
        Tanh(),
        Affine(inshape=(16, 4, 4), outshape=(50,),
               with_bias=True, init_method='bengio2010_tanh', rng=rng),
        Tanh(),
        Affine(inshape=(50,), outshape=(10,),
               with_bias=True, init_method='bengio2010_sigmoid', rng=rng),
        Softmax(),  # Use Softmax rather than Sigmoid!
    ]
    net = Sequential(edges, 'lenet')


    # Drawing graph
    net.draw_graph()

    # Setting Monitor functions
    monitors = [
        Monitor(
            net, ZeroOne(),  # Use ZeroOne rather than BinaryZeroOne!
            train_dp.clone(), test_dp,
            name='Cost-error', popup_figure=False,
        ),
    ]

    # Learn
    opt = GradientDescent(net, losses=CategoricalCrossentropy(),
                          data_provider=train_dp,
                          updater=GradientDescent.Updater(
                              method='default', base_lrate=lr),)

    coat = solver.train(opt, monitors=monitors, max_epoch=max_epoch)
    save_net(net, net_file_name)

# original net
orig_train_err, orig_train_cost = eval(net.forward([train_x])[0],train_y)
orig_test_err, orig_test_cost = eval(net.forward([test_x])[0],test_y)




######## Display results ##############
print "            || TRAIN COST  |  TRAIN ERROR | TEST ERROR  ||"
print "=========================================================="
print("Original     || {:.5f}     | {:.5f}     | {:.5f}     ||".format(orig_train_cost,orig_train_err*100,orig_test_err*100))



