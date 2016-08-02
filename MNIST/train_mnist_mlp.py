import numpy as np
from sdeepy.edge.affine import Affine
from sdeepy.edge.activation import Relu, Tanh
from sdeepy.edge import Softmax
from sdeepy.core.network import Sequential
from sdeepy.utils.morph_net import *
from sdeepy.edge.convolution import Convolution, MaxPooling
import sys
from sdeepy.datasets.base import load_mnist
from sdeepy.edge.loss_function import CategoricalCrossentropy
from sdeepy.data_provider import DataProviderFromMemory
from sdeepy.optimize import GradientDescent
from sdeepy.support import solver
from sdeepy.core.save import save_net
from sdeepy.core.load import load_net
import os.path

from morph_test.morph_net_loader import *
from morph_test.utils_local import *


######## train mnist from scratch ##############
batch_size = 1000
base_lrate = 0.1
max_epoch = 100
h1 = 200
h2 = 150

net_file_name = os.path.dirname(__file__) + '/mnist_200_tanh.sdn'

#Load MNIST Data
#################
print("Load MNIST Data ...")
train_data, train_labels, num_train_samples, test_data, test_labels, num_test_samples = morph_net_loader()
inshape = train_data.shape[1]
outshape = train_labels.shape[1]

trainDP = DataProviderFromMemory([train_data, train_labels],
                                 batch_size=batch_size, shuffle=True, )

#Build inital net
##################################
if os.path.isfile(net_file_name):
    print("Load inital net ...")
    net = load_net(net_file_name)
    A1 = net.get_edge_linear_order()[0]
else:
    print("Build and Train inital net ...")
    A1 = Affine(inshape=(inshape,), outshape=(h1,), init_method='bengio2010_tanh')
    A2 = Affine(inshape=(h1,), outshape=(outshape,), init_method='bengio2010_tanh')
    edges = [A1, Tanh(), A2, Softmax()]
    net = Sequential(edges)

    opt = GradientDescent(net,
                          losses=CategoricalCrossentropy(),
                          data_provider=trainDP,
                          updater=GradientDescent.Updater(method='default',
                                                          base_lrate=base_lrate))

    costs = solver.train(opt,max_epoch = max_epoch)
    save_net(net, net_file_name)


# original net
orig_train_err, orig_train_cost = eval(net.forward([train_data])[0],train_labels)
orig_test_err, orig_test_cost = eval(net.forward([test_data])[0],test_labels)
print train_labels


######## Display results ##############
print "            || TRAIN COST  |  TRAIN ERROR | TEST ERROR  ||"
print "=========================================================="
print("Original     || {:.5f}     | {:.5f}     | {:.5f}     ||".format(orig_train_cost,orig_train_err*100,orig_test_err*100))
