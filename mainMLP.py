"""
This Session is the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

# system libraries
import os
import sys
import timeit
import time


# third-party libraries
import numpy as np

import theano
import theano.tensor as T

# self-defined libraries
from mainLogisticReg import LogisticRegression
from loader import load_data
from hiddenLayer import HiddenLayer
from updateMethods import *


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, n_in, n_out, neurons, \
                 L1_reg=0.00, L2_reg=0.0001, learning_rate=0.01, \
                 n_epochs=1000, batch_size=20, name="mlp"):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.name = name

        # regularization tradeoff
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

        self.fcLayers=[]
        fcLayerNum=1
        print 'This is a MLP with %i layers'%(fcLayerNum)


        self.L1 = 0
        self.L2_sqr = 0


        currentInput=self.x
        currentInNum=n_in
        currentOutNum=neurons[0]

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        for i in xrange(fcLayerNum):
            currentLayer = HiddenLayer(
                rng=rng,
                input=currentInput,
                n_in=currentInNum,
                n_out=currentOutNum,
                activation=T.tanh
            )
            self.fcLayers.append(currentLayer)
            self.L1 += abs(currentLayer.W).sum()
            self.L2_sqr += (currentLayer.W ** 2).sum()
            currentInput=currentLayer.output
            if i!=fcLayerNum-1:
                currentInNum=neurons[i]
                currentOutNum=neurons[i+1]

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=currentInput,
            n_in=currentOutNum,
            n_out=n_out
        )

        self.L1 += abs(self.logRegressionLayer.W).sum()
        self.L2_sqr += (self.logRegressionLayer.W ** 2).sum()



        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors



        # the parameters of the model are the parameters of the two layer it is made out of
        self.params=self.logRegressionLayer.params
        for subNets in self.fcLayers:
            self.params+=subNets.params

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.cost = (
            self.negative_log_likelihood(self.y)
            + self.L1_reg * self.L1
            + self.L2_reg * self.L2_sqr
        )

        # update methods
        self.adadeltaUpdate=AdadeltaUpdate(self.params,self.cost)
        self.momentumUpdate=sgdMomentum(self.params,self.cost,self.learning_rate)

    def early_stop_training(self, train_set, valid_set, test_set):
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set['x'].get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = valid_set['x'].get_value(borrow=True).shape[0] / self.batch_size
        n_test_batches = test_set['x'].get_value(borrow=True).shape[0] / self.batch_size


        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(
            inputs=[self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: test_set['x'][self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.y: test_set['y'][self.index * self.batch_size:(self.index + 1) * self.batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: valid_set['x'][self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.y: valid_set['y'][self.index * self.batch_size:(self.index + 1) * self.batch_size]
            }
        )

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(self.cost, param) for param in self.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]


        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.momentumUpdate,  #updates,
            givens={
                self.x: train_set['x'][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set['y'][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))









def test_mlp(dataset='mnist.pkl.gz'):

    rng = np.random.RandomState(1234)

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set={}
    valid_set={}
    test_set={}
    train_set['x'] = train_set_x
    train_set['y'] = train_set_y
    valid_set['x'] = valid_set_x
    valid_set['y'] = valid_set_y
    test_set['x'] = test_set_x
    test_set['y'] = test_set_y


    # construct the MLP class
    classifier = MLP(
        rng, n_in=28*28, n_out=10, neurons=[500], \
                 L1_reg=0.00, L2_reg=0.0001, learning_rate=0.01, \
                 n_epochs=1000, batch_size=20, name="mlp"
    )
    classifier.early_stop_training(train_set, valid_set, test_set)





if __name__ == '__main__':
    test_mlp()