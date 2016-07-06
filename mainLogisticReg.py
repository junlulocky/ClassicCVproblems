"""
This session is logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

# system libraries
import cPickle

import os
import sys
import timeit


# third-party libraries
import numpy as np

import theano
import theano.tensor as T

# self-defined libraries
from loader import load_data


class LogisticRegression(object):
    """Multi-class Logistic Regression Classification

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, batch_size, learning_rate, n_epochs):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # allocate SYMBOLIC VARIABLE for the data
        self.index = T.lscalar()  # index to a [mini]batch

        # generate SYMBOLIC VARIABLES for input (x and y represent a minibatch)
        self.x = T.matrix('x')  # data, presented as rasterized images
        self.y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_value = np.zeros((n_in, n_out), dtype=theano.config.floatX )
        self.W = theano.shared(value=W_value, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        b_value = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_value, name='b', borrow=True)

        # SYMBOLIC EXPRESSION for computing the matrix of class-membership probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyperplane-k

        # probabilities of each class over the parameters
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        self.cost = self.negative_log_likelihood(self.y)



    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution. It is here used as the loss function to minimize

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with ...
        #     one row per example and one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        #     LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        #     the mean (across minibatch examples) of the elements in v,
        #     i.e., the mean log-likelihood across the minibatch....
        #     This allows for the learning rate choice to be less dependent of the minibatch size.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def early_stop_training(self, train_set, valid_set, test_set):
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set['x'].get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = valid_set['x'].get_value(borrow=True).shape[0] / self.batch_size
        n_test_batches = test_set['x'].get_value(borrow=True).shape[0] / self.batch_size

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        test_model = theano.function(
            inputs=[self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: test_set['x'][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: test_set['y'][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: valid_set['x'][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: valid_set['y'][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=self.cost, wrt=self.W)
        g_b = T.grad(cost=self.cost, wrt=self.b)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.W, self.W - self.learning_rate * g_W),
                   (self.b, self.b - self.learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=updates,
            givens={
                self.x: train_set['x'][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set['y'][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        # EARLY-STOPPING parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
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
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                       for i in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )

                        # save the best model
                        with open('best_model.pkl', 'w') as f:
                            cPickle.dump(self, f)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))


def test_logistic(dataset='mnist.pkl.gz'):

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


    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(n_in=28 * 28, n_out=10, batch_size=600, learning_rate=0.13, n_epochs=1000)
    classifier.early_stop_training(train_set, valid_set, test_set)


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.x],
        outputs=classifier.y_pred)

    # We can test it on some examples from test set
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


if __name__ == '__main__':
    test_logistic()
    #predict()