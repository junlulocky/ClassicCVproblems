import numpy as np

import theano
import theano.tensor as T

# class HiddenLayer(object):
#     def __init__(self, rng, input, n_in, n_out,
#                  activation, W=None, b=None,
#                  use_bias=False):
#
#         self.input = input
#         self.activation = activation
#
#         if W is None:
#             W_values = np.asarray(0.01 * rng.standard_normal(
#                 size=(n_in, n_out)), dtype=theano.config.floatX)
#             W = theano.shared(value=W_values, name='W')
#
#         if b is None:
#             b_values = np.zeros((n_out,), dtype=theano.config.floatX)
#             b = theano.shared(value=b_values, name='b')
#
#         self.W = W
#         self.b = b
#
#         if use_bias:
#             lin_output = T.dot(input, self.W) + self.b
#         else:
#             lin_output = T.dot(input, self.W)
#
#         self.output = (lin_output if activation is None else activation(lin_output))
#
#         # parameters of the model
#         if use_bias:
#             self.params = [self.W, self.b]
#         else:
#             self.params = [self.W]

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, use_bias=True):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))

        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]



# def dropout(rng,value,p):
#     '''
#     :dropout function
#     :type rng: np.random.RandomState
#     :param rng: random seed
#
#     :type value: T.tensor4
#     :param value: input value
#
#     :type p: float
#     :param p: dropout rate
#     '''
#     srng=T.shared_randomstreams.RandomStreams(rng.randint(254860))
#     mask=srng.binomial(n=1,p=1-p,size=value.shape)
#     return value*T.cast(mask,theano.config.floatX)

# class DropoutHiddenLayer(HiddenLayer):
#     def __init__(self,rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, dropoutRate=0.1):
#         HiddenLayer.__init__(self,rng,input,n_in,n_out,activation)
#         self.dropoutRate=dropoutRate
#         self.output=dropout(rng,self.output,dropoutRate)

def _output_from_dropout(rng, layer, p):
    """
    :type rng: np.random.RandomState
    :param rng: random seed

    :type value: T.tensor4
    :param value: input value

    :type p: float
    :param p: dropout rate
    '''
    """
    srng = T.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _output_from_dropout(rng, self.output, p=dropout_rate)
