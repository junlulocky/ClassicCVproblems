import theano
import theano.tensor as T


def ReLU(x):
    """
    Rectify linear unit
    """
    return T.switch(x<0,0,x)