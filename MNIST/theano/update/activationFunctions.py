import theano
import theano.tensor as T


# def ReLU(x):
#     """
#     Rectify linear unit
#     """
#     return T.switch(x<0,0,x)

##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)

#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)