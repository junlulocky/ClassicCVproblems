import numpy as np

import theano
import theano.tensor as T
from collections import defaultdict, OrderedDict

def as_floatX(variable):
    if isinstance(variable,float) or isinstance(variable,np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return T.cast(variable,theano.config.floatX)

def AdadeltaUpdate(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    Apply ada-delta updates

    :type params: tuple or list
    :para params: parameters

    :type cost:
    :para cost:

    :type rho: float
    :para rho:

    :type epsilon: float
    :para epsilon:

    :type norm_lim: int
    :para norm_lim:
    """
    updates=OrderedDict({})
    exp_sqr_grads=OrderedDict({})
    exp_sqr_update=OrderedDict({})
    g_params=[]
    for param in params:
        empty=np.zeros_like(param.get_value())
        exp_sqr_grads[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
        exp_sqr_update[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
        gp=T.grad(cost,param)
        g_params.append(gp)
    for param,gp in zip(params,g_params):
        exp_sg=exp_sqr_grads[param]
        exp_su=exp_sqr_update[param]
        update_exp_sg=rho*exp_sg+(1-rho)*T.sqr(gp)
        updates[exp_sg]=update_exp_sg

        step=-(T.sqrt(exp_su+epsilon)/T.sqrt(update_exp_sg+epsilon))*gp
        stepped_param=param+step

        update_exp_su=rho*exp_su+(1-rho)*T.sqr(step)
        updates[exp_su]=update_exp_su

        col_norms=T.sqrt(T.sum(T.sqr(stepped_param),axis=0))
        desired_norms=T.clip(col_norms,0,T.sqrt(norm_lim))
        scale=desired_norms/(1e-7+col_norms)
        updates[param]=stepped_param*scale
    return updates

def sgdMomentum(params, cost, learningRate, momentum=0.9):
    """
    SGD optimizer with momentum
    :type params: tuple or list
    :param params: parameters of the model

    :type cost: T.tensorType
    :param cost: goal to be optimized

    :type learningRate: float
    :param learningRate: learning rate

    :type momentum: float
    :param momentum: momentum weight
    """
    grads=T.grad(cost,params)
    updates=OrderedDict({})

    for param_i,grad_i in zip(params, grads):
        mparam_i=theano.shared(np.zeros(param_i.get_value().shape,dtype=theano.config.floatX),broadcastable=param_i.broadcastable)
        delta=momentum*mparam_i-learningRate*grad_i
        updates[mparam_i]=delta
        updates[param_i]=param_i+delta
    return updates

def sgdVanilla(params, cost, learning_rate):

    gparams = [T.grad(cost, param) for param in params]

    updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]
    return updates