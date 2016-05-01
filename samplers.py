import theano
import theano.tensor as T

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # veel sneller
import numpy as np

numpy_rng = np.random.RandomState(123)
theano_rng = RandomStreams(numpy_rng.randint(2**30))

## samplers

def bernoulli(a):
    """ Return the bernoulli sample of the data having the given parameter(a)"""
    return theano_rng.binomial(size=a.shape, n=1, p=a, dtype=theano.config.floatX)

def gaussian(a, var=1.0):
    """ Return the gaussian sample of the data having the given mean(a) and variance(var)"""
    std = T.sqrt(var)
    return theano_rng.normal(size=a.shape, avg=a, std=std, dtype=theano.config.floatX)

def multinomial(a):
    # 0 = minibatches
    # 1 = units
    # 2 = states
    p = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))
    # r 0 = minibatches * units
    # r 1 = states
    # this is the expected input for theano.nnet.softmax and theano_rng.multinomial
    s = theano_rng.multinomial(n=1, pvals=p, dtype=theano.config.floatX)
    return s.reshape(a.shape) # reshape back to original shape
