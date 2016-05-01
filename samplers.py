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

def exponential(a):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples)

def truncated_exponential(a, maximum=1.0):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples*(1 - T.exp(-a * maximum)))

def truncated_exponential_mean(a, maximum=1.0):
    # return (1 / a) + (maximum / (1 - T.exp(maximum*a))) # this is very unstable around a=0, even for a=0.001 it's already problematic.
    # here is a version that switches depending on the magnitude of the input
    m_real = (1 / a) + (maximum / (1 - T.exp(maximum*a)))
    m_approx = 0.5 - (1./12)*a + (1./720)*a**3 - (1./30240)*a**5 # + (1./1209600)*a**7 # this extra term is unnecessary, it's accurate enough
    return T.switch(T.abs_(a) > 0.5, m_real, m_approx)
    
