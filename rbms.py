from base import RBM
import units
import parameters

import theano
import theano.tensor as T

import numpy as np

class BinaryBinaryRBM(RBM):
    # the basic RBM, with binary visibles and binary hiddens
    def __init__(self, n_visible, n_hidden):
        super(BinaryBinaryRBM, self).__init__()
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.BinaryUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        self.W = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='W'), name='W') # weights
        self.bv = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bv(), name='bv'), name='bv') # visible bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bh(), name='bh'), name='bh') # hidden bias

    def _initial_W(self):
        return np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   high  =  4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   size  =  (self.n_visible, self.n_hidden)),
                   dtype =  theano.config.floatX)

    def _initial_bv(self):
        return np.zeros(self.n_visible, dtype = theano.config.floatX)

    def _initial_bh(self):
        return np.zeros(self.n_hidden, dtype = theano.config.floatX)


class GaussianBinaryRBM(RBM):
    # Gaussian visible units
    def __init__(self, n_visible, n_hidden):
        super(GaussianBinaryRBM, self).__init__()
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.GaussianUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        parameters.FixedBiasParameters(self, self.v.precision_units)
        self.W = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='W'), name='W') # weights
        self.bv = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bv(), name='bv'), name='bv') # visible bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bh(), name='bh'), name='bh') # hidden bias

    def _initial_W(self):
        return np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   high  =  4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   size  =  (self.n_visible, self.n_hidden)),
                   dtype =  theano.config.floatX)

    def _initial_bv(self):
        return np.zeros(self.n_visible, dtype = theano.config.floatX)

    def _initial_bh(self):
        return np.zeros(self.n_hidden, dtype = theano.config.floatX)
    
