from base import Parameters

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from misc import tensordot # better tensordot implementation that can be GPU accelerated
# tensordot = T.tensordot # use theano implementation

class FixedBiasParameters(Parameters):
    # Bias fixed at -1, which is useful for some energy functions (like Gaussian with fixed variance, Beta)
    def __init__(self, rbm, units, name=None):
        super(FixedBiasParameters, self).__init__(rbm, [units], name=name)
        self.variables = []
        self.u = units

        self.terms[self.u] = lambda vmap: T.constant(-1, theano.config.floatX) # T.constant is necessary so scan doesn't choke on it

    def energy_term(self, vmap):
        s = vmap[self.u]
        return T.sum(s, axis=range(1, s.ndim)) # NO minus sign! bias is -1 so this is canceled.
        # sum over all but the minibatch dimension.


class ProdParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]

        self.terms[self.vu] = lambda vmap: T.dot(vmap[self.hu], W.T)
        self.terms[self.hu] = lambda vmap: T.dot(vmap[self.vu], W)

        self.energy_gradients[self.var] = lambda vmap: vmap[self.vu].dimshuffle(0, 1, 'x') * vmap[self.hu].dimshuffle(0, 'x', 1)
        self.energy_gradient_sums[self.var] = lambda vmap: T.dot(vmap[self.vu].T, vmap[self.hu])

    def energy_term(self, vmap):
        return - T.sum(self.terms[self.hu](vmap) * vmap[self.hu], axis=1)
        # return - T.sum(T.dot(vmap[self.vu], self.var) * vmap[self.hu])
        # T.sum sums over the hiddens dimension.


class BiasParameters(Parameters):
    def __init__(self, rbm, units, b, name=None):
        super(BiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units

        self.terms[self.u] = lambda vmap: self.var

        self.energy_gradients[self.var] = lambda vmap: vmap[self.u]

    def energy_term(self, vmap):
        return - T.dot(vmap[self.u], self.var)
        # bias is NOT TRANSPOSED because it's a vector, and apparently vectors are COLUMN vectors by default.


class AdvancedProdParameters(Parameters):
    def __init__(self, rbm, units_list, dimensions_list, W, name=None):
        super(AdvancedProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]
        self.vd = dimensions_list[0]
        self.hd = dimensions_list[1]
        self.vard = self.vd + self.hd

        # there are vd visible dimensions and hd hidden dimensions, meaning that the weight matrix has
        # vd + hd = Wd dimensions.
        # the hiddens and visibles have hd+1 and vd+1 dimensions respectively, because the first dimension
        # is reserved for minibatches!
        self.terms[self.vu] = lambda vmap: tensordot(vmap[self.hu], W, axes=(range(1,self.hd+1),range(self.vd, self.vard)))
        self.terms[self.hu] = lambda vmap: tensordot(vmap[self.vu], W, axes=(range(1,self.vd+1),range(0, self.vd)))

        def gradient(vmap):
            v_indices = range(0, self.vd + 1) + (['x'] * self.hd)
            h_indices = [0] + (['x'] * self.vd) + range(1, self.hd + 1)
            v_reshaped = vmap[self.vu].dimshuffle(v_indices)
            h_reshaped = vmap[self.hu].dimshuffle(h_indices)
            return v_reshaped * h_reshaped

        self.energy_gradients[self.var] = gradient
        self.energy_gradient_sums[self.var] = lambda vmap: tensordot(vmap[self.vu], vmap[self.hu], axes=([0],[0]))
        # only sums out the minibatch dimension.

    def energy_term(self, vmap):
        # v_part = tensordot(vmap[self.vu], self.var, axes=(range(1, self.vd+1), range(0, self.vd)))
        v_part = self.terms[self.hu](vmap)
        neg_energy = tensordot(v_part, vmap[self.hu], axes=(range(1, self.hd+1), range(1, self.hd+1)))
        # we do not sum over the first dimension, which is reserved for minibatches!
        return - neg_energy # don't forget to flip the sign!


class AdvancedBiasParameters(Parameters):
    def __init__(self, rbm, units, dimensions, b, name=None):
        super(AdvancedBiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions

        self.terms[self.u] = lambda vmap: self.var

        self.energy_gradients[self.var] = lambda vmap: vmap[self.u]

    def energy_term(self, vmap):
        return - tensordot(vmap[self.u], self.var, axes=(range(1, self.ud+1), range(0, self.ud)))


class SharedBiasParameters(Parameters):
    """
    like AdvancedBiasParameters, but a given number of trailing dimensions are 'shared'.
    """
    def __init__(self, rbm, units, dimensions, shared_dimensions, b, name=None):
        super(SharedBiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions
        self.sd = shared_dimensions
        self.nd = self.ud - self.sd

        self.terms[self.u] = lambda vmap: T.shape_padright(self.var, self.sd)

        self.energy_gradients[self.var] = lambda vmap: T.mean(vmap[self.u], axis=self._shared_axes(vmap))

    def _shared_axes(self, vmap):
        d = vmap[self.u].ndim
        return range(d - self.sd, d)

    def energy_term(self, vmap):
        # b_padded = T.shape_padright(self.var, self.sd)
        # return - T.sum(tensordot(vmap[self.u], b_padded, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        # this does not work because tensordot cannot handle broadcastable dimensions.
        # instead, the dimensions of b_padded which are broadcastable should be summed out afterwards.
        # this comes down to the same thing. so:
        t = tensordot(vmap[self.u], self.var, axes=(range(1, self.nd+1), range(0, self.nd)))
        # now sum t over its trailing shared dimensions, which mimics broadcast + tensordot behaviour.
        axes = range(t.ndim - self.sd, t.ndim)
        return - T.sum(t, axis=axes)
