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


class ThirdOrderParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ThirdOrderParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 3
        self.var = W
        self.variables = [self.var]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]

        def term_u0(vmap):
            p = tensordot(vmap[self.u1], W, axes=([1],[1])) # (mb, u0, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u0)
            # cannot use two tensordots here because of the minibatch dimension.

        def term_u1(vmap):
            p = tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u1)

        def term_u2(vmap):
            p = tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u1].dimshuffle(0, 1, 'x'), axis=1) # (mb, u2)

        self.terms[self.u0] = term_u0
        self.terms[self.u1] = term_u1
        self.terms[self.u2] = term_u2

        def gradient(vmap):
            p = vmap[self.u0].dimshuffle(0, 1, 'x') * vmap[self.u1].dimshuffle(0, 'x', 1) # (mb, u0, u1)
            p2 = p.dimshuffle(0, 1, 2, 'x') * vmap[self.u2].dimshuffle(0, 'x', 'x', 1) # (mb, u0, u1, u2)
            return p2

        self.energy_gradients[self.var] = gradient

    def energy_term(self, vmap):
        return - T.sum(self.terms[self.u1](vmap) * vmap[self.u1], axis=1)
        # sum is over the u1 dimension, not the minibatch dimension!




class ThirdOrderFactoredParameters(Parameters):
    """
    Factored 3rd order parameters, connecting three Units instances. Each factored
    parameter matrix has dimensions (units_size, num_factors).
    """
    def __init__(self, rbm, units_list, variables, name=None):
        super(ThirdOrderFactoredParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 3
        assert len(variables) == 3
        self.variables = variables
        self.var0 = variables[0]
        self.var1 = variables[1]
        self.var2 = variables[2]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]
        self.prod0 = lambda vmap: T.dot(vmap[self.u0], self.var0) # (mb, f)
        self.prod1 = lambda vmap: T.dot(vmap[self.u1], self.var1) # (mb, f)
        self.prod2 = lambda vmap: T.dot(vmap[self.u2], self.var2) # (mb, f)
        self.terms[self.u0] = lambda vmap: T.dot(self.prod1(vmap) * self.prod2(vmap), self.var0.T) # (mb, u0)
        self.terms[self.u1] = lambda vmap: T.dot(self.prod0(vmap) * self.prod2(vmap), self.var1.T) # (mb, u1)
        self.terms[self.u2] = lambda vmap: T.dot(self.prod0(vmap) * self.prod1(vmap), self.var2.T) # (mb, u2)

        # if the same parameter variable is used multiple times, the energy gradients should be added.
        # so we need a little bit of trickery here to make this work.
        energy_gradient_sums_list = [
            lambda vmap: T.dot(vmap[self.u0].T, self.prod1(vmap) * self.prod2(vmap)), # (u0, f)
            lambda vmap: T.dot(vmap[self.u1].T, self.prod0(vmap) * self.prod2(vmap)), # (u1, f)
            lambda vmap: T.dot(vmap[self.u2].T, self.prod0(vmap) * self.prod1(vmap)), # (u2, f)
        ] # the T.dot also sums out the minibatch dimension

        energy_gradient_sums_dict = {}
        for var, grad in zip(self.variables, energy_gradient_sums_list):
            if var not in energy_gradient_sums_dict:
                energy_gradient_sums_dict[var] = []
            energy_gradient_sums_dict[var].append(grad)

        for var, grad_list in energy_gradient_sums_dict.items():
            def tmp(): # create a closure, otherwise grad_list will always
                # refer to the one of the last iteration!
                # TODO: this is nasty, is there a cleaner way?
                g = grad_list
                self.energy_gradient_sums[var] = lambda vmap: sum(f(vmap) for f in g)
            tmp()

        # TODO: do the same for the gradient without summing!

    def energy_term(self, vmap):
        return - T.sum(self.terms[self.u1](vmap) * vmap[self.u1], axis=1)
        # sum is over the u1 dimension, not the minibatch dimension!




class TransformedParameters(Parameters):
    """
    Transform parameter variables, adapt gradients accordingly
    """
    def __init__(self, params, transforms, transform_gradients, name=None):
        """
        params: a Parameters instance for which variables should be transformed
        transforms: a dict mapping variables to their transforms
        gradients: a dict mapping variables to the gradient of their transforms

        IMPORTANT: the original Parameters instance should not be used afterwards
        as it will be removed from the RBM.

        ALSO IMPORTANT: because of the way the chain rule is applied, the old
        Parameters instance is expected to be linear in the variables.

        Example usage:
            rbm = RBM(...)
            h = Units(...)
            v = Units(...)
            var_W = theano.shared(...)
            W = ProdParameters(rbm, [u, v], var_W, name='W')
            W_tf = TransformedParameters(W, { var_W: T.exp(var_W) }, { var_W: T.exp(var_W) }, name='W_tf')
        """
        self.encapsulated_params = params
        self.transforms = transforms
        self.transform_gradients = transform_gradients

        # remove the old instance, this one will replace it
        params.rbm.remove_parameters(params)
        # TODO: it's a bit nasty that the old instance is first added to the RBM and then removed again.
        # maybe there is a way to prevent this? For example, giving the old parameters a 'dummy' RBM
        # like in the factor implementation. But then this dummy has to be initialised first...

        # initialise
        super(TransformedParameters, self).__init__(params.rbm, params.units_list, name)

        self.variables = params.variables
        for u, l in params.terms.items(): # in the terms, replace the vars by their transforms
            self.terms[u] = lambda vmap: theano.clone(l(vmap), transforms)

        for v, l in params.energy_gradients.items():
            self.energy_gradients[v] = lambda vmap: l(vmap) * transform_gradients[v] # chain rule

        for v, l in params.energy_gradient_sums.items():
            self.energy_gradient_sums[v] = lambda vmap: l(vmap) * transform_gradients[v] # chain rule

    def energy_term(self, vmap):
        old = self.encapsulated_params.energy_term(vmap)
        return theano.clone(old, self.transforms)
