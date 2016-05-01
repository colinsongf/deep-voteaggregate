from base import Units, ProxyUnits
import samplers, activation_functions
import theano.tensor as T
import numpy as np


class BinaryUnits(Units):
    def success_probability_from_activation(self, vmap):
        return activation_functions.sigmoid(vmap[self])

    def success_probability(self, vmap):
        return self.success_probability_from_activation({ self: self.activation(vmap) })

    def sample_from_activation(self, vmap):
        p = self.success_probability_from_activation(vmap)
        return samplers.bernoulli(p)

    def mean_field_from_activation(self, vmap):
        return activation_functions.sigmoid(vmap[self])

    def free_energy_term_from_activation(self, vmap):
        # softplus of unit activations, summed over # units
        s = - T.nnet.softplus(vmap[self])
        # sum over all but the minibatch dimension
        return T.sum(s, axis=range(1, s.ndim))

    def log_prob_from_activation(self, vmap, activation_vmap):
        # the log probability mass function is actually the  negative of the
        # cross entropy between the unit values and the activations
        p = self.success_probability_from_activation(activation_vmap)
        return vmap[self] * T.log(p) + (1 - vmap[self]) * T.log(1 - p)

class GaussianUnits(Units):
    def __init__(self, rbm, name=None):
        super(GaussianUnits, self).__init__(rbm, name)
        proxy_name = (name + "_precision" if name is not None else None)
        self.precision_units = GaussianPrecisionProxyUnits(rbm, self, name=proxy_name)
        self.proxy_units = [self.precision_units]

    def mean_from_activation(self, vmap): # mean is the parameter
        return vmap[self]

    def mean(self, vmap):
        return self.mean_from_activation({ self: self.activation(vmap) })

    def sample_from_activation(self, vmap):
        mu = self.mean_from_activation(vmap)
        return samplers.gaussian(mu)

    def mean_field_from_activation(self, vmap):
        return vmap[self]

    def log_prob_from_activation(self, vmap, activation_vmap):
        return - np.log(np.sqrt(2*np.pi)) - ((vmap[self] - activation_vmap[self])**2 / 2.0)


# TODO later: gaussian units with custom fixed variance (maybe per-unit). This probably requires two proxies.

class SoftmaxUnits(Units):
    # 0 = minibatches
    # 1 = units
    # 2 = states
    def probabilities_from_activation(self, vmap):
        return activation_functions.softmax(vmap[self])

    def probabilities(self, vmap):
        return self.probabilities_from_activation({ self: self.activation(vmap) })

    def sample_from_activation(self, vmap):
        p = self.probabilities_from_activation(vmap)
        return samplers.multinomial(p)


class SoftmaxWithZeroUnits(Units):
    """
    Like SoftmaxUnits, but in this case a zero state is possible, yielding N+1 possible states in total.
    """
    def probabilities_from_activation(self, vmap):
        return activation_functions.softmax_with_zero(vmap[self])

    def probabilities(self, vmap):
        return self.probabilities_from_activation({ self: self.activation(vmap) })

    def sample_from_activation(self, vmap):
        p0 = self.probabilities_from_activation(vmap)
        s0 = samplers.multinomial(p0)
        s = s0[:, :, :-1] # chop off the last state (zero state)
        return s
