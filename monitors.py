import theano
import theano.tensor as T


def reconstruction_mse(stats, u):
    data = stats['data'][u]
    reconstruction = stats['model'][u]
    return T.mean((data - reconstruction) ** 2)

def reconstruction_error_rate(stats, u):
    data = stats['data'][u]
    reconstruction = stats['model'][u]
    return T.mean(T.neq(data, reconstruction))

def reconstruction_crossentropy(stats, u):
    data = stats['data'][u]
    reconstruction_activation = stats['model_activation'][u]
    return T.mean(T.sum(data*T.log(T.nnet.sigmoid(reconstruction_activation)) +
                  (1 - data)*T.log(1 - T.nnet.sigmoid(reconstruction_activation)), axis=1))
