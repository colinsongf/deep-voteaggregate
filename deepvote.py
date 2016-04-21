# rbm imports
import rbms, stats, updaters, trainers, updaters, monitors, units, parameters
from utils import generate_data, get_context

# theano imports
import theano
import theano.tensor as T
from theano import ProfileMode

# extra imports
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
plt.ion()

mode = None

# optimization parameters
learning_rate = 0.01
weight_decay = 0.02
minibatch_size = 8
epochs = 50

#dataset
data = sio.loadmat('q2g.mat')['data']

#construct the RBM model
n_visible = data.shape[1]
n_hidden = 4

rbm = rbms.GaussianBinaryRBM(n_visible, n_hidden)
initial_vmap = { rbm.v: T.matrix('v') }

# setting up the CD model. The parameters that can vary are k in CD-k or the mean-field parameters
# We use CDParamUpdater class for training the model
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=2)

# update each parameter
umap = {}
for var in rbm.variables:
    pu =  var + learning_rate * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

# set training parameters
t = trainers.MinibatchTrainer(rbm, umap)
mse = monitors.reconstruction_mse(s, rbm.v)
train = t.compile_function(initial_vmap, mb_size=minibatch_size, monitors=[mse], name='train', mode=mode)

# train the model
for epoch in xrange(epochs):
    print "Epoch %d" % epoch
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % np.mean(costs)
    # print rbm._get_hidden_units()
