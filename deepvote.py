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

from multiprocessing import Pool, freeze_support
import itertools
# import thread

mode = None

# optimization parameters
learning_rate = 0.001
weight_decay = 0.02
# minibatch_size = 8
epochs = 200

def rbm(threadName, data):
    n_visible = data.shape[1]
    n_hidden = 4

    # the rbm type
    rbm = rbms.GaussianBinaryRBM(n_visible, n_hidden)
    initial_vmap = { rbm.v: T.matrix('v') }

    # We use single-step contrastive divergence (CD-1) to train the RBM. For this, we can use
    # the CDUpdater. This requires symbolic CD-1 statistics:
    s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=2)

    # We create an updater for each parameter variable
    umap = {}
    for var in rbm.variables:
        # the learning rate is 0.001
        pu = var + learning_rate * updaters.CDUpdater(rbm, var, s)
        umap[var] = pu

    # training
    t = trainers.MinibatchTrainer(rbm, umap)
    mse = monitors.reconstruction_mse(s, rbm.v)
    train = t.compile_function(initial_vmap, mb_size=32, monitors=[mse], name='train', mode=mode)

    # # train the model
    for epoch in xrange(epochs):
        costs = [m for m in train({ rbm.v: data })]
        # print "Epoch %d, thread = %s" % (epoch, threadName)
        print "MSE = %.4f, thread = %s" % (np.mean(costs), threadName)
        # print "MSE = %s, thread = %s" % (costs, threadName)
        # print len(costs)

def rbm_star(a_b):
    # time.sleep(np.random.rand(1,1)*120)
    return rbm(*a_b)

def k_rbm(data):
    p = Pool(2)
    first_arg = ["Thread-1", "Thread-2"]
    second_arg = data
    p.map(rbm_star, itertools.izip(first_arg, itertools.repeat(second_arg)))


#dataset
data = sio.loadmat('data/dog_data.mat')['data']
# run the k-rbm model
k_rbm(data)
