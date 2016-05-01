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
import itertools, json

mode = None

# optimization parameters
learning_rate = 0.001
weight_decay = 0.02
epochs = 1

def rbm(threadName, data):
    n_visible = data.shape[1]
    n_hidden = 4

    # the rbm type
    rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)
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
    train = t.compile_function(initial_vmap, mb_size=1, monitors=[mse], name='train', mode=mode)

    # run for every sample point
    h,w = data.shape

    # costs for each thread
    costs_1 = []
    costs_2 = []
    costs_3 = []
    costs_4 = []
    # train the model
    for epoch in xrange(epochs):
        for i in xrange(h):
            # get cost for training the data point
            costs = [m for m in train({ rbm.v: data[i,:].reshape(1,n_visible) })]
            # print "MSE = %.4f, thread = %s" % (np.mean(costs), threadName)
            if threadName == "Thread-1":
                costs_1.append(np.mean(costs))
            elif threadName == "Thread-2":
                costs_2.append(np.mean(costs))
            elif threadName == "Thread-3":
                costs_3.append(np.mean(costs))
            else:
                costs_4.append(np.mean(costs))

    # all_costs = []
    # for i in xrange(h):
    #     # get cost for training the data point across all epochs
    #     cost_point = []
    #     for epoch in xrange(epochs):
    #         cost = [m for m in train({ rbm.v: data[i,:].reshape(1,40) })]
    #         print epoch, cost, threadName, i
    #         cost_point.append(cost)
    #
    #     all_costs.append(cost_point)


    return costs_1, costs_2, costs_3, costs_4


def rbm_star(a_b):
    # time.sleep(np.random.rand(1,1)*120)
    return rbm(*a_b)

def k_rbm(data):
    cost_dict = {}
    p = Pool(4)
    first_arg = ["Thread-1", "Thread-2", "Thread-3", "Thread-4"]
    second_arg = data
    a,b,c,d = p.map(rbm_star, itertools.izip(first_arg, itertools.repeat(second_arg)))
    # p.map(rbm_star, itertools.izip(first_arg, itertools.repeat(second_arg)))
    # get the costs from the tuples
    cost_1 = a[0]
    cost_2 = b[1]
    cost_3 = c[2]
    cost_4 = d[3]
    # find the cluster assignments
    for i in xrange(len(cost_1)):
        mincost = min(cost_1[i],cost_2[i],cost_3[i],cost_4[i])
        if mincost == cost_1[i]:
            cost_dict[i+1] = 1
        elif mincost == cost_2[i]:
            cost_dict[i+1] = 2
        elif mincost == cost_3[i]:
            cost_dict[i+1] = 3
        else:
            cost_dict[i+1] = 4

    json.dump(cost_dict, open("results1v4h.txt", 'w'))



if __name__ == "__main__":
    #dataset
    data = sio.loadmat('data/q2g.mat')['data']
    # run the k-rbm model
    k_rbm(data)
