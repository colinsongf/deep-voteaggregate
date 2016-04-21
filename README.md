Morb: a modular RBM implementation in Theano
============================================

Morb is a toolbox for building and training Restricted Boltzmann Machine models in Theano. It is intended to be modular, so that a variety of different models can be built from their elementary parts. A second goal is for it to be extensible, so that new algorithms and techniques can be plugged in easily.


![Schematic diagram of Morb's RBM architecture](https://raw.githubusercontent.com/benanne/morb/master/architecture.png)

Example
-------

Below is a simple example, in which an RBM with binary visibles and binary hiddens is trained on an unspecified dataset using one-step contrastive divergence (CD-1), with some weight decay.

```python
import base, units, parameters, stats, updaters, trainers, monitors
import numpy
import theano.tensor as T

## define hyperparameters
learning_rate = 0.01
weight_decay = 0.02
minibatch_size = 32
epochs = 50

## load dataset
data = ...

## construct RBM model
rbm = base.RBM()

rbm.v = units.BinaryUnits(rbm) # visibles
rbm.h = units.BinaryUnits(rbm) # hiddens

rbm.W = parameters.ProdParameters(rbm, [rbm.v, rbm.h], initial_W) # weights
rbm.bv = parameters.BiasParameters(rbm, rbm.v, initial_bv) # visible bias
rbm.bh = parameters.BiasParameters(rbm, rbm.h, initial_bh) # hidden bias

## define a variable map, that maps the 'input' units to Theano variables.
initial_vmap = { rbm.v: T.matrix('v') }

## compute symbolic CD-1 statistics
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1)

## create an updater for each parameter variable
umap = {}
for variable in [rbm.W.W, rbm.bv.b, rbm.bh.b]:
    new_value = variable + learning_rate * (updaters.CDUpdater(rbm, variable, s) - decay * updaters.DecayUpdater(variable))
    umap[variable] = new_value

## monitor reconstruction cost during training
mse = monitors.reconstruction_mse(s, rbm.v)

## train the model
t = trainers.MinibatchTrainer(rbm, umap)
train = t.compile_function(initial_vmap, mb_size=minibatch_size, monitors=[mse])

for epoch in range(epochs):
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % numpy.mean(costs)
```
