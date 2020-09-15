import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.loss import *

# name of job
NAME = "1"

# random key
key = random.PRNGKey(1)

# network config
direct_layers = [1] + [32]*4 + [1] 
direct_c0 = 1.0
direct_w0 = jnp.array([[1.0]]).T

inverse_layers = [1] + [32]*4 + [1]
inverse_c0 = 1.0
inverse_w0 = jnp.array([[1.0]]).T

# network training
metaloss = mae
optimizer = optimizers.adam
lr = 1e-3
weights = {
	"c": 1.0,
	"d": 1.0,
	"s": 1e-8,
	"l1": 1e-8,
	"l2": 1e-8,
}
batch_size = {
	"dirichlet": 300,
	"collocation": 20100,
}
iterations = 200000
print_every = 1000
save_every = 10000
loss_names = ["Loss", "c", "d", "s", "l1_reg", "l2_reg"]
log_file = None


# data
n_data = {
	"b": 100,
	"c": 201
}