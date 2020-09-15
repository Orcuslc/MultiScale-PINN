import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.loss import *

# name of job
NAME = "1"
PATH = "models"

# random key
key = random.PRNGKey(1)

# network config
layers = [4] + [32]*4 + [2] 
c0 = 1.0
w0 = jnp.array([[1.0, 1.0, 1.0, 1.0]]).T

# network training
metaloss = mae
optimizer = optimizers.adam
lr = 1e-3
weights = {
	"c1": 1.0,
	"c2": 10.0,
	"d1": 10.0,
	"d2": 10.0,
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
loss_names = ["Loss", "c1", "c2", "d1", "d2", "l1_reg", "l2_reg"]
log_file = None


# data
n_data = {
	"i": 100,
	"b": 100,
	"cx": 201, 
	"ct": 100,
	"quad": 2,
}