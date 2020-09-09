import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from collections import namedtuple

from scipy.io import loadmat

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.data import tensor_grid
from dataset import Batch_Generator

# x in [0, 2]
domain = jnp.array([[-1.0], [1.0]])

# analytic solution
u_fn = lambda x: jnp.sin(jnp.pi*x)

# dataset for dirichlet, collocation points
dataset_Dirichlet = namedtuple("dataset_Dirichlet", ["x", "u"])
dataset_Collocation = namedtuple("dataset_Collocation", ["x"])

def generate_dataset(n_b, n_c, n_d):
	x_b = domain*1.0
	u_b = u_fn(x_b)

	x_c = jnp.linspace(*domain[:, 0], n_c).reshape((-1, 1))

	x_d = jnp.linspace(*domain[:, 0], n_d).reshape((-1, 1))
	u_d = u_fn(x_d)

	dirichlet = dataset_Dirichlet(jnp.vstack([x_b, x_d]), jnp.vstack([u_b, u_d]))
	collocation = dataset_Collocation(jnp.vstack([x_b, x_c]))
	return dirichlet, collocation

def generate_batch_fn(key, batch_size, dirichlet, collocation, weights):
	subkeys = random.split(key, 2)
	Dirichlet = Batch_Generator(subkeys[0], dirichlet, batch_size["dirichlet"])
	Collocation = Batch_Generator(subkeys[1], collocation, batch_size["collocation"])
	
	def batch_fn(i):
		d = next(Dirichlet)
		c = next(Collocation)
		batch = {
			"dirichlet": dataset_Dirichlet(*d),
			"collocation": dataset_Collocation(*c),
			"weights": weights,
		}
		return batch

	def evaluate_batch_fn(i):
		batch = {
			"dirichlet": dataset_Dirichlet(*Dirichlet.dataset),
			"collocation": dataset_Collocation(*Collocation.dataset),
			"weights": weights,
		}
		return batch

	return batch_fn, evaluate_batch_fn