import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from collections import namedtuple

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.data import tensor_grid
from dataset import Batch_Generator

# (x, t) in [-1, 1] x [0, 0.02]
domain = jnp.array([[0.0, 0.0, 0.0],
					[1.0, 0.02, 1.0]])
epsilon = 1e-4
sigma = 1.0

# initial conditions
rho0_fn = lambda x: 1+0.5*jnp.sin(jnp.pi*2*x)
T0_fn = lambda x: 0.25+0.1*jnp.cos(jnp.pi*2*x)
r0_fn = lambda x, t, v: rho0_fn(x)*(jnp.exp(-((v - 0.75)/T0_fn(x))**2) + jnp.exp(-((v+0.75)/T0_fn(x))**2))
j0_fn = lambda x, t, v: jnp.zeros_like(x)


# dataset for initial, boundary conditions, collocation points
dataset_Dirichlet = namedtuple("dataset_Dirichlet", ["x", "t", "v", "r", "j"])
dataset_Collocation = namedtuple("dataset_Collocation", ["x", "t", "v"])
dataset_Quadrature = namedtuple("dataset_Quadrature", ["nodes", "weights"])

def generate_dataset(n_i, n_cx, n_ct, n_quad):
	nodes, weights = np.polynomial.legendre.leggauss(n_quad)
	nodes = jnp.array(0.5*(nodes+1), dtype = jnp.float32).reshape((-1, 1))
	weights = jnp.array(0.5*weights, dtype = jnp.float32).reshape((-1, 1))

	x_i = jnp.linspace(*domain[:, 0], n_i).reshape((-1, 1))
	v_i = nodes
	xv_i = tensor_grid([x_i, v_i])
	x_i, v_i = xv_i[:, 0:1], xv_i[:, 1:2]
	t_i = jnp.zeros_like(x_i)
	r_i = r0_fn(x_i, t_i, v_i)
	j_i = j0_fn(x_i, t_i, v_i)

	x_c = jnp.linspace(*domain[:, 0], n_cx).reshape((-1, 1))
	t_c = jnp.linspace(*domain[:, 1], n_ct).reshape((-1, 1))
	v_c = nodes
	xtv_c = tensor_grid([x_c, t_c, v_c])

	dirichlet = dataset_Dirichlet(x_i, t_i, v_i, r_i, j_i)
	collocation = dataset_Collocation(xtv_c[:, 0:1], xtv_c[:, 1:2], xtv_c[:, 2:3])
	quad = dataset_Quadrature(nodes, weights)
	return dirichlet, collocation, quad

def generate_batch_fn(key, batch_size, dirichlet, collocation, quad, weights):
	subkeys = random.split(key, 2)
	Dirichlet = Batch_Generator(subkeys[0], dirichlet, batch_size["dirichlet"])
	Collocation = Batch_Generator(subkeys[1], collocation, batch_size["collocation"])
	
	def batch_fn(i):
		batch = {
			"dirichlet": dataset_Dirichlet(*next(Dirichlet)),
			"collocation": dataset_Collocation(*next(Collocation)),
			"quad": quad,
			"weights": weights,
		}
		return batch

	def evaluate_batch_fn(i):
		batch = {
			"dirichlet": dataset_Dirichlet(*Dirichlet.dataset),
			"collocation": dataset_Collocation(*Collocation.dataset),
			"quad": quad,
			"weights": weights,
		}
		return batch

	# return batch_fn, evaluate_batch_fn
	return batch_fn, batch_fn