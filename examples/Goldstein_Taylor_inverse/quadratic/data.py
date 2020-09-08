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

# (x, t) in [-1, 1] x [0, 0.02]
domain = jnp.array([[-1.0, 0.0],
					[1.0, 0.02]])
epsilon = 1e-12

data_file = "problem2_3_snapshot_epsilon_1e-12.mat"

# initial conditions
u0_fn = lambda x, t: jnp.select([x <= 0, x > 0], 
								[2.0, 1.0])
v0_fn = lambda x, t: jnp.zeros_like(x)

# boundary conditions
ul_fn = ur_fn = u0_fn
vl_fn = vr_fn = v0_fn

# dataset for initial, boundary conditions, collocation points
dataset_Dirichlet = namedtuple("dataset_Dirichlet", ["x", "t", "u", "v"])
dataset_Collocation = namedtuple("dataset_Collocation", ["x", "t"])

def generate_dataset(n_i, n_b, n_cx, n_ct, n_dx, n_dt):
	x_i = jnp.linspace(*domain[:, 0], n_i).reshape((-1, 1))
	t_i = jnp.zeros_like(x_i)
	u_i = u0_fn(x_i, t_i)
	v_i = v0_fn(x_i, t_i)
	
	x_l = jnp.ones((n_b, 1))*domain[0, 0]
	x_r = jnp.ones((n_b, 1))*domain[1, 0]
	t_b = jnp.linspace(*domain[:, 1], n_b).reshape((-1, 1))
	u_l = ul_fn(x_l, t_b)
	u_r = ur_fn(x_r, t_b)
	v_l = vl_fn(x_l, t_b)
	v_r = vr_fn(x_r, t_b)

	x_c = jnp.linspace(*domain[:, 0], n_cx).reshape((-1, 1))
	t_c = jnp.linspace(*domain[:, 1], n_ct).reshape((-1, 1))
	xt_c = tensor_grid([x_c, t_c])

	data = loadmat(data_file)
	u_d, v_d, x_d, t_d = data["u_snapshots"], data["v_snapshots"], data["x"], data["t"].T
	u_d = u_d[1:-1:(len(x_d)//n_dx), 1:-1:(len(t_d)//n_dt)].reshape((-1, 1))
	v_d = v_d[1:-1:(len(x_d)//n_dx), 1:-1:(len(t_d)//n_dt)].reshape((-1, 1))
	x_d = x_d[1:-1:(len(x_d)//n_dx)]
	t_d = t_d[1:-1:(len(t_d)//n_dt)]
	xt_d = tensor_grid([x_d, t_d])

	dirichlet = dataset_Dirichlet(jnp.vstack([x_i, x_l, x_r, xt_d[:, 0:1]]), 
								jnp.vstack([t_i, t_b, t_b, xt_d[:, 1:2]]),
								jnp.vstack([u_i, u_l, u_r, u_d]),
								jnp.vstack([v_i, v_l, v_r, v_d]))
	collocation = dataset_Collocation(jnp.vstack([xt_c[:, 0:1], xt_d[:, 0:1]]), 
									jnp.vstack([xt_c[:, 1:2], xt_d[:, 1:2]]))
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
			# "collocation": dataset_Collocation(*[jnp.vstack([x, y]) for x, y in zip(d, c)]),
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