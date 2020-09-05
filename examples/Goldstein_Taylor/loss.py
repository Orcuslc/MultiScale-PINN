import jax
import jax.numpy as jnp
from jax import random

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import normalized_model
from jaxmeta.loss import l1_regularization, l2_regularization
from jaxmeta.grad import jacobian_fn, hessian_fn

from data import domain, epsilon
from config import metaloss

model = normalized_model(domain)
jacobian = jacobian_fn(model)

@jax.jit
def loss_fn_(params, batch):
	collocation, dirichlet = batch["collocation"], batch["dirichlet"]

	if collocation[0] is not None:
		uv = model(params, jnp.hstack([collocation.x, collocation.t]))
		u, v = uv[:, 0:1], uv[:, 1:2]
		duv_dxt = jacobian(params, jnp.hstack([collocation.x, collocation.t]))
		du_dt, dv_dt = duv_dxt[:, 0:1, 1], duv_dxt[:, 1:2, 1]
		du_dx, dv_dx = duv_dxt[:, 0:1, 0], duv_dxt[:, 1:2, 0]
		loss_c1 = metaloss(du_dt + dv_dx, 0)
		loss_c2 = metaloss(epsilon*dv_dt + du_dx, -v)
	else:
		loss_c1 = loss_c2 = 0

	if dirichlet[0] is not None:
		uv = model(params, jnp.hstack([dirichlet.x, dirichlet.t]))
		u, v = uv[:, 0:1], uv[:, 1:2]
		loss_d1 = metaloss(u, dirichlet.u)
		loss_d2 = metaloss(v, dirichlet.v)
	else:
		loss_d1 = loss_d2 = 0.0

	return loss_c1, loss_c2, loss_d1, loss_d2

@jax.jit
def loss_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2 = loss_fn_(params, batch)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + \
			l1_regularization(params, w["l1"]) + l2_regularization(params, w["l2"])

@jax.jit
def evaluate_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2 = loss_fn_(params, batch)
	l1 = l1_regularization(params, 1.0)
	l2 = l2_regularization(params, 1.0)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + w["l1"]*l1 + w["l2"]*l2, \
			loss_c1, loss_c2, loss_d1, loss_d2, l1, l2
