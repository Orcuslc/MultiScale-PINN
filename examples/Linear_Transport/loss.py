import jax
import jax.numpy as jnp
from jax import random

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import periodic_model_1d
from jaxmeta.loss import l1_regularization, l2_regularization
from jaxmeta.grad import jacobian_fn, hessian_fn
from jaxmeta.operator import quadrature_fn

from data import domain, epsilon, sigma
from config import metaloss

model, model_ = periodic_model_1d(domain, input_dim = 3)
jacobian = jacobian_fn(model_)
quadrature = quadrature_fn(model)

@jax.jit
def loss_fn_(params, batch):
	collocation, dirichlet, quad = batch["collocation"], batch["dirichlet"], batch["quad"]

	if collocation[0] is not None:
		rj = model(params, jnp.hstack([collocation.x, collocation.t, collocation.v]))
		r, j = rj[:, 0:1], rj[:, 1:2]
		drj_dxtv = jacobian(params, jnp.hstack([collocation.x, collocation.t, collocation.v]))
		dr_dt, dj_dt = drj_dxtv[:, 0:1, 1], drj_dxtv[:, 1:2, 1]
		dr_dx, dj_dx = drj_dxtv[:, 0:1, 0], drj_dxtv[:, 1:2, 0]
		rj_bar = quadrature(params, jnp.hstack([collocation.x, collocation.t]), quad.nodes, quad.weights)
		r_bar = rj_bar[0, :]
		loss_c1 = metaloss(epsilon**2*(dr_dt + collocation.v*dj_dx), sigma*(r_bar - r))
		loss_c2 = metaloss(epsilon**2*dj_dt + collocation.v*dr_dx, -sigma*j)
	else:
		loss_c1 = loss_c2 = 0

	if dirichlet[0] is not None:
		rj = model(params, jnp.hstack([dirichlet.x, dirichlet.t, dirichlet.v]))
		r, j = rj[:, 0:1], rj[:, 1:2]
		loss_d1 = metaloss(r, dirichlet.r)
		loss_d2 = metaloss(j, dirichlet.j)
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
