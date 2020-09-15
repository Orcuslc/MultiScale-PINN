import jax
import jax.numpy as jnp
from jax import random

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.model_init import init_siren_params, init_tanh_params
from models import simple_model, normalized_model, tanh_model
from jaxmeta.loss import l1_regularization, l2_regularization
from jaxmeta.grad import jacobian_fn, hessian_fn

from data import domain, epsilon
import config

key, *subkeys = random.split(config.key, 3)
direct_params = init_siren_params(subkeys[0], config.direct_layers, config.direct_c0, config.direct_w0)
inverse_params = jnp.array([2.0])

direct_model = normalized_model(domain)
jacobian = jacobian_fn(direct_model)

@jax.jit
def inverse_model(params, x):
	return 1 + params[0]/jnp.pi*jnp.cos(2*jnp.pi*x)

@jax.jit
def rhs(params, xt):
	direct_params, inverse_params = params
	duv_dxt = jacobian(direct_params, xt)
	du_dx = duv_dxt[:, 0]
	a = inverse_model(inverse_params, xt[0])
	return a*du_dx

rhs_jacobian = jacobian_fn(rhs)

params = [direct_params, inverse_params]

@jax.jit
def loss_fn_(params, batch):
	collocation, dirichlet = batch["collocation"], batch["dirichlet"]
	direct_params, inverse_params = params

	if collocation[0] is not None:
		uv = direct_model(direct_params, jnp.hstack([collocation.x, collocation.t]))
		a = inverse_model(inverse_params, collocation.x)
		u, v = uv[:, 0:1], uv[:, 1:2]
		duv_dxt = jacobian(direct_params, jnp.hstack([collocation.x, collocation.t]))
		du_dt, dv_dt = duv_dxt[:, 0:1, 1], duv_dxt[:, 1:2, 1]
		du_dx, dv_dx = duv_dxt[:, 0:1, 0], duv_dxt[:, 1:2, 0]
		rhs_grad = rhs_jacobian(params, jnp.hstack([collocation.x, collocation.t]))
		drhs_dx = rhs_grad[:, 0:1, 0]
		loss_c1 = config.metaloss(v, -a*du_dx)
		loss_c2 = config.metaloss(du_dt, drhs_dx)
	else:
		loss_c1 = loss_c2 = 0

	if dirichlet[0] is not None:
		uv = direct_model(direct_params, jnp.hstack([dirichlet.x, dirichlet.t]))
		u, v = uv[:, 0:1], uv[:, 1:2]
		loss_d1 = config.metaloss(u, dirichlet.u)
		# loss_d2 = config.metaloss(v, dirichlet.v)
		loss_d2 = 0
	else:
		loss_d1 = loss_d2 = 0.0

	return loss_c1, loss_c2, loss_d1, loss_d2

@jax.jit
def loss_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2 = loss_fn_(params, batch)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + \
			l1_regularization(params[0], w["l1"]) + \
			l2_regularization(params[0], w["l2"])

@jax.jit
def evaluate_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2 = loss_fn_(params, batch)
	l1 = l1_regularization(params[0], 1.0)
	l2 = l2_regularization(params[0], 1.0)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + w["l1"]*l1 + w["l2"]*l2, \
			loss_c1, loss_c2, loss_d1, loss_d2, l1, l2
