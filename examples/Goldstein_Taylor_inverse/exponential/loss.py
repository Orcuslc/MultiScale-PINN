import jax
import jax.numpy as jnp
from jax import random

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.model_init import init_siren_params
from models import simple_model, normalized_model
from jaxmeta.loss import l1_regularization, l2_regularization
from jaxmeta.grad import jacobian_fn, hessian_fn

from data import domain, epsilon
import config

key, *subkeys = random.split(config.key, 3)
direct_params = init_siren_params(subkeys[0], config.direct_layers, config.direct_c0, config.direct_w0)
inverse_params = init_siren_params(subkeys[1], config.inverse_layers, config.inverse_c0, config.inverse_w0)

direct_model = normalized_model(domain)
jacobian = jacobian_fn(direct_model)

inverse_model = simple_model()
hessian_inv = hessian_fn(inverse_model)

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
		da_dxx = hessian_inv(inverse_params, collocation.x)[:, 0:1, 0]
		loss_c1 = config.metaloss(du_dt + dv_dx, 0)
		loss_c2 = config.metaloss(epsilon*dv_dt + a*du_dx, -v)
		loss_s = config.metaloss(da_dxx, 0)
	else:
		loss_c1 = loss_c2 = 0
		loss_s = 0

	if dirichlet[0] is not None:
		uv = direct_model(direct_params, jnp.hstack([dirichlet.x, dirichlet.t]))
		u, v = uv[:, 0:1], uv[:, 1:2]
		loss_d1 = config.metaloss(u, dirichlet.u)
		loss_d2 = 0
		# loss_d2 = config.metaloss(v, dirichlet.v)
	else:
		loss_d1 = loss_d2 = 0.0

	return loss_c1, loss_c2, loss_d1, loss_d2, loss_s

@jax.jit
def loss_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2, loss_s = loss_fn_(params, batch)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + w["s"]*loss_s + \
			l1_regularization(params[0], w["l1"]) + l1_regularization(params[1], w["l1"]) + \
			l2_regularization(params[0], w["l2"]) + l2_regularization(params[1], w["l2"])

@jax.jit
def evaluate_fn(params, batch):
	w = batch["weights"]
	loss_c1, loss_c2, loss_d1, loss_d2, loss_s = loss_fn_(params, batch)
	l1 = l1_regularization(params[0], 1.0) + l1_regularization(params[1], 1.0)
	l2 = l2_regularization(params[0], 1.0) + l2_regularization(params[1], 1.0)
	return w["c1"]*loss_c1 + w["c2"]*loss_c2 + w["d1"]*loss_d1 + w["d2"]*loss_d2 + w["l1"]*l1 + w["l2"]*l2 + w["s"]*loss_s, \
			loss_c1, loss_c2, loss_d1, loss_d2, loss_s, l1, l2
