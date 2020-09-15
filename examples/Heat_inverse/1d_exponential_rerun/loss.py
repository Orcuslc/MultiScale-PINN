import jax
import jax.numpy as jnp
from jax import random

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jaxmeta.model_init import init_siren_params
from models import simple_model, normalized_model, tanh_model
from jaxmeta.loss import l1_regularization, l2_regularization
from jaxmeta.grad import jacobian_fn, hessian_fn

from data import domain
import config

key, *subkeys = random.split(config.key, 3)
direct_params = init_siren_params(subkeys[0], config.direct_layers, config.direct_c0, config.direct_w0)
inverse_params = init_siren_params(subkeys[1], config.inverse_layers, config.inverse_c0, config.inverse_w0)

direct_model = normalized_model(domain)
jacobian_direct = jacobian_fn(direct_model)

inverse_model = simple_model()


# @jax.jit
# def inverse_model(params, x):
# 	return 1 + jnp.exp(-(x-0.5)**2)

hessian_inv = hessian_fn(inverse_model)


@jax.jit
def lhs_operator(params, x):
	direct_params, inverse_params = params
	a = inverse_model(inverse_params, x)
	dc_dx = jacobian_direct(direct_params, x)[:, 0]
	return a*dc_dx

# jacobian = jax.jit(jax.grad(lhs_operator, 1))
jacobian = jacobian_fn(lhs_operator)

@jax.jit
def rhs_operator(x):
	return (jnp.pi**2*jnp.sin(jnp.pi*x)+2*jnp.pi*(x-0.5)*jnp.cos(jnp.pi*x))*jnp.exp(-(x-0.5)**2) \
	+ jnp.pi**2*jnp.sin(jnp.pi*x)

params = [direct_params, inverse_params]

@jax.jit
def loss_fn_(params, batch):
	collocation, dirichlet = batch["collocation"], batch["dirichlet"]
	direct_params, inverse_params = params

	if collocation[0] is not None:
		lhs = jacobian(params, collocation.x)[:, 0]
		rhs = rhs_operator(collocation.x)
		da_dxx = hessian_inv(inverse_params, collocation.x)[:, 0:1, 0]
		loss_c = config.metaloss(lhs+rhs, 0)
		loss_s = config.metaloss(da_dxx, 0)
	else:
		loss_c = 0
		loss_s = 0

	if dirichlet[0] is not None:
		u = direct_model(direct_params, dirichlet.x)
		loss_d = config.metaloss(u, dirichlet.u)
	else:
		loss_d = 0

	return loss_c, loss_d, loss_s

@jax.jit
def loss_fn(params, batch):
	w = batch["weights"]
	loss_c, loss_d, loss_s = loss_fn_(params, batch)
	return w["c"]*loss_c + w["d"]*loss_d + w["s"]*loss_s + \
			l1_regularization(params[0], w["l1"]) + l1_regularization(params[1], w["l1"]) + \
			l2_regularization(params[0], w["l2"]) + l2_regularization(params[1], w["l2"])

@jax.jit
def evaluate_fn(params, batch):
	w = batch["weights"]
	loss_c, loss_d, loss_s = loss_fn_(params, batch)
	l1 = l1_regularization(params[0], 1.0) + l1_regularization(params[1], 1.0)
	l2 = l2_regularization(params[0], 1.0) + l2_regularization(params[1], 1.0)
	return w["c"]*loss_c + w["d"]*loss_d + w["s"]*loss_s + w["l1"]*l1 + w["l2"]*l2, \
			loss_c, loss_d, loss_s, l1, l2
