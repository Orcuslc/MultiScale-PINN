import jax
import jax.numpy as jnp
from jax import random

def jacfwd_fn(model):
	@jax.jit
	def jac_(params, inputs):
		return jax.jit(jax.vmap(jax.jacfwd(model, 1), in_axes = (None, 0)))(params, inputs)
	return jac_

def jacrev_fn(model):
	@jax.jit
	def jac_(params, inputs):
		return jax.jit(jax.vmap(jax.jacrev(model, 1), in_axes = (None, 0)))(params, inputs)
	return jac_

def hessian_fn(model):
	@jax.jit
	def hes_(params, inputs):
		return jax.jit(jax.vmap(jax.hessian(model, 1), in_axes = (None, 0)))(params, inputs)
	return hes_

jacobian_fn = jacrev_fn