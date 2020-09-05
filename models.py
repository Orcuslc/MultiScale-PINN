import jax
import jax.numpy as jnp
from jax.ops import index, index_add, index_update

from functools import partial
from jaxmeta.data import normalize

def tanh_model():
	@jax.jit
	def model(params, x):
		for w, b in params[:-1]:
			x = jnp.tanh(jnp.dot(x, w) + b)
		return jnp.dot(x, params[-1][0]) + params[-1][1]
	return model

def simple_model():
	@jax.jit
	def model(params, x):
		for w, b in params[:-1]:
			x = jnp.sin(jnp.dot(x, w) + b)
		return jnp.dot(x, params[-1][0]) + params[-1][1]
	return model

def normalized_model(domain):
	@jax.jit
	def model(params, x):
		x = normalize(x, domain)
		for w, b in params[:-1]:
			x = jnp.sin(jnp.dot(x, w) + b)
		return jnp.dot(x, params[-1][0]) + params[-1][1]
	return model

def periodic_model_1d(domain, input_dim):
	m = jnp.hstack([jnp.array([1.] + [0.]*(input_dim-1)).reshape((-1, 1)), jnp.eye(input_dim)])

	@jax.jit
	def model(params, x): # for prediction
		x = normalize(x, domain)
		x = jnp.dot(x, m)
		x = index_update(x, index[:, 0], jnp.sin(jnp.pi*x[:, 0]))
		x = index_update(x, index[:, 1], jnp.cos(jnp.pi*x[:, 1]))
		for w, b in params[:-1]:
			x = jnp.sin(jnp.dot(x, w) + b)
		return jnp.dot(x, params[-1][0]) + params[-1][1]

	@jax.jit
	def model_(params, x): # for derivatives
		x = normalize(x, domain)
		x = jnp.dot(x, m)
		x = index_update(x, index[0], jnp.sin(jnp.pi*x[0]))
		x = index_update(x, index[1], jnp.cos(jnp.pi*x[1]))
		for w, b in params[:-1]:
			x = jnp.sin(jnp.dot(x, w) + b)
		return jnp.dot(x, params[-1][0]) + params[-1][1]

	return model, model_