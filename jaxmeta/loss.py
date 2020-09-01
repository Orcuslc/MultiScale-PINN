import jax
import jax.numpy as jnp

@jax.jit
def mse(pred, true):
	return jnp.mean(jnp.square(pred.reshape((-1, 1)) - true.reshape((-1, 1))))

@jax.jit
def mae(pred, true):
	return jnp.mean(jnp.abs(pred.reshape((-1, 1)) - true.reshape((-1, 1))))	

@jax.jit
def l2_regularization(params, w):
	return w*sum([jnp.sum(jnp.square(p[0])) for p in params])

@jax.jit
def l1_regularization(params, w):
	return w*sum([jnp.sum(jnp.abs(p[0])) for p in params])