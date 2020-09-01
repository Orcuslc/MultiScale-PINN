import jax
import jax.numpy as jnp
from jax import random

def init_siren_layer_params(key, scale, fan_in, fan_out, dtype = jnp.float32):
	w_key, b_key = random.split(key)
	w = random.uniform(w_key, (fan_in, fan_out), dtype, minval = -scale, maxval = scale)
	b = jnp.zeros((fan_out, ), dtype)
	return w, b

def init_tanh_layer_params(key, fan_in, fan_out, initializer = jax.nn.initializers.glorot_normal, dtype = jnp.float32):
	w_key, b_key = random.split(key)
	w_init_fn = initializer()
	w = w_init_fn(w_key, (fan_in, fan_out), dtype)
	b = jnp.zeros((fan_out, ), dtype)
	return w, b

def init_siren_params(key, layers, c0, w0, dtype = jnp.float32):
	keys = random.split(key, len(layers))
	weights = [w0] + [1.0]*(len(layers)-2)
	params = [init_siren_layer_params(k, w*jnp.sqrt(c0/fi), fi, fo, dtype) for k, w, fi, fo in zip(keys, weights, layers[:-1], layers[1:])]
	return params

def init_tanh_params(key, layers, initializer = jax.nn.initializers.glorot_normal, dtype = jnp.float32):
	keys = random.split(key, len(layers))
	if not isinstance(initializer, list):
		initializer = [initializer]*(len(layers)-1)
	params = [init_tanh_layer_params(k, fi, fo, init, dtype) for k, fi, fo, init in zip(keys, layers[:-1], layers[1:], initializer)]
	return params