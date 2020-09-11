import jax
import jax.numpy as jnp
from typing import List, Callable

def quadrature_fn(model: Callable[[List, jnp.array], jnp.array]) -> Callable:
	@jax.jit
	def quadrature(params: List, 
				   inputs: jnp.array, 
				   nodes: jnp.array, 
				   weights: jnp.array):
		N_nodes = nodes.shape[0]
		N_data = inputs.shape[0]
		inputs = jnp.repeat(inputs, N_nodes, axis = 0)
		nodes = jnp.tile(nodes, (N_data, 1))
		outputs = model(params, jnp.hstack([inputs, nodes]))
		outputs = outputs.T.reshape((outputs.shape[1], N_data, N_nodes))
		return outputs.dot(weights)
	return quadrature